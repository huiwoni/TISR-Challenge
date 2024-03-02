import torch
from torch.nn import functional as F
from torchvision.transforms import functional as FF
from PIL import Image
from torchvision.transforms import transforms

from basicsr.utils.registry import MODEL_REGISTRY
from hat.archs.TDA_SR_model import TDA_SRModel
from basicsr.metrics import calculate_metric
from basicsr.utils import imwrite, tensor2img

import math
from tqdm import tqdm
from os import path as osp

@MODEL_REGISTRY.register()
class TDA_HATModel(TDA_SRModel):

    def pre_process(self):
        # pad to multiplication of window_size
        window_size = self.opt['network_g']['window_size']
        self.scale = self.opt.get('scale', 1)
        self.mod_pad_h, self.mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            self.mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            self.mod_pad_w = window_size - w % window_size
        self.img = F.pad(self.lq, (0, self.mod_pad_w, 0, self.mod_pad_h), 'reflect')

    def process(self):
        # model inference
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.img)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.img)
            # self.net_g.train()

    def tile_process(self):
        """It will first crop input images to tiles, and then process each tile.
        Finally, all the processed tiles are merged into one images.
        Modified from: https://github.com/ata4/esrgan-launcher
        """
        batch, channel, height, width = self.img.shape
        output_height = height * self.scale
        output_width = width * self.scale
        output_shape = (batch, channel, output_height, output_width)

        # start with black image
        self.output = self.img.new_zeros(output_shape)
        tiles_x = math.ceil(width / self.opt['tile']['tile_size'])
        tiles_y = math.ceil(height / self.opt['tile']['tile_size'])

        # loop over all tiles
        for y in range(tiles_y):
            for x in range(tiles_x):
                # extract tile from input image
                ofs_x = x * self.opt['tile']['tile_size']
                ofs_y = y * self.opt['tile']['tile_size']
                # input tile area on total image
                input_start_x = ofs_x
                input_end_x = min(ofs_x + self.opt['tile']['tile_size'], width)
                input_start_y = ofs_y
                input_end_y = min(ofs_y + self.opt['tile']['tile_size'], height)

                # input tile area on total image with padding
                input_start_x_pad = max(input_start_x - self.opt['tile']['tile_pad'], 0)
                input_end_x_pad = min(input_end_x + self.opt['tile']['tile_pad'], width)
                input_start_y_pad = max(input_start_y - self.opt['tile']['tile_pad'], 0)
                input_end_y_pad = min(input_end_y + self.opt['tile']['tile_pad'], height)

                # input tile dimensions
                input_tile_width = input_end_x - input_start_x
                input_tile_height = input_end_y - input_start_y
                tile_idx = y * tiles_x + x + 1
                input_tile = self.img[:, :, input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

                # upscale tile
                try:
                    if hasattr(self, 'net_g_ema'):
                        self.net_g_ema.eval()
                        with torch.no_grad():
                            output_tile = self.net_g_ema(input_tile)
                    else:
                        self.net_g.eval()
                        with torch.no_grad():
                            output_tile = self.net_g(input_tile)
                except RuntimeError as error:
                    print('Error', error)
                print(f'\tTile {tile_idx}/{tiles_x * tiles_y}')

                # output tile area on total image
                output_start_x = input_start_x * self.opt['scale']
                output_end_x = input_end_x * self.opt['scale']
                output_start_y = input_start_y * self.opt['scale']
                output_end_y = input_end_y * self.opt['scale']

                # output tile area without padding
                output_start_x_tile = (input_start_x - input_start_x_pad) * self.opt['scale']
                output_end_x_tile = output_start_x_tile + input_tile_width * self.opt['scale']
                output_start_y_tile = (input_start_y - input_start_y_pad) * self.opt['scale']
                output_end_y_tile = output_start_y_tile + input_tile_height * self.opt['scale']

                # put tile into output image
                self.output[:, :, output_start_y:output_end_y,
                            output_start_x:output_end_x] = output_tile[:, :, output_start_y_tile:output_end_y_tile,
                                                                       output_start_x_tile:output_end_x_tile]
    def post_process(self):
        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - self.mod_pad_h * self.scale, 0:w - self.mod_pad_w * self.scale]

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]

            #-----------------------------------------------------------------------------------------------------------
            # test time augmentation
            #-----------------------------------------------------------------------------------------------------------
            ############################################################################################################ test time augmentation start

            lr_image_dic = {}

            sr_image_dic = {}

            lr_image_dic['img_0'] = val_data['lq']
            lr_image_dic['img_90'] = torch.rot90(val_data['lq'], 1, [2, 3])
            lr_image_dic['img_180'] = FF.rotate(val_data['lq'], 180)
            lr_image_dic['img_270'] = torch.rot90(val_data['lq'], 3, [2, 3])


            lr_image_dic['img_lr_flip'] = FF.hflip(val_data['lq'])
            lr_image_dic['img_lr_flip_90'] = torch.rot90(lr_image_dic['img_lr_flip'], 1, [2, 3])
            lr_image_dic['img_lr_flip_180'] = FF.rotate(lr_image_dic['img_lr_flip'], 180)
            lr_image_dic['img_lr_flip_270'] = torch.rot90(lr_image_dic['img_lr_flip'], 3, [2, 3])

            ############################################################################################################ test time augmentation end

            # ------------------------------------------------------------------------------------
            # create SR image input
            # ------------------------------------------------------------------------------------
            ##################################################################################### for start

            for image_name, lr_image in lr_image_dic.items():

                self.feed_data(lr_image)

                self.pre_process()

                if 'tile' in self.opt:
                    self.tile_process()
                else:
                    self.process()
                self.post_process()

                #---------------------------------------------------
                # save in dict
                #---------------------------------------------------
                #################################################### save in dict start

                visuals = self.get_current_visuals()
                sr_image_dic[image_name] = visuals

                #################################################### save in dict end

            #################################################################################### for end

            # -----------------------------------------------------------------------------------------------
            # Orientation restoration
            # -----------------------------------------------------------------------------------------------
            ################################################################################################# Orientation restoration start

            sr_image_dic['img_0'] = sr_image_dic['img_0']['result']
            sr_image_dic['img_90'] = torch.rot90(sr_image_dic['img_90']['result'], 3, [2, 3])
            sr_image_dic['img_180'] = FF.rotate(sr_image_dic['img_180']['result'], 180)
            sr_image_dic['img_270'] = torch.rot90(sr_image_dic['img_270']['result'], 1, [2, 3])


            sr_image_dic['img_lr_flip'] = FF.hflip(sr_image_dic['img_lr_flip']['result'])
            sr_image_dic['img_lr_flip_90'] = FF.hflip(torch.rot90(sr_image_dic['img_lr_flip_90']['result'], 3, [2, 3]))
            sr_image_dic['img_lr_flip_180'] = FF.hflip(FF.rotate(sr_image_dic['img_lr_flip_180']['result'], 180))
            sr_image_dic['img_lr_flip_270'] = FF.hflip(torch.rot90(sr_image_dic['img_lr_flip_270']['result'], 1, [2, 3]))
            ################################################################################################# Orientation restoration end

            one_channel_result = torch.cat(list(sr_image_dic.values()), dim=1).mean(dim = 1)

            result = torch.cat((one_channel_result,one_channel_result, one_channel_result), dim=0)

            sr_img = tensor2img(result)

            ############################################################################################################ test time augmentation end




            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

#            if save_img:
#                if self.opt['is_train']:
#                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
#                                             f'{img_name}_{current_iter}.png')
#                else:
#                    if self.opt['val']['suffix']:
#                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
#                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
#                    else:
#                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
#                                                 f'{img_name}_{self.opt["name"]}.png')
#                imwrite(sr_img, save_img_path)

            #---------------------------------------------------------------------------------------
            # for bmp
            #---------------------------------------------------------------------------------------
            #########################################################################################
            if save_img=="true":
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                              f'{img_name}_{current_iter}.bmp')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                  f'{img_name}_{self.opt["val"]["suffix"]}.bmp')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                  f'{img_name}.bmp')
                imwrite(sr_img, save_img_path)
            #########################################################################################

            # ---------------------------------------------------------------------------------------
            # calculate metric
            # ---------------------------------------------------------------------------------------
            #########################################################################################
            score = {}

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    # self.metric_results[name] += calculate_metric(metric_data, opt_)
                # ---------------------------------------------------------------------------------------
                # calculate metric
                # ---------------------------------------------------------------------------------------
                #########################################################################################

                    score[name] = calculate_metric(metric_data, opt_)
                    self.metric_results[name] += score[name]
                #########################################################################################
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
            #########################################################################################

            # ---------------------------------------------------------------------------------------
            # for val and bmp
            # ---------------------------------------------------------------------------------------
            #########################################################################################
            if save_img=="PSNR":
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                                 f'{img_name}_{current_iter}.bmp')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.bmp')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}{score["psnr"]}__{score["ssim"]}.bmp')
                imwrite(sr_img, save_img_path)
            #########################################################################################



        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)
