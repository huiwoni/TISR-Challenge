import numpy as np
import matplotlib.pyplot as plt
import argparse as arg

class make_graph:

    def __init__(self, input_path, loss_term, psnr_term, ssim_term, splite_term, loss_idx, psnr_idx, ssim_idx):

        self.loss_list = []
        self.psnr_list = []
        self.ssim_list = []
        self.input_path = input_path
        self.loss_term = loss_term.split(",")
        self.psnr_term = psnr_term.split(",")
        self.ssim_term = ssim_term.split(",")
        self.splite_term = splite_term
        self.loss_idx = loss_idx
        self.psnr_idx = psnr_idx
        self.ssim_idx = ssim_idx

        with open(input_path, 'r', encoding='utf-8') as in_file:

            line = in_file.readlines()

            self.info = line.copy()

        self.make_list()

    def find_term(self, term, target):

        length = len(term)
        count = 0

        for j in range(length):
            if term[j] in target:
                count += 1

        if count == length:
            return True

        else:
            return False


    def make_list(self):

        for i in (self.info):

            if self.find_term(self.loss_term, i):
                self.loss_list.append(i)
                continue

            if self.find_term(self.psnr_term, i):
                self.psnr_list.append(i)
                continue

            if self.find_term(self.ssim_term, i):
                self.ssim_list.append(i)

    def make_psnr_array(self, a_list):

        len_psnr = len(a_list)
        psnr_array = np.zeros(len_psnr)
        psnr_x = np.zeros(len_psnr)

        for idx, i in enumerate(a_list):
            n_psnr = i.split(self.splite_term)[self.psnr_idx]
            a = float(n_psnr[:7])
            psnr_array[idx] = a
            psnr_x[idx] = 5000 * (idx + 1)

        return psnr_array, psnr_x

    def make_ssim_array(self, a_list):

        len_ssim = len(a_list)
        ssim_array = np.zeros(len_ssim)
        ssim_x = np.zeros(len_ssim)

        for idx, i in enumerate(a_list):
            n_ssim = i.split(self.splite_term)[self.ssim_idx]
            a = float(n_ssim[:7])
            ssim_array[idx] = a
            ssim_x[idx] = 5000 * (idx + 1)

        return ssim_array, ssim_x

    def make_loss_array(self, a_list):

        len_loss = len(a_list)
        loss_array = np.zeros(len_loss)
        loss_x = np.zeros(len_loss)

        for idx, i in enumerate(a_list):
            n_loss = i.split(self.splite_term)[self.loss_idx]
            loss_array[idx] = float(n_loss)
            loss_x[idx] = 50 * (idx + 1)

        return loss_array, loss_x

    def show_graph(self):

        loss_array, loss_x = self.make_loss_array(self.loss_list)
        psnr_array, psnr_x = self.make_psnr_array(self.psnr_list)
        ssim_array, ssim_x = self.make_ssim_array(self.ssim_list)
        print(np.where(psnr_array==np.max(psnr_array)))
        plt.plot(loss_x, loss_array)
        plt.title('Loss')
        plt.show()

        plt.plot(psnr_x, psnr_array)
        plt.title('PSNR')
        plt.show()
        plt.plot(ssim_x, ssim_array)
        plt.title('SSIM')
        plt.show()

if __name__ == '__main__':

    parser = arg.ArgumentParser()
    parser.add_argument("--input_path", default = "./experiments/train_HAT_thermalSRx8_250000/train_train_HAT_thermalSRx8_20240125_004748.log")
    parser.add_argument("--loss_term", default = "train..")
    parser.add_argument("--psnr_term", default = "psnr,Best")
    parser.add_argument("--ssim_term", default = "ssim,Best")
    parser.add_argument("--splite_term", default=" ")
    parser.add_argument("--loss_idx", default = -2)
    parser.add_argument("--psnr_idx", default = 3)
    parser.add_argument("--ssim_idx", default = 3)

    args = parser.parse_args()

    go = make_graph(args.input_path, args.loss_term, args.psnr_term, args.ssim_term,
                    args.splite_term, args.loss_idx, args.psnr_idx, args.ssim_idx)

    go.show_graph()
