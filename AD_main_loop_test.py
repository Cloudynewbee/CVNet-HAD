from __future__ import print_function
import matplotlib.pyplot as plt
# matplotlib inline

import os
import numpy as np
import time
import scipy.io
import h5py
from sklearn.preprocessing import minmax_scale

from models.resnet import ResNet
from models.unet import UNet
from models.skip import ADNet
from models.autoencoder import AutoEncoder
from models.sspcab_torch import SSPCAB
import torch
import torch.optim
from torch.nn import functional as F
from PIL import Image

from utils.inpainting_utils import *
from utils import shuffle_utils
from utils.ssim_loss import SSIM, ssim
from showauc import getauc

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor


root_path = ".\data\\abu-beach\\abu-beach-4"
residual_root_path = "results/CV-param/pavia\\pavia_detection_"
background_root_path = "results/CV-param/pavia\\pavia_background_"

save_dir = 'results/CV-param/pavia/pavia_'

kernel_size = 1
dilatation = 2
lambda_ = 0.005

def main(kkkk, root_path, residual_path, background_path, kernel_size, dilatation, lambda_):
    kkk = kkkk
    # data input
    # **************************************************************************************************************
    # torch.cuda.empty_cache()

    # root_path = ".\data\\abu-urban\\abu-urban-1"
    # residual_root_path = ".\data\\abu-urban\\urban1_detection_"
    # background_root_path = ".\data\\abu-urban\\urban1_background_"

    # root_path = ".\data\\abu-beach\\abu-beach-4"
    # residual_root_path = ".\data\\abu-beach\\test\\beach4_detection_"
    # background_root_path = ".\data\\abu-beach\\test\\beach4_background_"

    # root_path = ".\data\\abu-beach\\abu-beach-4"
    # residual_root_path = ".\data\\abu-beach\\beach4_detection_9"
    # background_root_path = ".\data\\abu-beach\\beach4_background_9"

    # conv
    # root_path = ".\data\\abu-beach\\abu-beach-4"
    # residual_root_path = "results/none\\beach4_none_detection_1"
    # background_root_path = "results/none\\beach4_none_background_1"

    # root_path = ".\data\\abu-beach\\abu-beach-4-s"
    # residual_root_path = ".\data\\abu-beach\\beach4_detection_s_1"
    # background_root_path = ".\data\\abu-beach\\beach4_background_s_1"

    # root_path = ".\data\\abu-airport\\abu-airport-4"
    # residual_root_path = ".\data\\abu-airport\\airport4_detection_3"
    # background_root_path = ".\data\\abu-airport\\airport4_background_3"

    # root_path = ".\data\\sandiegocenter\\sandiego_center"
    # residual_root_path = ".\data\\sandiegocenter\\test-sandiegocenter\\sandiego_detection_t_"
    # background_root_path = ".\data\\sandiegocenter\\test-sandiegocenter\\sandiego_background_t_"

    # root_path = ".\data\\sandiegocenter\\sandiego_center"
    # residual_root_path = ".\data\\sandiegocenter\\sandiego_detection_16"
    # background_root_path = ".\data\\sandiegocenter\\sandiego_background_16"

    # conv
    # root_path = ".\data\\sandiegocenter\\sandiego_center"
    # residual_root_path = "results/CV-param\\sandiego_detection_"
    # background_root_path = "results/CV-param\\sandiego_background_"
    mask_root_path = "data/sandiegocenter/mask/mask_"
    val_residual_root_path = './data/sandiegocenter/val_cbam/sandiego_detection_'
    val_background_root_path = './data/sandiegocenter/val_cbam/sandiego_background_'

    # sandiego_n2v
    # root_path = ".\data\\sandiegocenter\\sandiego_n2v"
    # residual_root_path = "results/sandiego_center/sandiego_n2v_detection_1"
    # background_root_path = "results/sandiego_center/sandiego_n2v_background_1"

    # hydice_n2v
    # root_path = 'data/hydice/hudice_n2v'
    # residual_root_path = 'results/HYDICE1/hydice_n2v_detection_1'
    # background_root_path = 'results/HYDICE1/hydice_n2v_background_1'

    # pavia_n2v
    # root_path = 'data/abu-beach/pavia_n2v'
    # residual_root_path = 'results/pavia/pavia_n2v_detection_1'
    # background_root_path = 'results/pavia/pavia_n2v_background_1'

    # cutsandiego_n2v
    # root_path = 'data/sandiegocenter/cutsandiego_n2v'
    # residual_root_path = 'results/cutsandiego/cutsandiego_n2v_detection_1'
    # background_root_path = 'results/cutsandiego/cutsandiego_n2v_background_1'

    # root_path = ".\data\\sandiegocenter\\cutsandiego1"
    # residual_root_path = ".\data\\sandiegocenter\\test-cutsandiego\\cutsandiego_detection_"
    # background_root_path = ".\data\\sandiegocenter\\test-cutsandiego\\cutsandiego_background_"

    # root_path = ".\data\\sandiegocenter\\new_cutsandiego"
    # residual_root_path = ".\data\\sandiegocenter\\cutsandiego_detection_ssim_baseline_1"
    # background_root_path = ".\data\\sandiegocenter\\cutsandiego_background_ssim_baseline_1"

    # conv
    root_path = root_path
    residual_root_path = residual_path
    background_root_path = background_path

    root_path_l = "data/sandiegocenter/cutsandiego_l"
    root_path_s = "data/sandiegocenter/cutsandiego_s"

    thres = 0.000015
    channellss = 128
    layers = 5
    lambda_ = lambda_
    ratio = 0.6
    k = kernel_size
    d = dilatation
    sspcab = True

    file_name = root_path+".mat"
    # file_name_l = root_path_l+".mat"
    # file_name_s = root_path_s + ".mat"

    # mat = h5py.File(file_name)
    mat = scipy.io.loadmat(file_name)
    # mat_l = scipy.io.loadmat(file_name_l)
    # mat_s = scipy.io.loadmat(file_name_s)

    # img_h5 = mat["image"]
    img = mat['data'] # sandiego_center,airport,new_cutsandiego
    # img = mat['a']  # cutsandiego
    # img_l = mat_l['data']  # cutsandiego_l
    # img_s = mat_s['data']  # cutsandiego_s

    # img_np = np.array(img_h5)
    img_np = np.array(img)
    # img_np_l = np.array(img_l)
    # img_np_s = np.array(img_s)
    # img_np = img_np.transpose(0,2,1)


    row, col, band = img_np.shape
    # row_l, col_l, band_l = img_np_l.shape
    # row_s, col_s, band_s = img_np_s.shape

    img_np = minmax_scale(img_np.reshape(row * col, band)).reshape((row, col, band))
    # img_np_l = minmax_scale(img_np_l.reshape(row_l * col_l, band_l)).reshape((row_l, col_l, band_l))
    # img_np_s = minmax_scale(img_np_s.reshape(row_s * col_s, band_s)).reshape((row_s, col_s, band_s))

    img_np = img_np.transpose(2, 1, 0)
    # img_np = img_np.transpose(2, 0, 1)
    # img_np_l = img_np_l.transpose(2, 1, 0)
    # img_np_s = img_np_s.transpose(2, 1, 0)
    print(img_np.shape)
    # print(img_np_l.shape)
    # print(img_np_s.shape)
    img_var = torch.from_numpy(img_np).type(dtype)

    # img_size = img_var.size()
    # band = img_size[2]
    # row = img_size[1]
    # col = img_size[1]


    # model setup
    # **************************************************************************************************************
    # pad = 'reflection'  # 'zero'
    pad = 'zero'
    OPT_OVER = 'net'
    # OPTIMIZER = 'adam'
    method = '2D'
    input_depth = img_np.shape[0]
    LR = 0.01
    num_iter = 1001
    seed = 4
    param_noise = False
    reg_noise_std = 0.1 # 0 0.01 0.03 0.05
    net = ADNet(input_depth, img_np.shape[0],
               num_channels_down = [int(input_depth * ratio)] * layers,
               num_channels_up =   [int(input_depth * ratio)] * layers,
               num_channels_skip = [int(input_depth * ratio)] * layers,
               #  num_channels_down=[channellss] * layers,
               #  num_channels_up=[channellss] * layers,
               # num_channels_skip =    [128, 128, 128, 128, 128],
               filter_size_up = 3, filter_size_down = 5,
               need_upsample=False,
               upsample_mode='nearest', filter_skip_size=1,
               need_sigmoid=True, need_bias=True, pad=pad, act_fun='LeakyReLU', need1x1_up=True,
               sspcab=sspcab, kernel_dim=k, sspcab_stride=1, dilation=d, input_height=img_np.shape[1], input_weight=img_np.shape[2]).type(dtype)
    # net = UNet(input_depth, img_np.shape[0], upsample_mode='nearest',pad=pad).type(dtype)
    # net = ResNet(input_depth, img_np.shape[0], num_blocks=5, num_channels=channellss, pad=pad)
    # net = SSPCAB(input_depth)
    net = net.type(dtype) # see network structure
    # torch.manual_seed(kkk)
    net_input = get_noise(input_depth, method, img_np.shape[1:], seed=seed).type(dtype)
    # net_input = shuffle_utils.shuffleTensor(img_var.unsqueeze(0)).type(dtype)
    # print(net_input.shape)
    # net_input_l = get_noise(input_depth, method, img_np_l.shape[1:]).type(dtype)
    # net_input_s = get_noise(input_depth, method, img_np_s.shape[1:]).type(dtype)
    s  = sum(np.prod(list(p.size())) for p in net.parameters())
    print ('Number of params: %d' % s)
    # Loss
    mse = torch.nn.MSELoss().type(dtype)
    criterion = torch.nn.CrossEntropyLoss().type(dtype)
    ssim2 = SSIM(window_size=(2 * k + 2 * d + 1)).type(dtype)
    img_var = img_var[None, :].cuda()
    # print(img_var.shape)


    mask_var = torch.ones(1, band, row, col).cuda()
    residual_varr = torch.ones(row, col).cuda()

    def closure(iter_num, mask_varr, residual_varr):

        if param_noise:
            for n in [x for x in net.parameters() if len(x.size()) == 4]:
                n = n + n.detach().clone().normal_() * n.std() / 50

        net_input = net_input_saved
        # net_input_l = net_input_l_saved
        # net_input_s = net_input_s_saved
        if reg_noise_std > 0:
            net_input = net_input_saved + (noise.normal_() * reg_noise_std)
            # net_input_l = net_input_l_saved + (noise_l.normal_() * reg_noise_std)
            # net_input_s = net_input_s_saved + (noise_s.normal_() * reg_noise_std)

        # out = net(net_input_s, net_input, net_input_l)
        out = net(net_input)
        out_np = out.detach().cpu().squeeze().numpy()

        mask_var_clone = mask_varr.detach().clone()
        residual_var_clone = residual_varr.detach().clone()

        if iter_num % 200==0 and iter_num!=0:
        # if iter_num!=0:
            # weighting block
            img_var_clone = img_var.detach().clone()
            net_output_clone = out.detach().clone()
            temp = (net_output_clone[0, :] - img_var_clone[0, :]) * (net_output_clone[0, :] - img_var_clone[0, :])
            residual_img = temp.sum(0)

            residual_var_clone = residual_img
            r_max = residual_img.max()
            # residuals to weights
            residual_img = r_max - residual_img
            r_min, r_max = residual_img.min(), residual_img.max()
            residual_img = (residual_img - r_min) / (r_max - r_min)

            mask_size = mask_var_clone.size()
            for i in range(mask_size[1]):
                mask_var_clone[0, i, :] = residual_img[:]

            # scipy.io.savemat((mask_root_path + str(iter_num) + '.mat'), {'mask': mask_var_clone.detach().cpu().squeeze().numpy()})
            # scipy.io.savemat((val_residual_root_path + str(iter_num) + '.mat'), {'detection': residual_var_clone.detach().cpu().squeeze().numpy()})
            # scipy.io.savemat((val_background_root_path + str(iter_num) + '.mat'), {'detection': out_np.transpose(1, 2, 0)})

        # total_loss = lambda_ * (1 - ssim(out * mask_var_clone, img_var * mask_var_clone)) + mse(out * mask_var_clone, img_var * mask_var_clone)
        total_loss = mse(out * mask_var_clone, img_var * mask_var_clone) + lambda_ * (1 - ssim2(out * mask_var_clone, img_var * mask_var_clone))
        # total_loss = criterion(out * mask_var_clone, img_var * mask_var_clone)
        total_loss.backward()
        print("iteration: %d; loss: %f" % (iter_num+1, total_loss))

        return mask_var_clone, residual_var_clone, out_np, total_loss

    net_input_saved = net_input.detach().clone()
    # net_input_l_saved = net_input_l.detach().clone()
    # net_input_s_saved = net_input_s.detach().clone()
    noise = net_input.detach().clone()
    # noise_l = net_input_l.detach().clone()
    # noise_s = net_input_s.detach().clone()
    loss_np = np.zeros((1, 50), dtype=np.float32)
    loss_last = 0
    end_iter = False
    p = get_params(OPT_OVER, net, net_input)
    print('Starting optimization with ADAM')
    optimizer = torch.optim.Adam(p, lr=LR)
    for j in range(num_iter):
        optimizer.zero_grad()
        mask_var, residual_varr, background_img, loss = closure(j, mask_var, residual_varr)
        optimizer.step()

        if j >= 1:
            index = j-int(j/50)*50
            loss_np[0][index-1] = abs(loss-loss_last)
            if j % 50 == 0:
                mean_loss = np.mean(loss_np)
                if mean_loss < thres:
                    end_iter = True

        loss_last = loss

        if j == num_iter-1 or end_iter == True:
            residual_np = residual_varr.detach().cpu().squeeze().numpy()
            residual_path = residual_root_path + str(kkk) + str(kernel_size) + str(dilatation) + ".mat"
            scipy.io.savemat(residual_path, {'detection': residual_np})

            background_path = background_root_path + str(kkk) + str(kernel_size) + str(dilatation) + ".mat"
            scipy.io.savemat(background_path, {'detection': background_img.transpose(1, 2, 0)})
            return

if __name__ == "__main__":
    start = time.perf_counter()
    for kkkk in range(1, 2):
        main(kkkk, root_path, residual_root_path, background_root_path, kernel_size, dilatation, lambda_)
    end = time.perf_counter()
    print("runtime：", end-start)

    auc, auc_, detection, gt = getauc(root_path + '.mat', residual_root_path + '1' + str(kernel_size) + str(dilatation) + '.mat')
    print('auc:', auc)
    print('auc_:', auc_)
    img = plt.imshow(detection)
    # plt.title('auc=' + str(auc) + ',' + 'auc_=' + str(auc_))
    img.figure.savefig(save_dir + str(kernel_size) + str(dilatation) + '.jpg')

    plt.pause(5)





