import math
import torch
import torch.nn as nn
import numpy as np
import cv2
from skimage.measure.simple_metrics import compare_psnr

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        # nn.init.uniform(m.weight.data, 1.0, 0.02)
        m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
        nn.init.constant(m.bias.data, 0.0)

def batch_PSNR(img, imclean, data_range):
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    return (PSNR/Img.shape[0])

def data_augmentation(image, mode):
    out = np.transpose(image, (1,2,0))
    if mode == 0:
        # original
        out = out
    elif mode == 1:
        # flip up and down
        out = np.flipud(out)
    elif mode == 2:
        # rotate counterwise 90 degree
        out = np.rot90(out)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        out = np.rot90(out)
        out = np.flipud(out)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(out, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        out = np.rot90(out, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # rotate 270 degree
        out = np.rot90(out, k=3)
    elif mode == 7:
        # rotate 270 degree and flip
        out = np.rot90(out, k=3)
        out = np.flipud(out)
    return np.transpose(out, (2,0,1))

def data_aug(img, mode):

    if mode ==0:
        image=img
    elif mode == 1:
        image =torch.flip(img,[2])
    elif mode == 2:
        image= torch.flip(img,[3])
    elif mode == 3:
        image= torch.flip(img,[2,3])
    elif mode == 4:
        image = img.transpose(2,3)
    elif mode == 5:
        image = img.transpose(2,3)
        image = torch.flip(img,[2])
    elif mode == 6:
        image = img.transpose(2,3)
        image = torch.flip(img,[3])
    elif mode == 7:
        image = img.transpose(2,3)
        image = torch.flip(img,[2,3])
    return image
def load_img(filepath):
    img = cv2.imread(filepath,1)
    img = img.astype(np.float32)
    img = img/255.
    img = np.transpose(img,(2,0,1))
    return torch.Tensor(img).cuda()
