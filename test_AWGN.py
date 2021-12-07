import cv2
import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from models import DnCNN
from utils import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="DnCNN_Test")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--logdir", type=str, default="logs", help='path of log files')
parser.add_argument("--test_data", type=str, default='bsd68', help='test dataset')
parser.add_argument("--test_noiseL", type=float, default=25, help='noise level used on test set')
parser.add_argument("--alpha", type=float, default=0.5, help='R2R recorruption parameter')
parser.add_argument("--training", type=str, default="R2R", help='R2R or N2C')
opt = parser.parse_args()

def normalize(data):
    return data/255.

def main():
    # Build model
    print('Loading model ...\n')
    net = DnCNN(channels=1, num_of_layers=opt.num_of_layers)
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    model.load_state_dict(torch.load(os.path.join(opt.logdir, 'net.pth')))
    model.eval()
    # load data info
    print('Loading data info ...\n')
    files_source = glob.glob(os.path.join('data', opt.test_data, '*.png'))
    files_source.sort()
    # process data
    psnr_test = 0
    for f in files_source:
        # image
        Img = cv2.imread(f)
        Img = normalize(np.float32(Img[:,:,0]))
        Img = np.expand_dims(Img, 0)
        Img = np.expand_dims(Img, 1)
        ISource = torch.Tensor(Img)
        # noise
        noise = torch.FloatTensor(ISource.size()).normal_(mean=0, std=opt.test_noiseL/255.)
        # noisy image
        INoisy = ISource + noise
        ISource, INoisy = Variable(ISource.cuda()), Variable(INoisy.cuda())
        if opt.training == 'R2R':
            alpha = opt.alpha
            out_test = None
            aver_num = 50
            eps = opt.test_noiseL/255.
            for test_j in range(aver_num):
                INoisy_pert = INoisy + alpha*eps*torch.FloatTensor(INoisy.size()).normal_(mean=0, std=1.).cuda()
                with torch.no_grad():
                    out_test_single = model(INoisy_pert)
                if out_test  is None:
                    out_test= out_test_single.detach()
                else:
                    out_test += out_test_single.detach()
                del out_test_single
                
            out_test = torch.clamp(out_test/aver_num, 0., 1.)
            psnr = batch_PSNR(out_test, ISource, 1.)
            psnr_test += psnr
        else:
            with torch.no_grad(): # this can save much memory
                Out = torch.clamp(model(INoisy), 0., 1.)
        ## if you are using older version of PyTorch, torch.no_grad() may not be supported
        # ISource, INoisy = Variable(ISource.cuda(),volatile=True), Variable(INoisy.cuda(),volatile=True)
        # Out = torch.clamp(INoisy-model(INoisy), 0., 1.)
            psnr = batch_PSNR(Out, ISource, 1.)
            psnr_test += psnr
        print("%s PSNR %f" % (f, psnr))
    psnr_test /= len(files_source)
    print("\nPSNR on test data %f" % psnr_test)

if __name__ == "__main__":
    main()

