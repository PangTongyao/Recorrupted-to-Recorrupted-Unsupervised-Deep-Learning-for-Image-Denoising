import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from models import DnCNN
from dataset import prepare_data, Dataset_train, Dataset_val
from utils import *
from datetime import datetime

parser = argparse.ArgumentParser(description="DnCNN")
parser.add_argument("--prepare_data", action='store_true',  help='run prepare_data or not')
parser.add_argument("--batchSize", type=int, default=128, help="Training batch size")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=30, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--outf", type=str, default="logs", help='path of log files')
parser.add_argument("--noiseL", type=float, default=25, help='noise level; ignored when mode=B')
parser.add_argument("--alpha", type=float, default=0.5, help='alpha')

parser.add_argument("--val_noiseL", type=float, default=25, help='noise level used on validation set')
parser.add_argument("--gpu", type=int, default=0, help="gpu number")
parser.add_argument("--training", type=str, default="R2R", help='trainnig type')





def main():
    # Load dataset
    print('Loading dataset ...\n')
    

    sigma = opt.noiseL
    dataset_train = Dataset_train('train_sigma_%d' %sigma)
    dataset_val = Dataset_val('val_%d_Set68' %sigma)
    loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=opt.batchSize, shuffle=False)
    
    print("# of training samples: %d\n" % int(len(dataset_train)))

    # Build model
    net = DnCNN(channels=1, num_of_layers=opt.num_of_layers)
    net.apply(weights_init_kaiming)
    criterion = nn.MSELoss(size_average=False)
    # Move to GPU
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    criterion.cuda()
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    # training
    alpha = opt.alpha
    MODEL_PATH = opt.outf+"/logs/%s_%d/"%(opt.training,sigma)
    os.makedirs(MODEL_PATH, exist_ok=True)
   
    step = 0
    now = datetime.now()
    print('Start training.....',now.strftime("%H:%M:%S"))
    for epoch in range(opt.epochs):
        if epoch < opt.milestone:
            current_lr = opt.lr
        else:
            current_lr = opt.lr / 10.
        # set learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print('learning rate %f' % current_lr)
        # train
        for i, data in enumerate(loader_train, 0):
            # print(data[0].shape)
            clean = data[0]
            noisy = data[1]


            # training step
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            img_train = Variable(noisy.cuda())
            clean = Variable(clean.cuda())

            


            if opt.training == 'R2R':
                eps = sigma/255.
                pert = eps*torch.FloatTensor(img_train.size()).normal_(mean=0, std=1.).cuda()
                input_train = img_train + alpha*pert
                target_train = img_train - pert/alpha
            
            elif opt.training == 'N2C':
                input_train = img_train
                target_train = clean

                

            out_train = model(input_train)
            loss = criterion(out_train,target_train) / (target_train.size()[0]*2)


            loss.backward()
            optimizer.step()
            # results
            model.eval()

            out_train = torch.clamp(model(img_train), 0., 1.)

            psnr_train = batch_PSNR(out_train, clean, 1.)
            print("%s [epoch %d][%d/%d] loss: %.4f  PSNR: %.4f" %
                (opt.training,epoch+1, i+1, len(loader_train), loss.item(),psnr_train))
            # if you are using older version of PyTorch, you may need to change loss.item() to loss.data[0]

        ## the end of each epoch
        if opt.training == 'R2R':
            if (epoch+1) %1==0 or epoch == 0:
                model.eval()
                # validate
                psnr_val = 0
                psnr_val_pert = 0
                for k in range(len(dataset_val)):
                    img_val,imgn_val = dataset_val[k]
                    img_val = torch.unsqueeze(img_val,0)
                    imgn_val = torch.unsqueeze(imgn_val,0)
 
                    img_val, imgn_val = img_val.cuda(), imgn_val.cuda() 
   
                    out_val = None
                    aver_num = 50
                    eps = opt.val_noiseL/255.
                    for val_j in range(aver_num):
                        imgn_val_pert = imgn_val + alpha*eps*torch.FloatTensor(img_val.size()).normal_(mean=0, std=1.).cuda()
                        with torch.no_grad():
                            out_val_single = model(imgn_val_pert)
                        if out_val  is None:
                            out_val= out_val_single.detach()
                        else:
                            out_val += out_val_single.detach()
                        del out_val_single
                        
                    out_val = torch.clamp(out_val/aver_num, 0., 1.)
                    psnr_val_pert += batch_PSNR(out_val, img_val, 1.)
                    
                    with torch.no_grad():
                        out_val = torch.clamp(model(imgn_val),0.,1.)
                    psnr_val += batch_PSNR(out_val, img_val, 1.)
                psnr_val /= len(dataset_val)
                psnr_val_pert /= len(dataset_val)
                print("\n[epoch %d] PSNR_val: %.4f PNSR_val_pert: %.4f" % (epoch+1, psnr_val,psnr_val_pert))
        else:
             model.eval()
             # validate
             psnr_val = 0
             for k in range(len(dataset_val)):
                 img_val,imgn_val = dataset_val[k]
                 img_val = torch.unsqueeze(img_val,0)
                 imgn_val = torch.unsqueeze(imgn_val,0)
                 img_val, imgn_val = img_val.cuda(), imgn_val.cuda()
                 with torch.no_grad():
                     out_val = torch.clamp(model(imgn_val),0.,1.)                
                 psnr_val += batch_PSNR(out_val, img_val, 1.)
             psnr_val /= len(dataset_val)
             print("\n[epoch %d] PSNR_val: %.4f " % (epoch+1, psnr_val))
            
       
        torch.save(model.state_dict(), os.path.join(MODEL_PATH, 'net.pth'))
        now = datetime.now()
        print('Total training time.....',now.strftime("%H:%M:%S"))

        

if __name__ == "__main__":
    opt = parser.parse_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)
    if opt.prepare_data is True:
         prepare_data(data_path='./data/',sigma=25, patch_size=40, stride=10, aug_times=1)
    main()

