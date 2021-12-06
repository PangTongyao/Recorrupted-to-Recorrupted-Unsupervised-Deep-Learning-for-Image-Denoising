import os
import os.path
import numpy as np
import random
import h5py
import torch
import cv2
import glob
import torch.utils.data as udata
from utils import data_augmentation

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(2)

def normalize(data):
    return data/255.

def Im2Patch(img, win, stride=1):
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0:endw-win+0+1:stride, 0:endh-win+0+1:stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win*win,TotalPatNum], np.float32)
    for i in range(win):
        for j in range(win):
            patch = img[:,i:endw-win+i+1:stride,j:endh-win+j+1:stride]
            Y[:,k,:] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])
def Patch2Im(Y, win, img_size,stride=1):
    
    endc = img_size[0]
    endw = img_size[1]
    endh = img_size[2]
    Y = Y.reshape([endc,win*win,-1])
    img = np.zeros([endc,endw,endh],np.float32)
    weight = np.zeros([endc,endw,endh],np.float32)
    tempw = (endw-win) //stride +1
    temph = (endh-win) // stride +1
    k = 0
    for i in range(win):
        for j in range(win):
            img[:,i:endw-win+i+1:stride,j:endh-win+j+1:stride] = img[:,i:endw-win+i+1:stride,j:endh-win+j+1:stride] + Y[:,k,:].reshape(endc,tempw,temph)
            weight[:,i:endw-win+i+1:stride,j:endh-win+j+1:stride] = weight[:,i:endw-win+i+1:stride,j:endh-win+j+1:stride] + 1.
            k = k+1
    img = img/(weight+1e-6)
    return img
def prepare_data(data_path,sigma, patch_size, stride, aug_times=1):
    # train
    print('process training data')
    scales = [1, 0.9, 0.8, 0.7]
    files = glob.glob(os.path.join(data_path, 'train', '*.png'))
    files.sort()
    h5f = h5py.File('train_sigma_%s.h5'%(sigma), 'w')

    train_num = 0
    for i in range(len(files)):
        img = cv2.imread(files[i])
        img = np.float32(normalize(img))
        
        h, w, c = img.shape
        for k in range(len(scales)):
            Img = cv2.resize(img, (int(h*scales[k]), int(w*scales[k])), interpolation=cv2.INTER_CUBIC)
            Img = np.expand_dims(Img[:,:,0].copy(), 0)
            Img_noisy = Img + np.random.normal(0,sigma/255.,Img.shape)

            patches = Im2Patch(Img, win=patch_size, stride=stride)
            patches_noisy = Im2Patch(Img_noisy, win=patch_size, stride=stride)
            # patches_noisy2 = Im2Patch(Img_noisy2, win=patch_size, stride=stride)
            print("file: %s scale %.1f # samples: %d" % (files[i], scales[k], patches.shape[3]*aug_times))
            for n in range(patches.shape[3]):
                data = patches[:,:,:,n].copy()
                data_noisy = patches_noisy[:,:,:,n].copy()
 
                h5f.create_dataset(str(train_num), data=np.stack((data,data_noisy),axis=0))

                train_num += 1

                for m in range(aug_times-1):
                    data_aug = data_augmentation(data, np.random.randint(1,8))

                    data_noisy_aug = data_augmentation(data_noisy, np.random.randint(1,8))
                    
                    h5f.create_dataset(str(train_num)+"_aug_%d" % (m+1), data=np.stack((data_aug,data_noisy_aug),axis=0))

                    train_num += 1

    h5f.close()
    print('training set, # samples %d\n' % train_num)
#    # val
    print('\n process validation data')
    files.clear()
    files = glob.glob(os.path.join(data_path,  'bsd68/test*.png'))
    files.sort()
    h5f = h5py.File('val_%s_Set68.h5'%(sigma), 'w')
    val_num = 0
    for i in range(len(files)):
        print("file: %s" % files[i])
        img = cv2.imread(files[i])
        img = np.expand_dims(img[:,:,0], 0)
        img = np.float32(normalize(img))
        img_noisy = img + np.random.normal(0,sigma/255.,img.shape)
        h5f.create_dataset(str(val_num), data=np.stack((img,img_noisy),axis=0))
        val_num += 1
    h5f.close()
    
    print('val set, # samples %d\n' % val_num)

class Dataset_train(udata.Dataset):
    def __init__(self,  path='train'):
        super(Dataset_train, self).__init__()
        self.fullpath = path+'.h5'
        h5f = h5py.File(self.fullpath, 'r')

        self.keys = list(h5f.keys())
        random.shuffle(self.keys)
        h5f.close()
    def __len__(self):
        return len(self.keys)
    def __getitem__(self, index):

        h5f = h5py.File(self.fullpath, 'r')
        key = self.keys[index]
        data = np.array(h5f[key])
        clean = data[0]
        noisy = data[1]
        h5f.close()
        return  torch.Tensor(clean), torch.Tensor(noisy)

class Dataset_val(udata.Dataset):
    def __init__(self,  path='train'):
        super(Dataset_val, self).__init__()
        self.fullpath = path+'.h5'
        h5f = h5py.File(self.fullpath, 'r')

        self.keys = list(h5f.keys())
#        random.shuffle(self.keys)
        h5f.close()
    def __len__(self):
        return len(self.keys)
    def __getitem__(self, index):

        h5f = h5py.File(self.fullpath, 'r')
        key = self.keys[index]
        data = np.array(h5f[key])
        clean = data[0]
        noisy = data[1]
        h5f.close()
        return torch.Tensor(clean), torch.Tensor(noisy)
if __name__ == "__main__":
    prepare_data(data_path='./data/',sigma=25, patch_size=40, stride=10, aug_times=1)
 
