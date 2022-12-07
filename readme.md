# Recorrupted-to-Recorrupted: Unsupervised Deep Learning for Image Denoising

This repository is an PyTorch implementation of the paper [Recorrupted-to-Recorrupted: Unsupervised Deep Learning for Image Denoising](https://openaccess.thecvf.com/content/CVPR2021/html/Pang_Recorrupted-to-Recorrupted_Unsupervised_Deep_Learning_for_Image_Denoising_CVPR_2021_paper.html). The network we adopted is  [DnCNN](https://ieeexplore.ieee.org/document/7839189) and our implementation is based on [DnCNN-PyTorch](https://github.com/SaoYan/DnCNN-PyTorch). We give the author credit for the implementation of DnCNN.



## 1.Recorrupted-to-Recorrupted (R2R) Scheme

![image](https://user-images.githubusercontent.com/53853529/144895097-2f361e8e-317b-44ef-a45f-b64cf201a00a.png)


## 2.Experimental Results

Table 1. Quantitative comparison, in PSNR(dB)/SSIM, of different methods for AWGN removal on BSD68. The compared methods are categorized according to the type of training samples.  

![image](https://user-images.githubusercontent.com/53853529/144895572-98b2b2bb-79b7-4316-936b-d057491a68a7.png)




Table 2. Quantitative comparison, in PSNR(dB)/SSIM, of different non-learning and unsupervised methods for denoising real-world images from SIDD.  

![image](https://user-images.githubusercontent.com/53853529/144895637-4885b5b1-793c-4a7e-89e3-7a1fe56c7d48.png)


## 3.Implementations

### (1).Dependencies

- [PyTorch](http://pytorch.org/)(<0.4)
- [torchvision](https://github.com/pytorch/vision)
- OpenCV for Python
- [HDF5 for Python](http://www.h5py.org/)
- [tensorboardX](https://github.com/lanpa/tensorboard-pytorch) (TensorBoard for PyTorch)

### (2). Run Experiments on AWGN removal

Train R2R model for AWGN removal with noise level $\sigma =25$:

```
python3 train_AWGN.py --prepare_data --noiseL 25 --val_noiseL 25 --training R2R
```

### (3). The code for real world image denoising on SIDD dataset can be found [here](https://github.com/huanzheng551803/Real_R2R_denoising).
## 4.Citation

```
@InProceedings{Pang_2021_CVPR,
    author    = {Pang, Tongyao and Zheng, Huan and Quan, Yuhui and Ji, Hui},
    title     = {Recorrupted-to-Recorrupted: Unsupervised Deep Learning for Image Denoising},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {2043-2052}
}
```

