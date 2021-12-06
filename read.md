# Recorrupted-to-Recorrupted: Unsupervised Deep Learning for Image Denoising

This repository is an PyTorch implementation of the paper [Recorrupted-to-Recorrupted: Unsupervised Deep Learning for Image Denoising](https://openaccess.thecvf.com/content/CVPR2021/html/Pang_Recorrupted-to-Recorrupted_Unsupervised_Deep_Learning_for_Image_Denoising_CVPR_2021_paper.html). The network we adopted is  [DnCNN](https://ieeexplore.ieee.org/document/7839189) and our implementation is based on [DnCNN-PyTorch](https://github.com/SaoYan/DnCNN-PyTorch). We give the author credit for the implementation of DnCNN.



## 1.Recorrupted-to-Recorrupted (R2R) Scheme

For each noisy image $y=x+n$, where $n\sim\mathcal{N}(0,\Sigma_x)$, R2R generates paired data $(\hat{y},\tilde{y})$ by
$$
\widehat{y}=y+Az,\widetilde{y}=y-Bz,
$$
where $z\sim \mathcal{N}(0,I)$ is independent from $n$ and $AB^T=\Sigma_x$.

**Theorem 1.** Let $f_\theta$ denote the network parameterized by $\theta$. It holds that 
$$
\mathbb{E}_{x,y,z}\|f_\theta(\hat y)-\tilde y\|^2=\mathbb{E}_{x,y,z}\|f_\theta(\hat y)-x\| + const.
$$
*Example 1.* $n$ is additive white Gaussian  noise (AWGN): $n\sim \mathcal{N}(0,\sigma^2I)$. In our experiments, we  generate R2R training samples by
$$
\hat y = y +\frac12 \sigma z, \tilde y = y - 2\sigma  z.
$$
*Example 2.* $n\sim\mathcal{N}(0,\Sigma_x)$ is real image noise, where $\Sigma_x = diag (\beta_1 x+\beta_2)$,  $\beta_1$ and $\beta_2$ are two scalars. In our experiments, we  generate R2R training samples by
$$
\hat y = y + 2\sqrt{\Sigma_x}z, \tilde y = y -\frac12 \sqrt{\Sigma_x}z.
$$
In practice, $\Sigma_x$ is unavailable and only $\beta_1$ and $\beta_2$ are provided. We approximate $\Sigma_x$ by $diag(\beta_1y+\beta_2)$. 

## 2.Experimental Results

Table 1. Quantitative comparison, in PSNR(dB)/SSIM, of different methods for AWGN removal on BSD68. The compared methods are categorized according to the type of training samples.  

![image-20211206215025443](C:\Users\pangt\AppData\Roaming\Typora\typora-user-images\image-20211206215025443.png)



Table 2. Quantitative comparison, in PSNR(dB)/SSIM, of different non-learning and unsupervised methods for denoising real-world images from SIDD.  

![image-20211206215236971](C:\Users\pangt\AppData\Roaming\Typora\typora-user-images\image-20211206215236971.png)

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
python3 train_AWGN.py --prepare_data --noiseL 25 --training R2R
```

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

