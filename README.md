
<div align="center">

<h1>HyperSIGMA: Hyperspectral Intelligence Comprehension Foundation Model</h1>


[Di Wang](https://dotwang.github.io/)<sup>1 ∗</sup>, [Meiqi Hu](https://meiqihu.github.io/)<sup>1 ∗</sup>, [Yao Jin](https://scholar.google.com/citations?hl=en&user=PBqyF80AAAAJ)<sup>1 ∗</sup>, [Yuchun Miao](https://scholar.google.com/citations?hl=en&user=-ec3mwUAAAAJ)<sup>∗</sup>, [Jiaqi Yang](https://jqyang22.github.io/)<sup>1 ∗</sup>, [Yichu Xu](https://scholar.google.com/citations?hl=en&user=CxKy4lEAAAAJ)<sup>1 ∗</sup>, Xiaolei Qin<sup>1 ∗</sup>, [Jiaqi Ma](https://leonmakise.github.io/)<sup>1 ∗</sup>, Lingyu Sun<sup>1 ∗</sup>, Chenxing Li<sup>1 ∗</sup>, Chuan Fu<sup>2</sup>, [Hongruixuan Chen](https://chrx97.com/)<sup>3</sup>, [Chengxi Han](https://chengxihan.github.io/)<sup>1 †</sup>, [Naoto Yokoya](https://naotoyokoya.com/)<sup>3</sup>, [Jing Zhang](https://scholar.google.com/citations?hl=en&user=9jH5v74AAAAJ&hl=en)<sup>1 †</sup>, Minqiang Xu<sup>4</sup>, Lin Liu<sup>4</sup>, [Lefei Zhang](https://scholar.google.com/citations?user=BLKHwNwAAAAJ&hl=en)<sup>1</sup>, [Chen Wu](https://scholar.google.com/citations?user=DbTt_CcAAAAJ&hl=en)<sup>1 †</sup>, [Bo Du](https://scholar.google.com/citations?user=Shy1gnMAAAAJ&hl=en)<sup>1 †</sup>, [Dacheng Tao](https://scholar.google.com/citations?user=RwlJNLcAAAAJ&hl=en)<sup>5</sup>, [Liangpei Zhang](https://scholar.google.com/citations?user=vzj2hcYAAAAJ&hl=en)<sup>1 †</sup>

<sup>1</sup> Wuhan University, <sup>2</sup> Chongqing University,  <sup>3</sup> The University of Tokyo, <sup>4</sup> National Engineering Research Center of Speech and Language Information Processing, <sup>5</sup> Nanyang Technological University.

<sup>∗</sup> Equal contribution, <sup>†</sup> Corresponding author


<!-- **Paper: ([arXiv 2404.03425](https://arxiv.org/pdf/2404.03425.pdf))**  -->
[![arXiv paper](https://img.shields.io/badge/arXiv-2404.03425-b31b1b.svg)](https://arxiv.org/pdf/2404.03425)

</div>

<p align="center">
  <a href="#-update">Update</a> |
  <a href="#-overview">Overview</a> |
  <a href="#-datasets">Datasets</a> |
  <a href="#-pretrained-models">Pretrained Models</a> |
  <a href="#-usage">Usage</a> |
  <a href="#-statement">Statement</a>
</p >

<figure>
<div align="center">
<img src=Fig/logo.png width="20%">
</div>
</figure>


# 🔥 Update

**2024.06.18**

- The paper is post on arxiv!

# 🌞 Overview

**HyperSIGMA** is the first billion-level foundation model specifically designed for HSI interpretation. To tackle the
spectral and spatial redundancy challenges in HSIs, we introduce a novel sparse sampling attention (SSA) mechanism, which effectively
promotes the learning of diverse contextual features and serves as the basic block of HyperSIGMA. HyperSIGMA integrates spatial and
spectral features using a specially designed spectral enhancement module.</a>


<figure>
<div align="center">
<img src=Fig/framework.png width="80%">
</div>

<div align='center'>
 
**Figure 1. Framework of HyperSIGMA.**

</div>
<br>


Extensive experiments on various high-level and low-level HSI tasks demonstrate HyperSIGMA’s versatility and superior representational capability compared to current state-of-the-art methods. It outperforms advanced models like SpectralGPT, even those specifically designed for these tasks.

<figure>
<div align="center">
<img src=Fig/radarimg.png width="80%">
</div>
</figure>

**Figure 2. HyperSIGMA demonstrates superior performance across 16 datasets and 7 tasks, including both high-level and low-level hyperspectral tasks, as well as multispectral scenes.** 



# 📖 Datasets
To train the foundational model, we collected hyperspectral remote sensing image samples from around the globe, constructing a large-scale hyperspectral dataset named **HyperGlobal-450K** for pre-training. **HyperGlobal-450K** contains over 20 million three-band images, far exceeding the scale of existing hyperspectral datasets.

<figure>
<div align="center">
<img src=Fig/dataset.png width="80%">
</div>
</figure>

**Figure 3. The distribution of HyperGlobal-450K samples across the globe, comprising 1,701 images (1,486 EO-1 and 215 GF-5B) with hundreds of spectral bands.**

# 🚀 Pretrained Models

| Pretrain | Backbone | Model Weights |
| :------- | :------: | :------ |
| Spatial_MAE | ViT-B | [Baidu](https://pan.baidu.com/s/1kShixCeWhPGde-vLLxQLJg?pwd=vruc)  | 
| Spatial_MAE | ViT-L |  [Baidu](https://pan.baidu.com/s/11iwHFh8sfg9S-inxOYtJlA?pwd=d2qs)  |
| Spatial_MAE | ViT-H | [Baidu](https://pan.baidu.com/s/1gV9A_XmTCBRw90zjSt90ZQ?pwd=knuu) | 
| Spectral_MAE | ViT-B |  [Baidu](https://pan.baidu.com/s/1VinBf4qnN98aa6z7TZ-ENQ?pwd=mi2y)  |
| Spectral_MAE | ViT-L | [Baidu](https://pan.baidu.com/s/1tF2rG-T_65QA3UaG4K9Lhg?pwd=xvdd) | 
| Spectral_MAE | ViT-H |  [Baidu](https://pan.baidu.com/s/1Di9ffWuzxPZUagBCU4Px2w?pwd=bi9r)|



# 🔨 Usage

## Pretraining

We pretrain the HyperSIGMA with SLURM. This is an example of pretraining the large version of Spatial ViT:

```
srun -J spatmae -p xahdnormal --gres=dcu:4 --ntasks=64 --ntasks-per-node=4 --cpus-per-task=8 --kill-on-bad-exit=1 \
python main_pretrain_Spat.py \
--model 'spat_mae_l' --norm_pix_loss \
--data_path [pretrain data path] \
--output_dir [model saved patch] \
--log_dir [log saved path] \
--blr 1.5e-4 --batch_size 32 --gpu_num 64 --port 60001
```

Another example of pretraining the huge version of Spectral ViT:

```
srun -J specmae -p xahdnormal --gres=dcu:4 --ntasks=128 --ntasks-per-node=4 --cpus-per-task=8 --kill-on-bad-exit=1 \
python main_pretrain_Spec.py \
--model 'spec_mae_h' --norm_pix_loss \
--data_path [pretrain data path] \
--output_dir [model saved patch] \
--log_dir [log saved path] \
--blr 1.5e-4 --batch_size 16 --gpu_num 128 --port 60004  --epochs 1600 --mask_ratio 0.75 \
--use_ckpt 'True'
```

The training can be recovered by setting `--resume`

```
--resume [path of saved model]
```

## Finetuning

**Image Classification**: 

Using the following command:
```
python scripts/image_classification.py【请修改】
```
**Target Detection & Anomaly Detection**: 

Step1: Preprare coarse detections.

Step2: Taking an example of performing target detection on the Mosaic dataset using HyperSIGMA:

```
CUDA_VISIBLE_DEVICES=0 python Target_Detection/trainval.py --dataset 'mosaic' --mode  'ss'
```

**Change Detection**: 

Using the following command:
```
python scripts/change_detection.py
```

**Spectral Unmixing**: 

Using the following command:
```
python scripts/unmixing.py【请修改】
```

**Denoising**: 

Please refer to [Denoising-README](https://github.com/WHU-Sigma/HyperSIGMA/blob/8eb6f6b386a45f944d133ce9e33550a4d79fe7ca/ImageDenoising/readme.md).


**Super-Resolution**: 

Please refer to [SR-README](https://github.com/WHU-Sigma/HyperSIGMA/blob/8eb6f6b386a45f944d133ce9e33550a4d79fe7ca/ImageSuperResolution/readme.md).


# ⭐ Citation

If you find HyperSIGMA helpful, please consider giving this repo a ⭐ and citing:

```

```

# 🎺 Statement

This project is for research purpose only. For any other questions please contact di.wang at [gmail.com](mailto:wd74108520@gmail.com) or [whu.edu.cn](mailto:d_wang@whu.edu.cn).


# 💖 Thanks
This project is based on [MAE](https://github.com/facebookresearch/mae), [Swin Transformer](https://github.com/microsoft/Swin-Transformer), [VSA](https://github.com/ViTAE-Transformer/ViTAE-VSA), [RVSA](https://github.com/ViTAE-Transformer/Remote-Sensing-RVSA), [DAT](https://github.com/LeapLabTHU/DAT), [HTD-IRN](https://github.com/shendb2022/HTD-IRN), [GT-HAD](https://github.com/jeline0110/GT-HAD) and [MSDformer](https://github.com/Tomchenshi/MSDformer). Thanks for their wonderful work!


