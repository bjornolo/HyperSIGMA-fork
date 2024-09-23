"""generate testing mat dataset"""
import os
import numpy as np
import h5py
from os.path import join, exists
from scipy.io import loadmat, savemat

from util import crop_center, Visualize3D, minmax_normalize, rand_crop,BandMinMaxQuantileStateful
from PIL import Image
from skimage import io
import torch
data_path = '/home/lofty/CODE/HyperSIGMA-fork/ImageDenoising/data/HSI_Data/'  #change to datadir


def create_WDC_dataset():
    imgpath = data_path+'Hyperspectral_Project/HYPSO/virginiabeach_2024-08-22T14-59-41Z-l1a_updated.tif'
    imggt = io.imread(imgpath)
    imggt = torch.tensor(imggt, dtype=torch.float).permute(2, 0, 1)  # Change the order to (count, height, width)
    imgpath2 = data_path+'Hyperspectral_Project/dc.tif'
    imggt2 = io.imread(imgpath2)
    # imggt2 = torch.tensor(imggt2, dtype=torch.float).permute(2, 0, 1)  # Change the order to (count, height, width)
    print(f"The shape of imggt is: {imggt.shape}")
    print(f"The shape of imggt2 is: {imggt2.shape}")
    # Get the dimensions of the image
    channels, height, width = imggt.shape

    # Define the regions based on the dimensions
    test = imggt[:, height//2-100:height//2+100, width//4:width//4+200].clone()
    train_0 = imggt[:, :height//2-100, :].clone()
    train_1 = imggt[:, height//2+100:, :].clone()
    val = imggt[:, height//2-44:height//2+12, width//4+201:].clone()

    normalizer = BandMinMaxQuantileStateful()

    # fit train
    normalizer.fit([train_0, train_1])
    train_0 = normalizer.transform(train_0).cpu().numpy()
    train_1 = normalizer.transform(train_1).cpu().numpy()

    # fit test
    normalizer.fit([test])
    test = normalizer.transform(test).cpu().numpy()

    # val test
    normalizer.fit([val])
    val = normalizer.transform(val).cpu().numpy()

    # Create directories if they don't exist
    os.makedirs(data_path+"Hyperspectral_Project/HYPSO/train", exist_ok=True)
    os.makedirs(data_path+"Hyperspectral_Project/HYPSO/test", exist_ok=True)
    os.makedirs(data_path+"Hyperspectral_Project/HYPSO/val", exist_ok=True)

    savemat(data_path+"Hyperspectral_Project/HYPSO/train/train_0.mat", {'data': train_0})
    savemat(data_path+"Hyperspectral_Project/HYPSO/train/train_1.mat", {'data': train_1})
    savemat(data_path+"Hyperspectral_Project/HYPSO/test/test.mat", {'data': test})
    savemat(data_path+"Hyperspectral_Project/HYPSO/val/val.mat", {'data': val})


if __name__ == '__main__':
    create_WDC_dataset()

