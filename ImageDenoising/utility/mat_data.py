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
import rasterio
data_path = '/home/lofty/CODE/HyperSIGMA-fork/ImageDenoising/data/HSI_Data/Hyperspectral_Project/GLORIA/'  #change to datadir
image_name = 'gloria_2024-09-24T08-30-46Z-l1a.tif'


def create_WDC_dataset():
    imgpath = data_path+image_name
    # imggt = io.imread(imgpath)
    with rasterio.open(imgpath) as dataset:
        imggt=dataset.read()
        print(f"The shape of imggt is: {imggt.shape}")

    imggt = torch.tensor(imggt, dtype=torch.float)
    # #SST original code
    # test = imggt[:, 600:800, 50:250].clone() 200x200
    # train_0 = imggt[:, :600, :].clone()
    # train_1 = imggt[:, 800:, :].clone()
    # val = imggt[:, 600:656, 251:].clone()56x56

    # #HYPSO values
    # test = imggt[:, 600:800, 50:250].clone() 200x200
    # train_0 = imggt[:, :600, :307].clone()
    # train_1 = imggt[:, 800:, :307].clone()
    # val = imggt[:, 600:656, 251:307].clone()

    #HYPSO FLAT values (598, 1092) GLORIA
    train_0 = imggt[:, :, :300].clone() 
    train_1 = imggt[:, :, 600:900].clone()
    test = imggt[:, 100:300, 350:550].clone() #200x200
    val = imggt[:, 300:356, 400:456].clone() #56x56


    
    ##Modified code
    # imgpath2 = data_path+'Hyperspectral_Project/WDC/dc.tif'
    # imggt2 = io.imread(imgpath2)
    # print(f"The shape of imggt is: {imggt.shape}")
    # # imggt=imggt.transpose(2,0,1)
    # # print(f"The shape of imggt transpose is: {imggt.shape}")
    # print(f"The shape of imggt2 is: {imggt2.shape}")
    # # Get the dimensions of the image
    # imggt = torch.tensor(imggt, dtype=torch.float).permute(2, 0, 1)  # Change the order to (count, height, width)
    # channels, height, width = imggt.shape
    # return

    # # Define the regions based on the dimensions
    # test = imggt[:, height//2-100:height//2+100, width//4:width//4+200].clone()
    # train_0 = imggt[:, :height//2-100, :].clone()
    # train_1 = imggt[:, height//2+100:, :].clone()
    # val = imggt[:, height//2-44:height//2+12, width//4+201:].clone()


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
    os.makedirs(data_path+"train", exist_ok=True)
    os.makedirs(data_path+"test", exist_ok=True)
    os.makedirs(data_path+"val", exist_ok=True)

    savemat(data_path+"train/train_0.mat", {'data': train_0})
    savemat(data_path+"train/train_1.mat", {'data': train_1})
    savemat(data_path+"test/test.mat", {'data': test})
    savemat(data_path+"val/val.mat", {'data': val})

def create_Hypso_dataset():
    imgpath = data_path+image_name
    # imggt = io.imread(imgpath)
    with rasterio.open(imgpath) as dataset:
        imggt=dataset.read()
        print(f"The shape of imggt is: {imggt.shape}")

    imggt = torch.tensor(imggt, dtype=torch.float)
 
    train_0 = imggt[:, :, :200].clone() 
    train_1 = imggt[:, :, 600:800].clone()
    test = imggt[:, 100:300, 350:550].clone() #200x200
    val = imggt[:, 300:356, 400:456].clone() #56x56

    # Simple normalization function
    def normalize(data):
        return data / 36855.0  # Normalize to [0, 1] range

    # Normalize train data
    train_0 = normalize(train_0).numpy()
    train_1 = normalize(train_1).numpy()

    # Normalize test data
    test = normalize(test).numpy()

    # Normalize validation data
    val = normalize(val).numpy()
    print(f"The shape of train_0 is: {train_0.shape}")
    print(f"The shape of train_1 is: {train_1.shape}")
    print(f"The shape of test is: {test.shape}")
    print(f"The shape of val is: {val.shape}")
    # Create directories if they don't exist
    os.makedirs(data_path+"train", exist_ok=True)
    os.makedirs(data_path+"test", exist_ok=True)
    os.makedirs(data_path+"val", exist_ok=True)

    savemat(data_path+"train/train_0.mat", {'data': train_0})
    savemat(data_path+"train/train_1.mat", {'data': train_1})
    savemat(data_path+"test/test.mat", {'data': test})
    savemat(data_path+"val/val.mat", {'data': val})


if __name__ == '__main__':
    # create_WDC_dataset()
    create_Hypso_dataset()

