modelname=$1 # spatsigma hypersigma
weight_path=$2

# python hsi_denoising_test.py -a $modelname -p hypersigma_gaussian -r -rp $weight_path --testdir  /mnt/code/users/yuchunmiao/SST-master/data/Hyperspectral_Project/WDC/test_noise/Patch_Cases/Case5  --basedir original_test --pretrain_path ./pre_train/spat-vit-base-ultra-checkpoint-1599.pth
python hsi_denoising_test.py -a $modelname -p hypersigma_gaussian -r -rp $weight_path --testdir  ./data/HSI_Data/Hyperspectral_Project/HYPSO2/test_noise/Patch_Cases/Case5  --basedir ./data/HSI_Data/Hyperspectral_Project/HYPSO2/results --pretrain_path /home/lofty/CODE/HyperSIGMA-fork/spat-base.pth
# python hsi_denoising_test.py -a $modelname -p hypersigma_gaussian -r -rp $weight_path --testdir  ./data/HSI_Data/Hyperspectral_Project/HYPSO2/train_noise/Patch_Cases/Case5  --basedir ./data/HSI_Data/Hyperspectral_Project/HYPSO2/results2 --pretrain_path /home/lofty/CODE/HyperSIGMA-fork/spat-base.pth