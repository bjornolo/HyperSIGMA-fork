lr=$1
model=$2
gpu=$3
loss=$4
epoch=$5

batch_size=4
pretrain_path=/home/lofty/CODE/HyperSIGMA-fork/spat-base.pth
file=$(basename $pretrain_path .pth)
output=./output/original_${model}_${lr}_${file}_batch${batch_size}_warmup_${loss}_epoch_${epoch}_gaussian_new_fusion
mkdir ${output}
CUDA_VISIBLE_DEVICES=$gpu python hsi_denoising_gaussian_wdc.py -a $model -p hypersigma_gaussian -b ${batch_size} --training_dataset_path /home/lofty/CODE/HyperSIGMA-fork/ImageDenoising/data/HSI_Data/Hyperspectral_Project/WDC/wdc.db --lr $lr --basedir $output --pretrain_path $pretrain_path --loss ${loss} --epoch ${epoch} 2>&1 | tee ${output}/training.log