lr=$1
model=$2
gpu=$3
loss=$4
epoch=$5
training_dataset_path=$6

batch_size=4
pretrain_path=/home/lofty/CODE/HyperSIGMA-fork/spat-base.pth
file=$(basename $pretrain_path .pth)
output=./output/original_${model}_${lr}_${file}_batch${batch_size}_warmup_${loss}_epoch_${epoch}_complex_s3_8point_HYPSO2
mkdir ${output}
CUDA_VISIBLE_DEVICES=$gpu python hsi_denoising_complex_wdc.py -a $model -p hypersigma_gaussian -b ${batch_size} --training_dataset_path $training_dataset_path --lr $lr --basedir $output --pretrain_path $pretrain_path --loss ${loss} --epoch ${epoch} 2>&1 | tee ${output}/training.log