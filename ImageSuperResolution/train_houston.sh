n_scale=$1
batch_size=$2
model_title=$3 # SpatSIGMA, HyperSIGMA
dataset_name=$4 # houston

CUDA_VISIBLE_DEVICES=0 \
python main38_houston.py train \
--model_title $model_title \
--n_scale $n_scale \
--la1 0.3 \
--la2 0.1 \
--dataset_name $dataset_name \
--epochs 350 \
--gpus 0 \
--batch_size $batch_size \
--learning_rate 6e-5 \
--pretrain_path /home/lofty/CODE/HyperSIGMA-fork/spat-base.pth
