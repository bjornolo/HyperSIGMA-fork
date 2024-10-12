n_scale=$1
model_title=$2 # SpatSIGMA, HyperSIGMA
dataset_name=$3
weight_path=$4  

CUDA_VISIBLE_DEVICES=0 \
python main38_houston.py test \
--model_title $model_title \
--weight_path $weight_path \
--n_scale $n_scale \
--dataset_name $dataset_name \
--gpus 0


