#pretrain_data_path='/home/lofty/CODE/HyperSIGMA-fork/Pretrain/data'
#model_saved_patch='/home/lofty/CODE/HyperSIGMA-fork/Pretrain/model'
#logs_path='/home/lofty/CODE/HyperSIGMA-fork/Pretrain/logs'

python main_pretrain_Spat.py \
--model 'spat_mae_b' \
--norm_pix_loss \
--data_path /home/lofty/CODE/HyperSIGMA-fork/Pretrain/data/Spat \
--output_dir /home/lofty/CODE/HyperSIGMA-fork/Pretrain/model \
--log_dir /home/lofty/CODE/HyperSIGMA-fork/Pretrain/logs \
--blr 1.5e-4 \
--batch_size 32 \
--gpu_num 1 \
--epochs 10 \
--warmup_epochs 1
