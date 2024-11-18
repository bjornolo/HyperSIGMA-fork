testdir=$1

python hsi_denoising_test.py \
-a hypersigma \
-p hypersigma_gaussian \
-r \
-rp ./output/original_hypersigma_1e-4_spat-base_batch4_warmup_l2_epoch_1_complex_s3_8point_HYPSO2/hypersigma_gaussian/model_latest.pth \
--testdir  $testdir  \
--basedir ./data/HSI_Data/Hyperspectral_Project/GRIZZLY/results  \
--pretrain_path /home/lofty/CODE/HyperSIGMA-fork/spat-base.pth