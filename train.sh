export CUDA_VISIBLE_DEVICES=0,1,2

python -m torch.distributed.launch --nproc_per_node=3 --master_port 29501 --use_env eval.py --aug_scale --aug_crop --aug_translate --config configs/r50_dior.py --start_epoch 0 --distributed  
#--resume ./checkpoint.pth

