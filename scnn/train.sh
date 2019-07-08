#!usr/bin/env sh

CUDA_VISIBLE_DEVICES=4,5,6,7 python train.py \
    --batch_size  4 \
    --epoch 20 \
    --learning_rate 1e-4 \
    --num_steps 60000 \
    --data_dir /home/BackUp/docker-file/wjt/Changshu \
    --data_list /home/BackUp/docker-file/wjt/Changshu/list/train_gt.txt \
    --input_size 512,640 \
    --pretrained_params_path ./pretrained_params/pretrained_params.ckpt \
    --save_dir ./prediction \
    --save_num_images 2 \
    --save_pred_every 500 \
    --snapshot_dir ./snapshots \
2>&1|tee ./train.log
