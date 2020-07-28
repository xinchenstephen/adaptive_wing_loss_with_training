CUDA_VISIBLE_DEVICES=0 python ../eval.py \
                    --val_img_dir='./data/300w/images/wing/images' \
                    --val_landmarks_dir='./data/300w/images/wing/landmarks' \
                    --ckpt_save_path='./AdaptiveWingLoss/experiments/eval_iccv_0620' \
                    --hg_blocks=4 \
                    --pretrained_weights='./AdaptiveWingLoss/ckpt/WFLW_4HG.pth' \
                    --num_landmarks=68 \
                    --end_relu='False' \
                    --batch_size=20 \

