#!/bin/bash
#
#SBATCH --job-name=static_train
#SBATCH --output=output3.txt
#SBATCH --error=error3.txt
#
#SBATCH --ntasks=1
#SBATCH --time=300:00:00
#SBATCH --mem=40G
#SBATCH --partition=a800
#SBATCH --qos=a800
#SBATCH --gres=gpu:1


python llava/train/train_mem_gcg.py \
    --model_name_or_path /home/zengxiyu24/RobustVLM/llava-v1.5-7b \
    --version v1 \
    --data_path ./playground/data/new_joint_train_data_gpt4.json \
    --normal_image_folder ./playground/data/normal_loss_images \
    --adversarial_image_folder ./playground/data/imagenet/my_test_sampled \
    --vision_tower /home/zengxiyu24/LLaVA/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter False \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llava_static \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_steps 500 \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --gcg_step 0 \
    --pgd_iterations 10 \
    --prompt_path /home/zengxiyu24/LLaVA/playground/data/dataset.csv