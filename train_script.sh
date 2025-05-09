#!/bin/bash

# 加载 conda 初始化脚本
# source ~/miniforge3/etc/profile.d/conda.sh

# 激活指定的 conda 环境
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate lerobot

#HYDRA_FULL_ERROR=1

# 生成当前时间的时间戳
timestamp=$(date +%Y-%m-%d_%H-%M-%S)

# 使用时间戳命名日志文件 
log_dir="./logs"
mkdir -p "$log_dir" && chmod 755 "$log_dir"
logfile="${log_dir}/${timestamp}_train.log"



## 使用nohup命令运行脚本并将输出重定向到日志文件
## 使用ps -wwo pid,user,lstart,cmd [PID]查看空闲的GPU

## pi0 train start
# task_name=stack_cup_0414_merged_rf10
# nohup bash -c "CUDA_VISIBLE_DEVICES=0 HYDRA_FULL_ERROR=1  python lerobot/scripts/train.py \
# --policy.path='/workspace/data/jiahuan/huggingface/models/pi0' \
# --dataset.repo_id='/workspace/data/TR2/hugging_face/${task_name}' \
# --output_dir='/workspace/data/huxian/training/lerobot/pi0/$(date +"%Y-%m-%d")/${task_name}-$(date +"%m%d_%H%M")' \
# --batch_size=10 \
# --save_freq=10000 \
# --steps=500000 \
# --num_workers=12 \
# --wandb.enable=true \
# --wandb.entity='GBuilders' \
# --wandb.project='pi0_TR2' \
# --wandb.disable_artifact=true \
# --job_name='${task_name}_$(date +"%m%d_%H%M_%S")'" > "$logfile" 2>&1 &

## pi0 train resume,resume command canbe found in resume_comman.log in output_dir
# nohup bash -c "
# HYDRA_FULL_ERROR=1 python lerobot/scripts/train.py --policy.path=/workspace/data/jiahuan/huggingface/models/pi0 --dataset.repo_id=/workspace/data/TR2/hugging_face/stack_cup_same_0429 --batch_size=10 --save_freq=10000 --steps=500000 --num_workers=12 --wandb.enable=true --wandb.entity=GBuilders --wandb.project=pi0_TR2 --wandb.disable_artifact=true --job_name=stack_cup_same_0429_0429_1911_33 --resume=true --config_path=/data/huxian/training/lerobot/pi0/2025-04-29/stack_cup_same_0429-0429_1911/checkpoints/last/pretrained_model/train_config.json
# " > "$logfile" 2>&1 &


## act train start
task_name=pick_and_place_merged_0105_0
nohup bash -c "HYDRA_FULL_ERROR=1 python lerobot/scripts/train.py \
--policy.type='act' \
--dataset.repo_id='/workspace/data/TR2/hugging_face/${task_name}' \
--output_dir='/workspace/data/huxian/training/lerobot/act/$(date +"%Y-%m-%d")/${task_name}-$(date +"%m%d_%H%M")' \
--batch_size=256 \
--save_freq=10000 \
--steps=500000 \
--num_workers=12 \
--wandb.enable=true \
--wandb.entity='GBuilders' \
--wandb.project='act_tr2' \
--wandb.disable_artifact=true \
--job_name='${task_name}_$(date +"%m%d_%H%M_%S")'" > "$logfile" 2>&1 &

# 提示日志文件的位置
echo "Training script is running in the background. Check the log file: $logfile"