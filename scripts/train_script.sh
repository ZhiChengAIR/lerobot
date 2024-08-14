#!/bin/bash

# 加载 conda 初始化脚本
source ~/anaconda3/etc/profile.d/conda.sh

# 激活指定的 conda 环境
conda activate lerobot

#HYDRA_FULL_ERROR=1

# 生成当前时间的时间戳
timestamp=$(date +%Y-%m-%d_%H-%M-%S)

# 使用时间戳命名日志文件
logfile="${timestamp}_train.log"

# 环境变量和命令
command='DATA_DIR="/home/h666/code/dataset/hf_dataset/zcai" HYDRA_FULL_ERROR=1 gdb -ex r --args python ../lerobot/scripts/train.py'

# 使用nohup命令运行脚本并将输出重定向到日志文件
nohup bash -c "$command" > "$logfile" 2>&1 &

# 提示日志文件的位置
echo "Training script is running in the background. Check the log file: $logfile"

