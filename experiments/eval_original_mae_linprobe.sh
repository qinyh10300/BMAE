#!/bin/bash
export OMP_NUM_THREADS=4

NUM_GPUS=1
MODEL="vit_tiny_patch16"
DATA_PATH="./datasets/cifar10_dataset"
OUTPUT_DIR="./ckpts/original_mae/linprobe_final"
BATCH_SIZE=64
EPOCHS=100
WARMUP_EPOCHS=10
BASE_LR=0.01
WEIGHT_DECAY=0

# 获取当前日期和时间
CURRENT_DATETIME=$(date +"%Y-%m-%d_%H-%M-%S")

# 动态生成日志目录
LOG_DIR="./logs/original_mae_linprobe/tb_${CURRENT_DATETIME}"

torchrun --nproc_per_node=${NUM_GPUS} --master_port=12362 main_linprobe.py \
    --world_size ${NUM_GPUS} \
    --model ${MODEL} \
    --data_path ${DATA_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --warmup_epochs ${WARMUP_EPOCHS} \
    --blr ${BASE_LR} \
    --weight_decay ${WEIGHT_DECAY} \
    --log_dir ${LOG_DIR} \
    --finetune "./ckpts/original_mae/pretrained/checkpoint-199.pth" \
    --nb_classes 10 \
    --dist_eval
