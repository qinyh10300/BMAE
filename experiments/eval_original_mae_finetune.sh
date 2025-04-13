#!/bin/bash
export OMP_NUM_THREADS=4

NUM_GPUS=1
MODEL="vit_tiny_patch16"
DATA_PATH="./datasets/cifar10_dataset"
OUTPUT_DIR="./ckpts/original_mae/finetune_final"
BATCH_SIZE=32
EPOCHS=100
WARMUP_EPOCHS=10
BASE_LR=1e-3
INPUT_SIZE=32
WEIGHT_DECAY=0
DROP_PATH=0.05

# 获取当前日期和时间
CURRENT_DATETIME=$(date +"%Y-%m-%d_%H-%M-%S")

# 动态生成日志目录
LOG_DIR="./logs/original_mae_finetune/tb_${CURRENT_DATETIME}"

torchrun --nproc_per_node=${NUM_GPUS} --master_port=12366 main_finetune.py \
    --world_size ${NUM_GPUS} \
    --model ${MODEL} \
    --data_path ${DATA_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    --warmup_epochs ${WARMUP_EPOCHS} \
    --blr ${BASE_LR} \
    --input_size ${INPUT_SIZE} \
    --weight_decay ${WEIGHT_DECAY} \
    --drop_path ${DROP_PATH} \
    --log_dir ${LOG_DIR} \
    --finetune "./ckpts/original_mae/pretrained/checkpoint-199.pth" \
    --nb_classes 10 \
    --mixup 0.8 \
    --cutmix 1.0 \
    --dist_eval
