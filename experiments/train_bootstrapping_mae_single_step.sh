#!/bin/bash
export OMP_NUM_THREADS=4

NUM_GPUS=1
MODEL="mae_vit_tiny_patch4"
DATA_PATH="./datasets/cifar10_dataset"
OUTPUT_DIR="./ckpts/B_mae_1/pretrained"
BATCH_SIZE=64
ACCUM=2
EPOCHS=200
WARMUP_EPOCHS=10
BASE_LR=1.5e-4
INPUT_SIZE=32
MASK_RATIO=0.75

# 获取当前日期和时间
CURRENT_DATETIME=$(date +"%Y-%m-%d_%H-%M-%S")

# 动态生成日志目录
LOG_DIR="./logs/B_mae_1/tb_${CURRENT_DATETIME}"

python main_pretrain_bootstrapped_single_step.py \
    --model ${MODEL} \
    --data_path ${DATA_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --batch_size ${BATCH_SIZE} \
    --accum_iter ${ACCUM} \
    --epochs ${EPOCHS} \
    --warmup_epochs ${WARMUP_EPOCHS} \
    --blr ${BASE_LR} \
    --input_size ${INPUT_SIZE} \
    --mask_ratio ${MASK_RATIO} \
    --norm_pix_loss \
    --log_dir ${LOG_DIR}\
    --bootstrapping \
    --last_model "./ckpts/original_mae/pretrained/checkpoint-199.pth"