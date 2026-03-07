#!/bin/bash
# ============================================================================
# 独立评估脚本：生成样本 + 计算 FID
#
# 用法:
#   # 评估指定实验目录下最新 checkpoint
#   ./eval.sh --exp-name xl_kv1 --gpu 0,1,2,3 --num-gpus 4
#
#   # 评估指定 step（可逗号分隔多个）
#   ./eval.sh --exp-name xl_kv1 --gpu 0,1,2,3 --num-gpus 4 --eval-steps 200000,400000
#
#   # 直接指定 ckpt 路径
#   ./eval.sh --ckpt results/xl_kv1/checkpoints/0400000.pt --gpu 0,1 --num-gpus 2
#
#   # 只跑 FID，跳过生成（已有 npz）
#   ./eval.sh --exp-name xl_kv1 --eval-steps 400000 --fid-only
# ============================================================================
set -e

# ── 默认参数 ──────────────────────────────────────────────────────────────────
EXP_NAME=""
CKPT_PATH=""             # 直接指定 ckpt，与 exp-name 二选一
EVAL_STEPS=""            # 逗号分隔；空则自动取最新
GPU="${GPU:-0,1,2,3,4,5,6,7}"
NUM_GPUS="${NUM_GPUS:-8}"
RESULTS_DIR="${RESULTS_DIR:-results}"

# 模型
MODEL="${MODEL:-DiT-XL/2}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"
ENCODER_DEPTH="${ENCODER_DEPTH:-8}"
NUM_KV_LAYERS="${NUM_KV_LAYERS:-1}"
# enc-dim / enc-num-heads 优先从 checkpoint 的 args 字段读取，此处为兜底
ENC_DIM="${ENC_DIM:-1024}"
ENC_NUM_HEADS="${ENC_NUM_HEADS:-16}"

# 采样
VAE="${VAE:-mse}"
CFG_SCALE="${CFG_SCALE:-1.0}"
NUM_SAMPLING_STEPS="${NUM_SAMPLING_STEPS:-250}"
NUM_FID_SAMPLES="${NUM_FID_SAMPLES:-50000}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
GLOBAL_SEED="${GLOBAL_SEED:-0}" 

# 评估
REF_BATCH="${REF_BATCH:-/workspace/SIT/VIRTUAL_imagenet256_labeled.npz}"
LATENTS_STATS="${LATENTS_STATS:-pretrained_models/sdvae-ft-mse-f8d4-latents-stats.pt}"

# 模式开关
FID_ONLY="${FID_ONLY:-false}"   # true = 跳过生成，直接找已有 npz 计算 FID

# ── 解析命令行参数 ─────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --exp-name)          EXP_NAME="$2";         shift 2 ;;
        --ckpt)              CKPT_PATH="$2";        shift 2 ;;
        --eval-steps)        EVAL_STEPS="$2";       shift 2 ;;
        --gpu)               GPU="$2";              shift 2 ;;
        --num-gpus)          NUM_GPUS="$2";         shift 2 ;;
        --results-dir)       RESULTS_DIR="$2";      shift 2 ;;
        --model)             MODEL="$2";            shift 2 ;;
        --image-size)        IMAGE_SIZE="$2";       shift 2 ;;
        --encoder-depth)     ENCODER_DEPTH="$2";    shift 2 ;;
        --num-kv-layers)     NUM_KV_LAYERS="$2";    shift 2 ;;
        --enc-dim)           ENC_DIM="$2";          shift 2 ;;
        --enc-num-heads)     ENC_NUM_HEADS="$2";    shift 2 ;;
        --vae)               VAE="$2";              shift 2 ;;
        --cfg-scale)         CFG_SCALE="$2";        shift 2 ;;
        --num-sampling-steps) NUM_SAMPLING_STEPS="$2"; shift 2 ;;
        --num-fid-samples)   NUM_FID_SAMPLES="$2";  shift 2 ;;
        --eval-batch-size)   EVAL_BATCH_SIZE="$2";  shift 2 ;;
        --ref-batch)         REF_BATCH="$2";        shift 2 ;;
        --latents-stats)     LATENTS_STATS="$2";    shift 2 ;;
        --fid-only)          FID_ONLY="true";       shift ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# ── 参数校验 ──────────────────────────────────────────────────────────────────
if [ -z "$EXP_NAME" ] && [ -z "$CKPT_PATH" ]; then
    echo "错误: 请指定 --exp-name <name> 或 --ckpt <path>"
    exit 1
fi

export CUDA_VISIBLE_DEVICES="${GPU}"

# ── 若直接指定 ckpt，则只评估该单个 ckpt ─────────────────────────────────────
if [ -n "$CKPT_PATH" ]; then
    if [ ! -f "$CKPT_PATH" ]; then
        echo "错误: 找不到 checkpoint: ${CKPT_PATH}"
        exit 1
    fi
    # 从路径推导 exp 名和 step
    CKPT_DIR=$(dirname "$CKPT_PATH")
    EXP_DIR=$(dirname "$CKPT_DIR")
    EXP_NAME=$(basename "$EXP_DIR")
    STEP_STR=$(basename "$CKPT_PATH" .pt)
    EVAL_STEPS="$STEP_STR"
    SAVE_PATH="$EXP_DIR"
else
    SAVE_PATH="${RESULTS_DIR}/${EXP_NAME}"
    CKPT_DIR="${SAVE_PATH}/checkpoints"

    if [ ! -d "$CKPT_DIR" ]; then
        echo "错误: checkpoint 目录不存在: ${CKPT_DIR}"
        exit 1
    fi

    # 若未指定 eval-steps，取最新的 checkpoint
    if [ -z "$EVAL_STEPS" ]; then
        LATEST_CKPT=$(ls "${CKPT_DIR}"/*.pt 2>/dev/null | sort -V | tail -1)
        if [ -z "$LATEST_CKPT" ]; then
            echo "错误: ${CKPT_DIR} 下未找到任何 .pt 文件"
            exit 1
        fi
        EVAL_STEPS=$(basename "${LATEST_CKPT}" .pt | sed 's/^0*//')
        [ -z "$EVAL_STEPS" ] && EVAL_STEPS=0
        echo "自动选择最新 checkpoint: $(basename ${LATEST_CKPT})"
    fi
fi

echo "================================================"
echo "评估实验: ${EXP_NAME}"
echo "GPU:      ${GPU} (${NUM_GPUS} GPUs)"
echo "模型:     ${MODEL}"
echo "steps:    ${EVAL_STEPS}"
echo "cfg:      ${CFG_SCALE}  steps: ${NUM_SAMPLING_STEPS}"
echo "================================================"

MASTER_PORT=$((20000 + RANDOM % 10000))
MODEL_STR="${MODEL/\//-}"

IFS=',' read -ra STEPS_ARRAY <<< "$EVAL_STEPS"
for RAW_STEP in "${STEPS_ARRAY[@]}"; do
    STEP=$(echo "$RAW_STEP" | xargs)          # 去空白
    STEP_INT=$(echo "$STEP" | sed 's/^0*//')  # 去前导零，得到整数
    [ -z "$STEP_INT" ] && STEP_INT=0

    # 优先 zero-padded 格式
    CKPT_FILE="${CKPT_DIR}/$(printf '%07d' ${STEP_INT}).pt"
    if [ ! -f "$CKPT_FILE" ]; then
        CKPT_FILE="${CKPT_DIR}/${STEP}.pt"
    fi
    if [ ! -f "$CKPT_FILE" ]; then
        echo "未找到 checkpoint step=${STEP}，跳过"
        continue
    fi

    STEP_STR=$(printf '%07d' ${STEP_INT})

    echo ""
    echo "------------------------------------------------"
    echo "checkpoint: ${CKPT_FILE}"
    echo "------------------------------------------------"

    SAMPLE_DIR_BASE="${SAVE_PATH}/samples"
    FOLDER_NAME="${MODEL_STR}-${STEP_STR}-size${IMAGE_SIZE}-cfg${CFG_SCALE}-seed${GLOBAL_SEED}"
    SAMPLE_FOLDER="${SAMPLE_DIR_BASE}/${FOLDER_NAME}"
    SAMPLE_NPZ="${SAMPLE_DIR_BASE}/${FOLDER_NAME}.npz"

    # ── 生成样本 ────────────────────────────────────────────────────────────
    if [ "$FID_ONLY" = "false" ]; then
        if [ -f "$SAMPLE_NPZ" ]; then
            echo "已存在 npz，跳过生成: ${SAMPLE_NPZ}"
        else
            echo "生成样本 (${NUM_FID_SAMPLES} 张)..."
            torchrun --standalone \
                --nproc_per_node="${NUM_GPUS}" \
                --master_port="${MASTER_PORT}" \
                sample_encoder_ddp.py \
                --model "${MODEL}" \
                --ckpt "${CKPT_FILE}" \
                --image-size "${IMAGE_SIZE}" \
                --num-enc-kv-layers "${NUM_KV_LAYERS}" \
                --enc-dim "${ENC_DIM}" \
                --enc-num-heads "${ENC_NUM_HEADS}" \
                --encoder-depth "${ENCODER_DEPTH}" \
                --vae "${VAE}" \
                --cfg-scale "${CFG_SCALE}" \
                --num-sampling-steps "${NUM_SAMPLING_STEPS}" \
                --num-fid-samples "${NUM_FID_SAMPLES}" \
                --per-proc-batch-size "${EVAL_BATCH_SIZE}" \
                --global-seed "${GLOBAL_SEED}" \
                --sample-dir "${SAMPLE_DIR_BASE}" \
                --latents-stats-path "${LATENTS_STATS}"
            MASTER_PORT=$((MASTER_PORT + 1))
        fi
    fi

    # ── 计算 FID ────────────────────────────────────────────────────────────
    if [ ! -f "$SAMPLE_NPZ" ]; then
        # sample_encoder_ddp 把 npz 存在 sample_dir/<folder_name>.npz
        # 尝试宽松匹配（cfg 可能被格式化为 1.5 或 1.50 等）
        SAMPLE_NPZ=$(ls "${SAMPLE_DIR_BASE}/${MODEL_STR}-${STEP_STR}-size${IMAGE_SIZE}"*.npz 2>/dev/null | head -1)
    fi

    if [ -f "$SAMPLE_NPZ" ]; then
        echo "计算 FID..."
        FID_SAVE="${SAVE_PATH}/fid_results"
        python evaluations/evaluator.py \
            --ref_batch "${REF_BATCH}" \
            --sample_batch "${SAMPLE_NPZ}" \
            --save_path "${FID_SAVE}" \
            --step "${STEP_INT}" \
            --num_steps "${NUM_SAMPLING_STEPS}" \
            --cfg "${CFG_SCALE}"
    else
        echo "未找到 npz 文件，跳过 FID 计算"
        echo "  期望路径: ${SAMPLE_NPZ}"
    fi
done

echo ""
echo "================================================"
echo "评估完成！结果保存在: ${SAVE_PATH}/fid_results/"
echo "================================================"
