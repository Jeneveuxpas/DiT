#!/bin/bash
# ============================================================================
# DiT Encoder KV — 统一启动脚本（训练 + 评估）
#
# 用法:
#   ./launch.sh --config configs/dit_xl_kv-2.0-repa-0.5.yaml --exp-name dit_xl_kv-2.0-repa-0.5 --gpu 0,1,2,3 --num-gpus 4 --wandb
#   ./launch.sh --exp-name xl_kv1 --model-size XL --gpu 4,5,6,7 --num-gpus 4
#   ./launch.sh --exp-name xl_kv1 --gpu 4,5,6,7 --num-gpus 4 --eval-only
#   ./launch.sh --exp-name xl_kv1 --gpu 4,5,6,7 --num-gpus 4 --resume-step 200000
#   ./launch.sh --exp-name xl_kv1 --gpu 0,1,2,3 --num-gpus 4 --eval-steps 200000,400000
# ============================================================================
set -e

# ── 默认参数 ──────────────────────────────────────────────────────────────────
CONFIG=""
GPU="${GPU:-0,1,2,3,4,5,6,7}"
NUM_GPUS="${NUM_GPUS:-8}"
RESUME_STEP="${RESUME_STEP:-0}"
EVAL_STEPS="${EVAL_STEPS:-}"        # 逗号分隔，空则评估最新 ckpt
EVAL_ONLY="${EVAL_ONLY:-false}"
SKIP_EVAL="${SKIP_EVAL:-false}"

# 模型
MODEL_SIZE="${MODEL_SIZE:-XL}"

# 数据
DATA_DIR="${DATA_DIR:-/dev/shm/data}"
LATENTS_STATS="${LATENTS_STATS:-pretrained_models/sdvae-ft-mse-f8d4-latents-stats.pt}"
IMAGE_SIZE="${IMAGE_SIZE:-256}"


# 训练
EPOCHS="${EPOCHS:-1400}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-256}"
CKPT_EVERY="${CKPT_EVERY:-50000}"
LOG_EVERY="${LOG_EVERY:-100}"

# Encoder KV
USE_KV="${USE_KV:-true}"
ENC_TYPE="${ENC_TYPE:-vit_large_patch14_dinov2.lvd142m}"
ENC_RESOLUTION="${ENC_RESOLUTION:-224}"
ENC_LAYER_INDICES="${ENC_LAYER_INDICES:-}"   # 空则用 --num-kv-layers
NUM_KV_LAYERS="${NUM_KV_LAYERS:-1}"
KV_PROJ_TYPE="${KV_PROJ_TYPE:-linear}"
KV_NORM_TYPE="${KV_NORM_TYPE:-layer}"

# 两阶段
STAGE1_STEPS="${STAGE1_STEPS:-30000}"
DISTILL_COEFF="${DISTILL_COEFF:-1.0}"

# REPA
PROJ_COEFF="${PROJ_COEFF:-0.5}"
ENCODER_DEPTH="${ENCODER_DEPTH:-8}"
REPA_LOSS="${REPA_LOSS:-cosine}"

# wandb
USE_WANDB="${USE_WANDB:-false}"
WANDB_PROJECT="${WANDB_PROJECT:-DiT-EncoderKV}"

# 评估
NUM_FID_SAMPLES="${NUM_FID_SAMPLES:-50000}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
NUM_SAMPLING_STEPS="${NUM_SAMPLING_STEPS:-250}"
CFG_SCALE="${CFG_SCALE:-1.0}"
VAE="${VAE:-mse}"
REF_BATCH="${REF_BATCH:-/workspace/SIT/VIRTUAL_imagenet256_labeled.npz}"

# ── 解析命令行参数 ─────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)           CONFIG="$2";            shift 2 ;;
        --exp-name)         EXP_NAME="$2";          shift 2 ;;
        --gpu)              GPU="$2";               shift 2 ;;
        --num-gpus)         NUM_GPUS="$2";          shift 2 ;;
        --model-size)       MODEL_SIZE="$2";        shift 2 ;;
        --data-dir)         DATA_DIR="$2";          shift 2 ;;
        --latents-stats)    LATENTS_STATS="$2";     shift 2 ;;
        --epochs)           EPOCHS="$2";            shift 2 ;;
        --global-batch-size) GLOBAL_BATCH_SIZE="$2"; shift 2 ;;
        --stage1-steps)     STAGE1_STEPS="$2";      shift 2 ;;
        --num-kv-layers)    NUM_KV_LAYERS="$2";     shift 2 ;;
        --enc-layer-indices) ENC_LAYER_INDICES="$2"; shift 2 ;;
        --kv-proj-type)     KV_PROJ_TYPE="$2";      shift 2 ;;
        --kv-norm-type)     KV_NORM_TYPE="$2";      shift 2 ;;
        --distill-coeff)    DISTILL_COEFF="$2";     shift 2 ;;
        --proj-coeff)       PROJ_COEFF="$2";        shift 2 ;;
        --encoder-depth)    ENCODER_DEPTH="$2";     shift 2 ;;
        --enc-type)         ENC_TYPE="$2";          shift 2 ;;
        --cfg-scale)        CFG_SCALE="$2";         shift 2 ;;
        --resume-step)      RESUME_STEP="$2";       shift 2 ;;
        --eval-steps)       EVAL_STEPS="$2";        shift 2 ;;
        --eval-only)        EVAL_ONLY="true";       shift ;;
        --skip-eval)        SKIP_EVAL="true";       shift ;;
        --no-kv)            USE_KV="false";         shift ;;
        --wandb)            USE_WANDB="true";       shift ;;
        --wandb-project)    WANDB_PROJECT="$2";     shift 2 ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

# ── 检查必要参数 ───────────────────────────────────────────────────────────────
if [ -z "$EXP_NAME" ]; then
    echo "请指定实验名: --exp-name <name>"
    exit 1
fi

# ── 若提供 config，从中读取关键字段（覆盖 shell 默认值）────────────────────────
if [ -n "$CONFIG" ]; then
    if [ ! -f "$CONFIG" ]; then
        echo "config 文件不存在: $CONFIG"
        exit 1
    fi
    _YAML=$(python3 -c "
import yaml, sys
d = yaml.safe_load(open('$CONFIG')) or {}
print(d.get('model', ''))
print(d.get('enc-type', ''))
" 2>/dev/null || echo -e "\n")
    _YAML_MODEL=$(echo "$_YAML" | sed -n '1p')
    _YAML_ENC_TYPE=$(echo "$_YAML" | sed -n '2p')

    if [ -n "$_YAML_MODEL" ]; then
        case "$_YAML_MODEL" in
            DiT-XL*) MODEL_SIZE="XL" ;;
            DiT-L*)  MODEL_SIZE="L"  ;;
            DiT-B*)  MODEL_SIZE="B"  ;;
            DiT-S*)  MODEL_SIZE="S"  ;;
        esac
        echo "从 config 检测到模型: ${_YAML_MODEL} (MODEL_SIZE=${MODEL_SIZE})"
    fi
    if [ -n "$_YAML_ENC_TYPE" ]; then
        ENC_TYPE="$_YAML_ENC_TYPE"
        echo "从 config 检测到 enc-type: ${ENC_TYPE}"
    fi
fi

# ── 根据 MODEL_SIZE 设置模型名 ─────────────────────────────────────────────────
case "$MODEL_SIZE" in
    B|b)   MODEL="DiT-B/2"; ENC_DIM=768;  ENC_NUM_HEADS=12 ;;
    L|l)   MODEL="DiT-L/2"; ENC_DIM=1024; ENC_NUM_HEADS=16 ;;
    XL|xl) MODEL="DiT-XL/2"; ENC_DIM=1024; ENC_NUM_HEADS=16 ;;
    S|s)   MODEL="DiT-S/2"; ENC_DIM=384;  ENC_NUM_HEADS=6  ;;
    *)
        echo "无效的模型大小: $MODEL_SIZE，可选: S B L XL"
        exit 1
        ;;
esac

SAVE_PATH="results/${EXP_NAME}"
export CUDA_VISIBLE_DEVICES="${GPU}"

MASTER_PORT=$((29500 + RANDOM % 1000))

# ── 打印配置 ───────────────────────────────────────────────────────────────────
echo "================================================"
echo "DiT Encoder KV — Attention Scaffolding"
echo "================================================"
echo "实验名:       ${EXP_NAME}"
echo "模型:         ${MODEL}"
echo "GPU:          ${GPU} (${NUM_GPUS} GPUs)"
echo "数据目录:     ${DATA_DIR}"
echo "Stage1 步数:  ${STAGE1_STEPS}"
echo "KV 层数:      ${NUM_KV_LAYERS}"
echo "蒸馏系数:     ${DISTILL_COEFF}"
echo "REPA 系数:    ${PROJ_COEFF}"
echo "================================================"

# ── 训练阶段 ───────────────────────────────────────────────────────────────────
if [ "$EVAL_ONLY" = "false" ]; then
    echo "开始训练..."

    # 构建训练命令
    TRAIN_ARGS=(
        --model "${MODEL}"
        --data-dir "${DATA_DIR}"
        --latents-stats-path "${LATENTS_STATS}"
        --results-dir results
        --image-size "${IMAGE_SIZE}"
        --epochs "${EPOCHS}"
        --global-batch-size "${GLOBAL_BATCH_SIZE}"
        --stage1-steps "${STAGE1_STEPS}"
        --distill-coeff "${DISTILL_COEFF}"
        --proj-coeff "${PROJ_COEFF}"
        --encoder-depth "${ENCODER_DEPTH}"
        --repa-loss "${REPA_LOSS}"
        --enc-type "${ENC_TYPE}"
        --enc-resolution "${ENC_RESOLUTION}"
        --num-kv-layers "${NUM_KV_LAYERS}"
        --kv-proj-type "${KV_PROJ_TYPE}"
        --kv-norm-type "${KV_NORM_TYPE}"
        --ckpt-every "${CKPT_EVERY}"
        --log-every "${LOG_EVERY}"
    )

    [ -n "$CONFIG" ]            && TRAIN_ARGS+=(--config "${CONFIG}")
    [ "$USE_KV" = "true" ]      && TRAIN_ARGS+=(--use-kv)
    [ -n "$ENC_LAYER_INDICES" ] && TRAIN_ARGS+=(--enc-layer-indices "${ENC_LAYER_INDICES}")
    [ -n "$EXP_NAME" ]          && TRAIN_ARGS+=(--exp-name "${EXP_NAME}")
    [ "$USE_WANDB" = "true" ]   && TRAIN_ARGS+=(--wandb --wandb-project "${WANDB_PROJECT}")

    # resume
    if [ "$RESUME_STEP" -gt 0 ]; then
        RESUME_CKPT=$(printf "${SAVE_PATH}/checkpoints/%07d.pt" ${RESUME_STEP})
        if [ -f "$RESUME_CKPT" ]; then
            echo "从 checkpoint 恢复: ${RESUME_CKPT} (step ${RESUME_STEP})"
            TRAIN_ARGS+=(--resume-step "${RESUME_STEP}")
        else
            echo "警告: 未找到 checkpoint ${RESUME_CKPT}，从头开始训练"
        fi
    fi

    torchrun --standalone --nproc_per_node="${NUM_GPUS}" --master_port="${MASTER_PORT}" \
        train_encoder.py "${TRAIN_ARGS[@]}"

    echo "================================================"
    echo "训练完成！"
    echo "================================================"
fi

# ── 评估阶段 ───────────────────────────────────────────────────────────────────
if [ "$SKIP_EVAL" = "false" ]; then
    echo "开始评估..."

    # checkpoint 目录即 results/<exp_name>/checkpoints
    CKPT_DIR="${SAVE_PATH}/checkpoints"

    # 确定要评估的 checkpoint steps
    if [ -z "$EVAL_STEPS" ]; then
        LATEST_CKPT=$(ls ${CKPT_DIR}/*.pt 2>/dev/null | sort -V | tail -1)
        if [ -z "$LATEST_CKPT" ]; then
            echo "未找到 checkpoint，跳过评估"
            exit 0
        fi
        EVAL_STEPS=$(basename ${LATEST_CKPT} .pt)
        echo "使用最新 checkpoint: ${EVAL_STEPS}"
    fi

    EVAL_MASTER_PORT=$((29500 + RANDOM % 1000))

    IFS=',' read -ra STEPS_ARRAY <<< "$EVAL_STEPS"
    for STEP in "${STEPS_ARRAY[@]}"; do
        STEP=$(echo $STEP | xargs)
        CKPT_PATH="${CKPT_DIR}/${STEP}.pt"

        if [ ! -f "$CKPT_PATH" ]; then
            # 尝试 zero-padded 格式
            CKPT_PATH="${CKPT_DIR}/$(printf '%07d' ${STEP}).pt"
        fi

        if [ ! -f "$CKPT_PATH" ]; then
            echo "未找到 checkpoint: ${STEP}，跳过"
            continue
        fi

        echo "------------------------------------------------"
        echo "评估 checkpoint: ${STEP}"
        echo "checkpoint 路径: ${CKPT_PATH}"
        echo "------------------------------------------------"

        SAMPLE_DIR_BASE="${CKPT_DIR}"

        # 生成样本（多 GPU）
        torchrun --standalone --nproc_per_node=${NUM_GPUS} --master_port=${EVAL_MASTER_PORT} \
            sample_encoder_ddp.py \
            --model ${MODEL} \
            --ckpt ${CKPT_PATH} \
            --image-size ${IMAGE_SIZE} \
            --num-enc-kv-layers ${NUM_KV_LAYERS} \
            --enc-dim ${ENC_DIM} \
            --enc-num-heads ${ENC_NUM_HEADS} \
            --encoder-depth ${ENCODER_DEPTH} \
            --vae ${VAE} \
            --cfg-scale ${CFG_SCALE} \
            --num-sampling-steps ${NUM_SAMPLING_STEPS} \
            --num-fid-samples ${NUM_FID_SAMPLES} \
            --per-proc-batch-size ${EVAL_BATCH_SIZE} \
            --sample-dir ${SAMPLE_DIR_BASE} \
            --latents-stats-path ${LATENTS_STATS}

        # 找到生成的 npz 文件
        MODEL_STR="${MODEL/\//-}"
        STEP_INT=$(echo ${STEP} | sed 's/^0*//' | grep -o '[0-9]*')
        [ -z "$STEP_INT" ] && STEP_INT=0
        STEP_STR=$(printf '%07d' ${STEP_INT})
        SAMPLE_NPZ="${SAMPLE_DIR_BASE}/${MODEL_STR}-${STEP_STR}-size${IMAGE_SIZE}-cfg${CFG_SCALE}-seed0.npz"

        if [ ! -f "$SAMPLE_NPZ" ]; then
            # 宽松匹配（仅匹配当前 step）
            SAMPLE_NPZ=$(ls ${SAMPLE_DIR_BASE}/${MODEL_STR}-${STEP_STR}-size${IMAGE_SIZE}*.npz 2>/dev/null | head -1)
        fi
        
        if [ -f "$SAMPLE_NPZ" ]; then
            echo "计算 FID..."
            python evaluations/evaluator.py \
                --ref_batch ${REF_BATCH} \
                --sample_batch ${SAMPLE_NPZ} \
                --save_path ${SAVE_PATH}/checkpoints \
                --step ${STEP} \
                --num_steps ${NUM_SAMPLING_STEPS} \
                --cfg ${CFG_SCALE}
        else
            echo "未找到 npz 文件，请检查 ${SAMPLE_DIR_BASE}"
        fi

        EVAL_MASTER_PORT=$((EVAL_MASTER_PORT + 1))
    done
fi

echo "================================================"
echo "全部完成！结果保存在: results/"
echo "================================================"
