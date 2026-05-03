#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FINETUNE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${FINETUNE_ROOT}"

# Single-GPU Qwen3-VL-4B LoRA SFT for VideoProxy 10K video descriptions.
# The default effective batch is 32 = 1 GPU * micro batch 4 * grad accum 8.
# Do not set BATCH_SIZE=32 as the micro batch for 256-frame video SFT.

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29512}"

MODEL_PATH="${MODEL_PATH:-/m2v_intern/xuboshen/models/Qwen3-VL-4B-Instruct}"
DATASET_USE="${DATASET_USE:-videoproxy_desc_10k}"

SFT_MODEL_ROOT="${SFT_MODEL_ROOT:-/m2v_intern/xuboshen/zgw/SFT-Models/VideoProxyMixed}"
EXP_NAME="${EXP_NAME:-qwen3vl4b_videoproxy_desc10k_lora_2ep_1gpu_256f_tok8192}"
OUTPUT_DIR="${OUTPUT_DIR:-${SFT_MODEL_ROOT}/${EXP_NAME}}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-${OUTPUT_DIR}/tensorboard}"
RUN_NAME="${RUN_NAME:-${EXP_NAME}}"

DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-scripts/zero2.json}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-8}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-2}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-8192}"

VIDEO_FPS="${VIDEO_FPS:-2}"
VIDEO_MIN_FRAMES="${VIDEO_MIN_FRAMES:-4}"
VIDEO_MAX_FRAMES="${VIDEO_MAX_FRAMES:-256}"
VIDEO_TOKEN_PIXEL_FACTOR="${VIDEO_TOKEN_PIXEL_FACTOR:-2048}"  # 32 * 32 * 2 for Qwen3-VL video tokens.
VIDEO_MIN_TOKENS="${VIDEO_MIN_TOKENS:-256}"
VIDEO_MAX_TOKENS="${VIDEO_MAX_TOKENS:-8192}"
VIDEO_MIN_PIXELS="${VIDEO_MIN_PIXELS:-$((VIDEO_MIN_TOKENS * VIDEO_TOKEN_PIXEL_FACTOR))}"
VIDEO_MAX_PIXELS="${VIDEO_MAX_PIXELS:-$((VIDEO_MAX_TOKENS * VIDEO_TOKEN_PIXEL_FACTOR))}"

LORA_R="${LORA_R:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
SAVE_STEPS="${SAVE_STEPS:-250}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"

EFFECTIVE_BATCH=$((NPROC_PER_NODE * BATCH_SIZE * GRAD_ACCUM_STEPS))

mkdir -p "${OUTPUT_DIR}" "${TENSORBOARD_DIR}"

echo "[videoproxy-desc-10k] model: ${MODEL_PATH}"
echo "[videoproxy-desc-10k] dataset: ${DATASET_USE}"
echo "[videoproxy-desc-10k] output: ${OUTPUT_DIR}"
echo "[videoproxy-desc-10k] tensorboard: ${TENSORBOARD_DIR}"
echo "[videoproxy-desc-10k] gpus: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, nproc=${NPROC_PER_NODE}"
echo "[videoproxy-desc-10k] batch: micro=${BATCH_SIZE}, grad_accum=${GRAD_ACCUM_STEPS}, effective=${EFFECTIVE_BATCH}"
echo "[videoproxy-desc-10k] video: frames=${VIDEO_MAX_FRAMES}, tokens=${VIDEO_MAX_TOKENS}, pixels=${VIDEO_MAX_PIXELS}, fps=${VIDEO_FPS}"

torchrun --nproc_per_node=${NPROC_PER_NODE} \
  --master_addr="${MASTER_ADDR}" \
  --master_port="${MASTER_PORT}" \
  qwenvl/train/train_qwen.py \
  --deepspeed "${DEEPSPEED_CONFIG}" \
  --model_name_or_path "${MODEL_PATH}" \
  --dataset_use "${DATASET_USE}" \
  --data_flatten True \
  --lora_enable True \
  --lora_r "${LORA_R}" \
  --lora_alpha "${LORA_ALPHA}" \
  --lora_dropout "${LORA_DROPOUT}" \
  --bf16 \
  --gradient_checkpointing True \
  --per_device_train_batch_size "${BATCH_SIZE}" \
  --per_device_eval_batch_size "$((BATCH_SIZE * 2))" \
  --gradient_accumulation_steps "${GRAD_ACCUM_STEPS}" \
  --learning_rate "${LEARNING_RATE}" \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  --model_max_length "${MODEL_MAX_LENGTH}" \
  --video_fps "${VIDEO_FPS}" \
  --video_min_frames "${VIDEO_MIN_FRAMES}" \
  --video_max_frames "${VIDEO_MAX_FRAMES}" \
  --video_min_pixels "${VIDEO_MIN_PIXELS}" \
  --video_max_pixels "${VIDEO_MAX_PIXELS}" \
  --eval_strategy no \
  --save_strategy steps \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit 2 \
  --logging_steps "${LOGGING_STEPS}" \
  --logging_dir "${TENSORBOARD_DIR}" \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.03 \
  --weight_decay 0 \
  --max_grad_norm 1 \
  --report_to tensorboard \
  --run_name "${RUN_NAME}" \
  --output_dir "${OUTPUT_DIR}"
