#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FINETUNE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${FINETUNE_ROOT}"

# Two-GPU Qwen3-VL-4B LoRA SFT smoke run for VideoProxy description data.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29511}"

MODEL_PATH="${MODEL_PATH:-/m2v_intern/xuboshen/models/Qwen3-VL-4B-Instruct}"
DATASET_USE="${DATASET_USE:-videoproxy_desc_smoke}"
OUTPUT_DIR="${OUTPUT_DIR:-./output/qwen3vl4b_videoproxy_desc_smoke_lora_256f_px65536}"
RUN_NAME="${RUN_NAME:-qwen3vl4b_videoproxy_desc_smoke_lora_256f_px65536}"

DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-scripts/zero2.json}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
MAX_STEPS="${MAX_STEPS:-20}"
SAVE_STEPS="${SAVE_STEPS:-20}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-8192}"

VIDEO_FPS="${VIDEO_FPS:-2}"
VIDEO_MIN_FRAMES="${VIDEO_MIN_FRAMES:-4}"
VIDEO_MAX_FRAMES="${VIDEO_MAX_FRAMES:-256}"
VIDEO_MIN_PIXELS="${VIDEO_MIN_PIXELS:-3136}"
VIDEO_MAX_PIXELS="${VIDEO_MAX_PIXELS:-65536}"

LORA_R="${LORA_R:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"
LORA_DROPOUT="${LORA_DROPOUT:-0.05}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
REPORT_TO="${REPORT_TO:-none}"

echo "[videoproxy-desc-smoke] model: ${MODEL_PATH}"
echo "[videoproxy-desc-smoke] dataset: ${DATASET_USE}"
echo "[videoproxy-desc-smoke] output: ${OUTPUT_DIR}"
echo "[videoproxy-desc-smoke] gpus: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, nproc=${NPROC_PER_NODE}"
echo "[videoproxy-desc-smoke] video: frames=${VIDEO_MAX_FRAMES}, pixels=${VIDEO_MAX_PIXELS}, fps=${VIDEO_FPS}"

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
  --max_steps "${MAX_STEPS}" \
  --model_max_length "${MODEL_MAX_LENGTH}" \
  --video_fps "${VIDEO_FPS}" \
  --video_min_frames "${VIDEO_MIN_FRAMES}" \
  --video_max_frames "${VIDEO_MAX_FRAMES}" \
  --video_min_pixels "${VIDEO_MIN_PIXELS}" \
  --video_max_pixels "${VIDEO_MAX_PIXELS}" \
  --eval_strategy no \
  --save_strategy steps \
  --save_steps "${SAVE_STEPS}" \
  --save_total_limit 1 \
  --logging_steps 1 \
  --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.03 \
  --weight_decay 0 \
  --max_grad_norm 1 \
  --report_to "${REPORT_TO}" \
  --run_name "${RUN_NAME}" \
  --output_dir "${OUTPUT_DIR}"
