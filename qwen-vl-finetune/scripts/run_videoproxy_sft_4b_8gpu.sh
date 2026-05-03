#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FINETUNE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${FINETUNE_ROOT}"

# Rerun the three VideoProxy LoRA SFT jobs that previously used too-small
# Qwen3-VL video budgets. Run this script on one 8-GPU node for Qwen3-VL-4B.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT_BASE="${MASTER_PORT_BASE:-29600}"

MODEL_PATH="${MODEL_PATH:-/m2v_intern/xuboshen/models/Qwen3-VL-4B-Instruct}"
SFT_MODEL_ROOT="${SFT_MODEL_ROOT:-/m2v_intern/xuboshen/zgw/SFT-Models/VideoProxyMixed}"

NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-2}"
VIDEO_MAX_TOKENS="${VIDEO_MAX_TOKENS:-8192}"
VIDEO_MAX_FRAMES="${VIDEO_MAX_FRAMES:-256}"
MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH:-16384}"
SAVE_STEPS="${SAVE_STEPS:-250}"
LOGGING_STEPS="${LOGGING_STEPS:-1}"
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"

# Keep effective batch at 32 for each task by default: 8 GPUs * 1 * 4.
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"

run_task() {
  local task_name="$1"
  local launcher="$2"
  local port_offset="$3"
  local exp_name="$4"

  echo
  echo "========== [4B] ${task_name}: ${exp_name} =========="
  echo "model=${MODEL_PATH}"
  echo "launcher=${launcher}"
  echo "gpus=${CUDA_VISIBLE_DEVICES}, nproc=${NPROC_PER_NODE}, epochs=${NUM_TRAIN_EPOCHS}"
  echo "video_tokens=${VIDEO_MAX_TOKENS}, max_frames=${VIDEO_MAX_FRAMES}, max_length=${MODEL_MAX_LENGTH}"

  MODEL_PATH="${MODEL_PATH}" \
  SFT_MODEL_ROOT="${SFT_MODEL_ROOT}" \
  NPROC_PER_NODE="${NPROC_PER_NODE}" \
  MASTER_ADDR="${MASTER_ADDR}" \
  MASTER_PORT="$((MASTER_PORT_BASE + port_offset))" \
  NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS}" \
  VIDEO_MAX_TOKENS="${VIDEO_MAX_TOKENS}" \
  VIDEO_MAX_FRAMES="${VIDEO_MAX_FRAMES}" \
  MODEL_MAX_LENGTH="${MODEL_MAX_LENGTH}" \
  BATCH_SIZE="${BATCH_SIZE}" \
  GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS}" \
  SAVE_STEPS="${SAVE_STEPS}" \
  LOGGING_STEPS="${LOGGING_STEPS}" \
  DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS}" \
  EXP_NAME="${exp_name}" \
  bash "${launcher}"
}

run_task vdc scripts/run_videoproxy_desc_10k_lora_1gpu.sh 0 \
  qwen3vl4b_videoproxy_vdc10k_lora_2ep_8gpu_256f_tok8192_len16k_rerun
run_task dvc scripts/run_videoproxy_dvc_10k_lora_1gpu.sh 1 \
  qwen3vl4b_videoproxy_dvc10k_lora_2ep_8gpu_256f_tok8192_len16k_rerun
run_task proxy scripts/run_videoproxy_proxy_mix_sft_lora_1gpu.sh 2 \
  qwen3vl4b_videoproxy_proxy_mix_sft_lora_2ep_8gpu_256f_tok8192_len16k_rerun
