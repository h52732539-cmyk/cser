#!/usr/bin/env bash
#SBATCH --job-name=cser-msrvtt-updated
#SBATCH -p i64m1tga40u
#SBATCH -o reports/setup/slurm-%x-%j.out
#SBATCH -e reports/setup/slurm-%x-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=144:00:00
#SBATCH -D /hpc2hdd/home/yyan047/xuhaodong/cser

# CSER-only runner for the updated selector/safety protocol.
#
# Defaults now target the raw MSR-VTT 10K checkout. The prepare stage writes the
# CSER query CSV and gallery manifest under DATA_DIR before Phase1/2/3 run.

set -Eeuo pipefail

REPO_ROOT="/hpc2hdd/home/yyan047/xuhaodong/cser"
MSRVTT_ROOT="/hpc2hdd/home/yyan047/xuhaodong/MSRVTT"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data/msrvtt_real_10k}"
REPORT_ROOT="${REPORT_ROOT:-${REPO_ROOT}/reports/cser_only/${SLURM_JOB_ID:-manual}}"
STATUS_FILE="${REPO_ROOT}/reports/setup/cser_only_latest.status"

CONDA_ENV="${CONDA_ENV:-cser}"
CSER_EPOCHS="${CSER_EPOCHS:-300}"
CSER_PHASE2_EPOCHS="${CSER_PHASE2_EPOCHS:-${CSER_EPOCHS}}"
CSER_PHASE3_EPOCHS="${CSER_PHASE3_EPOCHS:-${CSER_EPOCHS}}"
CSER_TRAIN_DEVICE="${CSER_TRAIN_DEVICE:-auto}"
CSER_TRAIN_BATCH_SIZE="${CSER_TRAIN_BATCH_SIZE:-256}"
SET_VALUE_TRAIN_BATCH_SIZE="${SET_VALUE_TRAIN_BATCH_SIZE:-128}"
CSER_BUDGET="${CSER_BUDGET:-5.0}"
CSER_ALPHA="${CSER_ALPHA:-0.05}"
PHASE2_CANDIDATE_TOP_K="${PHASE2_CANDIDATE_TOP_K:-500}"
PHASE2_TOPK_LIST="${PHASE2_TOPK_LIST:-100,300,500,1000}"
MIN_DELTA_SWEEP="${MIN_DELTA_SWEEP:-0.0,0.001,0.002,0.005}"
SET_VALUE_MAIN_MIN_DELTA="${SET_VALUE_MAIN_MIN_DELTA:-0.001}"
CAPTIONS_PER_VIDEO="${CAPTIONS_PER_VIDEO:-1}"
CSER_CACHE_CHECKPOINT_INTERVAL="${CSER_CACHE_CHECKPOINT_INTERVAL:-100}"

RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
RUN_PREPARE="${RUN_PREPARE:-1}"
RUN_PHASE1="${RUN_PHASE1:-1}"
RUN_PHASE2="${RUN_PHASE2:-1}"
RUN_MIN_DELTA_SWEEP="${RUN_MIN_DELTA_SWEEP:-1}"
RUN_PHASE3="${RUN_PHASE3:-1}"

ANNOTATION="${ANNOTATION:-${MSRVTT_ROOT}/annotation/MSR_VTT.json}"
VIDEOS_DIR="${VIDEOS_DIR:-${MSRVTT_ROOT}/videos/all}"
CSV="${CSV:-${DATA_DIR}/msrvtt10k_queries.csv}"
CSER_VIDEO_GALLERY="${CSER_VIDEO_GALLERY:-${DATA_DIR}/cser_gallery}"
CSER_EXPERT_CACHE="${CSER_EXPERT_CACHE:-${DATA_DIR}/cser_expert_cache}"

CURRENT_STAGE="bootstrap"

write_status() {
  local state="$1"
  local detail="$2"
  local rc="${3:-0}"
  mkdir -p "$(dirname "${STATUS_FILE}")"
  {
    printf 'updated_at=%s\n' "$(date --iso-8601=seconds)"
    printf 'state=%s\n' "${state}"
    printf 'detail=%s\n' "${detail}"
    printf 'exit_code=%s\n' "${rc}"
    printf 'job_id=%s\n' "${SLURM_JOB_ID:-manual}"
    printf 'node=%s\n' "${SLURMD_NODENAME:-unknown}"
    printf 'report_root=%s\n' "${REPORT_ROOT}"
    printf 'data_dir=%s\n' "${DATA_DIR}"
    printf 'annotation=%s\n' "${ANNOTATION}"
    printf 'videos_dir=%s\n' "${VIDEOS_DIR}"
    printf 'cser_video_gallery=%s\n' "${CSER_VIDEO_GALLERY}"
    printf 'cser_expert_cache=%s\n' "${CSER_EXPERT_CACHE}"
  } >"${STATUS_FILE}"
}

on_error() {
  local rc="$?"
  write_status failed "${CURRENT_STAGE}" "${rc}"
  printf '[failed] stage=%s exit_code=%s\n' "${CURRENT_STAGE}" "${rc}" >&2
  exit "${rc}"
}
trap on_error ERR

on_signal() {
  local signal="$1"
  local rc="$2"
  trap - ERR
  write_status cancelled "${CURRENT_STAGE}:${signal}" "${rc}"
  printf '[cancelled] stage=%s signal=%s exit_code=%s\n' \
    "${CURRENT_STAGE}" "${signal}" "${rc}" >&2
  exit "${rc}"
}
trap 'on_signal TERM 143' TERM
trap 'on_signal INT 130' INT

run_stage() {
  local label="$1"
  shift
  CURRENT_STAGE="${label}"
  write_status running "${CURRENT_STAGE}"
  printf '\n[%s] START %s\n' "$(date --iso-8601=seconds)" "${CURRENT_STAGE}"
  "$@"
  printf '[%s] DONE  %s\n' "$(date --iso-8601=seconds)" "${CURRENT_STAGE}"
}

require_path() {
  local path="$1"
  local label="$2"
  if [[ ! -e "${path}" ]]; then
    printf '[missing] %s: %s\n' "${label}" "${path}" >&2
    return 2
  fi
}

validate_inputs() {
  require_path "${CSV}" "CSER query csv"
  require_path "${CSER_VIDEO_GALLERY}" "CSER video gallery or manifest directory"
}

validate_raw_inputs() {
  require_path "${ANNOTATION}" "MSR-VTT annotation"
  require_path "${VIDEOS_DIR}" "MSR-VTT raw videos directory"
}

write_run_config() {
  mkdir -p "${REPORT_ROOT}"
  {
    printf 'repo_root=%s\n' "${REPO_ROOT}"
    printf 'data_dir=%s\n' "${DATA_DIR}"
    printf 'annotation=%s\n' "${ANNOTATION}"
    printf 'videos_dir=%s\n' "${VIDEOS_DIR}"
    printf 'csv=%s\n' "${CSV}"
    printf 'cser_video_gallery=%s\n' "${CSER_VIDEO_GALLERY}"
    printf 'cser_expert_cache=%s\n' "${CSER_EXPERT_CACHE}"
    printf 'captions_per_video=%s\n' "${CAPTIONS_PER_VIDEO}"
    printf 'cache_checkpoint_interval=%s\n' "${CSER_CACHE_CHECKPOINT_INTERVAL}"
    printf 'slurm_cpus_per_task=%s\n' "${SLURM_CPUS_PER_TASK:-unknown}"
    printf 'ort_gpu_overlay=%s\n' "${ORT_GPU_OVERLAY:-unset}"
    printf 'ort_intra_op_num_threads=%s\n' "${ORT_INTRA_OP_NUM_THREADS:-unset}"
    printf 'ort_inter_op_num_threads=%s\n' "${ORT_INTER_OP_NUM_THREADS:-unset}"
    printf 'cser_epochs=%s\n' "${CSER_EPOCHS}"
    printf 'cser_phase2_epochs=%s\n' "${CSER_PHASE2_EPOCHS}"
    printf 'cser_phase3_epochs=%s\n' "${CSER_PHASE3_EPOCHS}"
    printf 'cser_train_device=%s\n' "${CSER_TRAIN_DEVICE}"
    printf 'cser_train_batch_size=%s\n' "${CSER_TRAIN_BATCH_SIZE}"
    printf 'set_value_train_batch_size=%s\n' "${SET_VALUE_TRAIN_BATCH_SIZE}"
    printf 'cser_budget=%s\n' "${CSER_BUDGET}"
    printf 'cser_alpha=%s\n' "${CSER_ALPHA}"
    printf 'phase2_candidate_top_k=%s\n' "${PHASE2_CANDIDATE_TOP_K}"
    printf 'phase2_topk_list=%s\n' "${PHASE2_TOPK_LIST}"
    printf 'min_delta_sweep=%s\n' "${MIN_DELTA_SWEEP}"
  } >"${REPORT_ROOT}/run_config.env"
}

run_phase2_config() {
  local label="$1"
  local selector="$2"
  local roster="$3"
  local min_delta="$4"
  local safety_config="$5"
  local topk_list="$6"
  local out_dir="${REPORT_ROOT}/cser/phase2/${label}"

  local cmd=(
    python -B -m cser.run_phase2
    --videos "${CSER_VIDEO_GALLERY}"
    --csv "${CSV}"
    --real-models
    --gallery-cache "${CSER_EXPERT_CACHE}"
    --out-dir "${out_dir}"
    --epochs "${CSER_PHASE2_EPOCHS}"
    --train-device "${CSER_TRAIN_DEVICE}"
    --train-batch-size "${CSER_TRAIN_BATCH_SIZE}"
    --set-value-batch-size "${SET_VALUE_TRAIN_BATCH_SIZE}"
    --budget "${CSER_BUDGET}"
    --alpha "${CSER_ALPHA}"
    --candidate-top-k "${PHASE2_CANDIDATE_TOP_K}"
    --selector "${selector}"
    --expert-roster "${roster}"
    --safety-config "${safety_config}"
  )

  if [[ "${selector}" == "set_value_safe" ]]; then
    cmd+=(--min-delta "${min_delta}")
  fi
  if [[ -n "${topk_list}" ]]; then
    cmd+=(--candidate-top-k-list "${topk_list}")
  fi

  run_stage "${label}" "${cmd[@]}"
}

cd "${REPO_ROOT}"
mkdir -p "${REPORT_ROOT}" reports/setup

source /hpc2hdd/home/yyan047/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

ORT_GPU_OVERLAY="${ORT_GPU_OVERLAY:-${HOME}/.cache/cser/onnxruntime-gpu-1.26.0}"
if [[ -d "${ORT_GPU_OVERLAY}/onnxruntime" ]]; then
  export PYTHONPATH="${ORT_GPU_OVERLAY}:${PYTHONPATH:-}"
fi
NVIDIA_PYTHON_LIBS="$(
  find "${CONDA_PREFIX}/lib/python3.11/site-packages/nvidia" \
    -mindepth 2 -maxdepth 2 -type d -name lib -printf '%p:' 2>/dev/null || true
)"
export LD_LIBRARY_PATH="${NVIDIA_PYTHON_LIBS}${LD_LIBRARY_PATH:-}"

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export ORT_INTRA_OP_NUM_THREADS="${ORT_INTRA_OP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-1}}"
export ORT_INTER_OP_NUM_THREADS="${ORT_INTER_OP_NUM_THREADS:-1}"
export CSER_CACHE_CHECKPOINT_INTERVAL

INSIGHTFACE_WARNING_FILTERS="ignore::FutureWarning:insightface.utils.transform,ignore::FutureWarning:insightface.utils.face_align"
export PYTHONWARNINGS="${PYTHONWARNINGS:+${PYTHONWARNINGS},}${INSIGHTFACE_WARNING_FILTERS}"

export LITEVTR_MODEL_DIR="${LITEVTR_MODEL_DIR:-${REPO_ROOT}/models}"
export MOBILECLIP_CKPT="${MOBILECLIP_CKPT:-${REPO_ROOT}/models/mobileclip2/mobileclip2_s0.pt}"
export MOMENT_DETR_REPO="${MOMENT_DETR_REPO:-${REPO_ROOT}/models/moment_detr}"
export MOMENT_DETR_CKPT="${MOMENT_DETR_CKPT:-${REPO_ROOT}/models/moment_detr/run_on_video/moment_detr_ckpt/model_best.ckpt}"
export INSIGHTFACE_ROOT="${INSIGHTFACE_ROOT:-/hpc2hdd/home/yyan047/.insightface}"
export TORCH_HOME="${TORCH_HOME:-/hpc2hdd/home/yyan047/.cache/torch}"

write_status running bootstrap
write_run_config
printf '[job] id=%s node=%s cwd=%s report_root=%s\n' \
  "${SLURM_JOB_ID:-manual}" "${SLURMD_NODENAME:-unknown}" "${REPO_ROOT}" "${REPORT_ROOT}"

if [[ "${RUN_PREFLIGHT}" == "1" ]]; then
  run_stage gpu_check nvidia-smi
  run_stage static_expert_check \
    python -B scripts/check_cser_experts.py \
    --out reports/setup/cser_expert_status.json
  run_stage runtime_real_model_check \
    python -B scripts/check_cser_runtime_models.py
fi

if [[ "${RUN_PREPARE}" == "1" ]]; then
  run_stage validate_raw_inputs validate_raw_inputs
  run_stage prepare_msrvtt_real_10k \
    python -B scripts/prepare_msrvtt_real_10k.py \
    --annotation "${ANNOTATION}" \
    --videos-dir "${VIDEOS_DIR}" \
    --out-dir "${DATA_DIR}" \
    --captions-per-video "${CAPTIONS_PER_VIDEO}"
fi

run_stage validate_inputs validate_inputs

if [[ "${RUN_PHASE1}" == "1" ]]; then
  run_stage cser_phase1 \
    python -B -m cser.run_phase1 \
    --videos "${CSER_VIDEO_GALLERY}" \
    --csv "${CSV}" \
    --real-models \
    --gallery-cache "${CSER_EXPERT_CACHE}" \
    --out-dir "${REPORT_ROOT}/cser/phase1" \
    --epochs "${CSER_EPOCHS}" \
    --train-device "${CSER_TRAIN_DEVICE}" \
    --train-batch-size "${CSER_TRAIN_BATCH_SIZE}"
fi

if [[ "${RUN_PHASE2}" == "1" ]]; then
  run_phase2_config cser_phase2_marginal_value_all \
    marginal_value_greedy all 0.0 Mondrian_reduce "${PHASE2_TOPK_LIST}"

  run_phase2_config cser_phase2_marginal_density_all \
    marginal_density_greedy all 0.0 Mondrian_reduce ""

  run_phase2_config cser_phase2_set_value_safe_all \
    set_value_safe all "${SET_VALUE_MAIN_MIN_DELTA}" Mondrian_reduce ""

  run_phase2_config cser_phase2_set_value_safe_no_face_id \
    set_value_safe no_face_id "${SET_VALUE_MAIN_MIN_DELTA}" Mondrian_reduce ""

  run_phase2_config cser_phase2_set_value_safe_semantic_highlight_scene \
    set_value_safe semantic_highlight_scene "${SET_VALUE_MAIN_MIN_DELTA}" Mondrian_reduce ""

  run_phase2_config cser_phase2_set_value_safe_no_face_id_report \
    set_value_safe no_face_id "${SET_VALUE_MAIN_MIN_DELTA}" Mondrian_report ""

  if [[ "${RUN_MIN_DELTA_SWEEP}" == "1" ]]; then
    IFS=',' read -ra DELTAS <<<"${MIN_DELTA_SWEEP}"
    for delta in "${DELTAS[@]}"; do
      delta="${delta// /}"
      [[ -z "${delta}" ]] && continue
      delta_label="${delta//./p}"
      run_phase2_config "cser_phase2_set_value_safe_no_face_id_delta_${delta_label}" \
        set_value_safe no_face_id "${delta}" Mondrian_reduce ""
    done
  fi
fi

if [[ "${RUN_PHASE3}" == "1" ]]; then
  run_stage cser_phase3 \
    python -B -m cser.run_phase3 \
    --videos "${CSER_VIDEO_GALLERY}" \
    --csv "${CSV}" \
    --real-models \
    --gallery-cache "${CSER_EXPERT_CACHE}" \
    --out-dir "${REPORT_ROOT}/cser/phase3" \
    --epochs "${CSER_PHASE3_EPOCHS}" \
    --train-device "${CSER_TRAIN_DEVICE}" \
    --train-batch-size "${CSER_TRAIN_BATCH_SIZE}" \
    --budget "${CSER_BUDGET}" \
    --alpha "${CSER_ALPHA}" \
    --candidate-top-k "${PHASE2_CANDIDATE_TOP_K}"
fi

CURRENT_STAGE="complete"
write_status complete cser_experiments_complete
printf '\n[done] CSER-only MSR-VTT experiments: %s\n' "${REPORT_ROOT}"
