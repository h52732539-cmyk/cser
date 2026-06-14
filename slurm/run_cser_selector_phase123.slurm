#!/usr/bin/env bash
#SBATCH --job-name=cser-selector-p123
#SBATCH -p i64m1tga800u
#SBATCH -o reports/setup/slurm-%x-%j.out
#SBATCH -e reports/setup/slurm-%x-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=144:00:00
#SBATCH -D /hpc2hdd/home/yyan047/xuhaodong/cser

# Phase1/2/3 rerun for the repaired selector. The default path reuses the
# complete gallery cache and Phase1 train/test artifacts from job 9814419.

set -Eeuo pipefail

REPO_ROOT="${REPO_ROOT:-/hpc2hdd/home/yyan047/xuhaodong/cser}"
MSRVTT_ROOT="${MSRVTT_ROOT:-/hpc2hdd/home/yyan047/xuhaodong/MSRVTT}"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data/msrvtt_real_10k}"
REPORT_ROOT="${REPORT_ROOT:-${REPO_ROOT}/reports/cser_selector/${SLURM_JOB_ID:-manual}}"
SHARED_ARTIFACT_DIR="${SHARED_ARTIFACT_DIR:-${DATA_DIR}/selector_artifacts_v2}"
LEGACY_9814419_PHASE1="${LEGACY_9814419_PHASE1:-${REPO_ROOT}/reports/cser_only/9814419/cser/phase1}"

CONDA_ENV="${CONDA_ENV:-cser}"
CSV="${CSV:-${DATA_DIR}/msrvtt10k_queries.csv}"
CSER_VIDEO_GALLERY="${CSER_VIDEO_GALLERY:-${DATA_DIR}/cser_gallery}"
CSER_EXPERT_CACHE="${CSER_EXPERT_CACHE:-${DATA_DIR}/cser_expert_cache}"

CSER_EPOCHS="${CSER_EPOCHS:-300}"
CSER_TRAIN_DEVICE="${CSER_TRAIN_DEVICE:-cuda}"
CSER_TRAIN_BATCH_SIZE="${CSER_TRAIN_BATCH_SIZE:-256}"
SET_VALUE_TRAIN_BATCH_SIZE="${SET_VALUE_TRAIN_BATCH_SIZE:-128}"
CSER_BUDGET="${CSER_BUDGET:-5.0}"
CSER_ALPHA="${CSER_ALPHA:-0.05}"
PHASE2_CANDIDATE_TOP_K="${PHASE2_CANDIDATE_TOP_K:-500}"
PHASE2_TOPK_LIST="${PHASE2_TOPK_LIST:-100,300,500,1000}"
SET_VALUE_MIN_DELTA="${SET_VALUE_MIN_DELTA:-0.001}"

# Single Phase2 configuration switch. The two completed legacy selectors are
# intentionally absent from the default execution path.
PHASE2_CONFIG="${PHASE2_CONFIG:-set_value_safe_no_face_id}"
export PHASE2_CONFIG
export PATH="/opt/slurm/bin:${PATH}"

RUN_PHASE1="${RUN_PHASE1:-1}"
RUN_PHASE2="${RUN_PHASE2:-1}"
RUN_PHASE3="${RUN_PHASE3:-1}"
IMPORT_9814419_ARTIFACTS="${IMPORT_9814419_ARTIFACTS:-1}"

CURRENT_STAGE="bootstrap"

run_stage() {
  local label="$1"
  shift
  CURRENT_STAGE="${label}"
  printf '\n[%s] START %s\n' "$(date --iso-8601=seconds)" "${label}"
  "$@"
  printf '[%s] DONE  %s\n' "$(date --iso-8601=seconds)" "${label}"
}

on_error() {
  local rc="$?"
  printf '[failed] stage=%s exit_code=%s\n' "${CURRENT_STAGE}" "${rc}" >&2
  exit "${rc}"
}
trap on_error ERR

validate_complete_cache() {
  python - "${CSER_EXPERT_CACHE}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
cache = root / "gallery_cache.npz"
manifest_path = root / "manifest.json"
if not cache.is_file() or not manifest_path.is_file():
    raise SystemExit(f"missing complete gallery cache under {root}")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if not manifest.get("complete"):
    raise SystemExit(f"gallery cache is not complete: {manifest_path}")
if int(manifest.get("n_videos_loaded", 0)) != 10000:
    raise SystemExit("expected the complete 10K cache from job 9814419")
print(json.dumps(manifest, indent=2))
PY
}

import_legacy_artifacts() {
  [[ "${IMPORT_9814419_ARTIFACTS}" == "1" ]] || return 0
  mkdir -p "${SHARED_ARTIFACT_DIR}/svn"
  for name in oracle_train.npz oracle_test.npz; do
    if [[ ! -f "${SHARED_ARTIFACT_DIR}/${name}" ]]; then
      cp "${LEGACY_9814419_PHASE1}/${name}" "${SHARED_ARTIFACT_DIR}/${name}"
    fi
  done
  for name in svn.pt svn_config.json; do
    if [[ ! -f "${SHARED_ARTIFACT_DIR}/svn/${name}" ]]; then
      cp "${LEGACY_9814419_PHASE1}/svn/${name}" \
        "${SHARED_ARTIFACT_DIR}/svn/${name}"
    fi
  done
}

resolve_phase2_config() {
  case "${PHASE2_CONFIG}" in
    set_value_safe_no_face_id)
      SELECTOR="set_value_safe"
      EXPERT_ROSTER="no_face_id"
      ;;
    set_value_safe_semantic_highlight_scene)
      SELECTOR="set_value_safe"
      EXPERT_ROSTER="semantic_highlight_scene"
      ;;
    set_value_safe_all)
      SELECTOR="set_value_safe"
      EXPERT_ROSTER="all"
      ;;
    marginal_value_no_face_id)
      SELECTOR="marginal_value_greedy"
      EXPERT_ROSTER="no_face_id"
      ;;
    marginal_density_no_face_id)
      SELECTOR="marginal_density_greedy"
      EXPERT_ROSTER="no_face_id"
      ;;
    *)
      printf 'unknown PHASE2_CONFIG=%s\n' "${PHASE2_CONFIG}" >&2
      return 2
      ;;
  esac
}

cd "${REPO_ROOT}"
mkdir -p "${REPORT_ROOT}" "${SHARED_ARTIFACT_DIR}" reports/setup

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

export LITEVTR_MODEL_DIR="${LITEVTR_MODEL_DIR:-${REPO_ROOT}/models}"
export MOBILECLIP_CKPT="${MOBILECLIP_CKPT:-${REPO_ROOT}/models/mobileclip2/mobileclip2_s0.pt}"
export MOMENT_DETR_REPO="${MOMENT_DETR_REPO:-${REPO_ROOT}/models/moment_detr}"
export MOMENT_DETR_CKPT="${MOMENT_DETR_CKPT:-${REPO_ROOT}/models/moment_detr/run_on_video/moment_detr_ckpt/model_best.ckpt}"
export INSIGHTFACE_ROOT="${INSIGHTFACE_ROOT:-${HOME}/.insightface}"
export TORCH_HOME="${TORCH_HOME:-${HOME}/.cache/torch}"

resolve_phase2_config
CALIBRATION_ARGS=()
if [[ "${SELECTOR}" == "set_value_safe" ]]; then
  CALIBRATION_ARGS+=(--calibrate-min-delta)
fi
run_stage validate_complete_cache validate_complete_cache
run_stage import_9814419_artifacts import_legacy_artifacts
run_stage gpu_check nvidia-smi

# This check runs inside the allocated compute node after the exact Slurm
# environment above has been activated. Fail if the GPU provider is shadowed.
run_stage onnxruntime_cuda_check python - <<'PY'
import json
import onnxruntime as ort
import torch

info = {
    "version": ort.__version__,
    "file": ort.__file__,
    "providers": ort.get_available_providers(),
    "torch_version": torch.__version__,
    "torch_cuda_runtime": torch.version.cuda,
    "torch_cuda_available": torch.cuda.is_available(),
    "torch_cuda_device": (
        torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    ),
}
print(json.dumps(info, indent=2))
if "CUDAExecutionProvider" not in info["providers"]:
    raise SystemExit("CUDAExecutionProvider is unavailable")
if not info["torch_cuda_available"]:
    raise SystemExit("PyTorch CUDA is unavailable")
PY

git rev-parse HEAD >"${REPORT_ROOT}/git-sha.txt"
git status --short >"${REPORT_ROOT}/git-status.txt"
git diff --binary >"${REPORT_ROOT}/worktree.diff"
conda env export >"${REPORT_ROOT}/conda-environment.yml"
pip freeze >"${REPORT_ROOT}/pip-freeze.txt"
sha256sum \
  "${CSER_EXPERT_CACHE}/gallery_cache.npz" \
  "${CSV}" >"${REPORT_ROOT}/input-sha256.txt"

python - "${REPORT_ROOT}/run-manifest.json" <<'PY'
import json
import os
import platform
import socket
import subprocess
import sys

import onnxruntime as ort
import torch

manifest = {
    "job_id": os.environ.get("SLURM_JOB_ID"),
    "node": os.environ.get("SLURMD_NODENAME", socket.gethostname()),
    "git_sha": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip(),
    "python": platform.python_version(),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "onnxruntime_version": ort.__version__,
    "onnxruntime_file": ort.__file__,
    "onnxruntime_providers": ort.get_available_providers(),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "phase2_config": os.environ.get(
        "PHASE2_CONFIG", "set_value_safe_no_face_id"),
}
with open(sys.argv[1], "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=2, sort_keys=True)
PY

COMMON_ARGS=(
  --videos "${CSER_VIDEO_GALLERY}"
  --csv "${CSV}"
  --real-models
  --gallery-cache "${CSER_EXPERT_CACHE}"
  --artifact-dir "${SHARED_ARTIFACT_DIR}"
  --epochs "${CSER_EPOCHS}"
  --train-device "${CSER_TRAIN_DEVICE}"
  --train-batch-size "${CSER_TRAIN_BATCH_SIZE}"
)

if [[ "${RUN_PHASE1}" == "1" ]]; then
  run_stage phase1 python -B -m cser.run_phase1 \
    "${COMMON_ARGS[@]}" \
    --out-dir "${REPORT_ROOT}/cser/phase1"
fi

if [[ "${RUN_PHASE2}" == "1" ]]; then
  run_stage "phase2_${PHASE2_CONFIG}" python -B -m cser.run_phase2 \
    "${COMMON_ARGS[@]}" \
    --out-dir "${REPORT_ROOT}/cser/phase2/${PHASE2_CONFIG}" \
    --set-value-batch-size "${SET_VALUE_TRAIN_BATCH_SIZE}" \
    --budget "${CSER_BUDGET}" \
    --alpha "${CSER_ALPHA}" \
    --candidate-top-k "${PHASE2_CANDIDATE_TOP_K}" \
    --candidate-top-k-list "${PHASE2_TOPK_LIST}" \
    --selector "${SELECTOR}" \
    --expert-roster "${EXPERT_ROSTER}" \
    --min-delta "${SET_VALUE_MIN_DELTA}" \
    "${CALIBRATION_ARGS[@]}" \
    --safety-config Mondrian_reduce
fi

if [[ "${RUN_PHASE3}" == "1" ]]; then
  run_stage phase3 python -B -m cser.run_phase3 \
    "${COMMON_ARGS[@]}" \
    --out-dir "${REPORT_ROOT}/cser/phase3" \
    --set-value-batch-size "${SET_VALUE_TRAIN_BATCH_SIZE}" \
    --budget "${CSER_BUDGET}" \
    --alpha "${CSER_ALPHA}" \
    --candidate-top-k "${PHASE2_CANDIDATE_TOP_K}" \
    --selector "${SELECTOR}" \
    --expert-roster "${EXPERT_ROSTER}" \
    --min-delta "${SET_VALUE_MIN_DELTA}" \
    "${CALIBRATION_ARGS[@]}"
fi

printf '\n[done] repaired selector Phase1/2/3: %s\n' "${REPORT_ROOT}"
