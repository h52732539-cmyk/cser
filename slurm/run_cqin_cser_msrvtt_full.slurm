#!/usr/bin/env bash
#SBATCH --job-name=cqin-cser-msrvtt-full  # 指定作业名称
#SBATCH -p i64m1tga40u         # 指定GPU队列
#SBATCH -o reports/setup/slurm-%x-%j.out  # 指定作业标准输出文件，%j为作业号
#SBATCH -e reports/setup/slurm-%x-%j.err  # 指定作业标准错误输出文件
#SBATCH -n 8            # 指定CPU总核心数
#SBATCH --gres=gpu:1    # 指定GPU卡数
#SBATCH --time=144:00:00  # 指定作业最长运行时间，格式为days-hours:minutes:seconds
#SBATCH -D /hpc2hdd/home/yyan047/xuhaodong/cser        # 指定作业执行路径为/apps


set -Eeuo pipefail

REPO_ROOT="/hpc2hdd/home/yyan047/xuhaodong/cser"
MSRVTT_ROOT="/hpc2hdd/home/yyan047/xuhaodong/MSRVTT"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data/msrvtt_real_1k}"
REPORT_ROOT="${REPORT_ROOT:-${REPO_ROOT}/reports/full/${SLURM_JOB_ID:-manual}}"
STATUS_FILE="${REPO_ROOT}/reports/setup/cqin_cser_full_latest.status"

CQIN_EPOCHS="${CQIN_EPOCHS:-200}"
CQIN_EXTRA_EPOCHS="${CQIN_EXTRA_EPOCHS:-150}"
CSER_EPOCHS="${CSER_EPOCHS:-300}"

ANNOTATION="${MSRVTT_ROOT}/annotation/MSR_VTT.json"
VIDEOS_DIR="${MSRVTT_ROOT}/videos/all"
VIDEO_LIST="${MSRVTT_ROOT}/structured-symlinks/test_list_miech.txt"
CACHE="${DATA_DIR}/msrvtt_cache.npz"
CSV="${DATA_DIR}/msrvtt_test_1k.csv"
TEXT_EMBS="${DATA_DIR}/msrvtt_test_1k.text_embs.npy"
CSER_GALLERY="${DATA_DIR}/cser_gallery"

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
  } >"${STATUS_FILE}"
}

on_error() {
  local rc="$?"
  write_status failed "${CURRENT_STAGE}" "${rc}"
  printf '[failed] stage=%s exit_code=%s\n' "${CURRENT_STAGE}" "${rc}" >&2
  exit "${rc}"
}
trap on_error ERR

run_stage() {
  local label="$1"
  shift
  CURRENT_STAGE="${label}"
  write_status running "${CURRENT_STAGE}"
  printf '\n[%s] START %s\n' "$(date --iso-8601=seconds)" "${CURRENT_STAGE}"
  "$@"
  printf '[%s] DONE  %s\n' "$(date --iso-8601=seconds)" "${CURRENT_STAGE}"
}

cd "${REPO_ROOT}"
mkdir -p "${REPORT_ROOT}" reports/setup

source /hpc2hdd/home/yyan047/miniconda3/etc/profile.d/conda.sh
conda activate cser
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export LITEVTR_MODEL_DIR="${REPO_ROOT}/models"
export MOBILECLIP_CKPT="${REPO_ROOT}/models/mobileclip2/mobileclip2_s0.pt"
export MOMENT_DETR_REPO="${REPO_ROOT}/models/moment_detr"
export MOMENT_DETR_CKPT="${REPO_ROOT}/models/moment_detr/run_on_video/moment_detr_ckpt/model_best.ckpt"
export INSIGHTFACE_ROOT="/hpc2hdd/home/yyan047/.insightface"
export TORCH_HOME="/hpc2hdd/home/yyan047/.cache/torch"

write_status running bootstrap
printf '[job] id=%s node=%s cwd=%s\n' \
  "${SLURM_JOB_ID:-manual}" "${SLURMD_NODENAME:-unknown}" "${REPO_ROOT}"

run_stage gpu_check nvidia-smi
run_stage static_expert_check \
  python -B scripts/check_cser_experts.py \
  --out reports/setup/cser_expert_status.json
run_stage runtime_real_model_check \
  python -B scripts/check_cser_runtime_models.py

# run_stage prepare_msrvtt_real_1k \
#   python -B scripts/prepare_msrvtt_real_1k.py \
#   --annotation "${ANNOTATION}" \
#   --videos-dir "${VIDEOS_DIR}" \
#   --video-list "${VIDEO_LIST}" \
#   --out-dir "${DATA_DIR}"

# run_stage cqin_final_eval \
#   python -B scripts/run_final_eval.py \
#   --cache "${CACHE}" \
#   --csv "${CSV}" \
#   --text-embs "${TEXT_EMBS}" \
#   --out-dir "${REPORT_ROOT}/cqin/final_eval" \
#   --epochs "${CQIN_EPOCHS}" \
#   --seeds 42 123 456 789 1024

# run_stage cqin_route_bank_ablation \
#   python -B scripts/run_route_bank_ablation.py \
#   --cache "${CACHE}" \
#   --csv "${CSV}" \
#   --text-embs "${TEXT_EMBS}" \
#   --out-dir "${REPORT_ROOT}/cqin/route_bank_ablation" \
#   --epochs "${CQIN_EXTRA_EPOCHS}"

# run_stage cqin_pareto_sweep \
#   python -B scripts/run_pareto_sweep.py \
#   --cache "${CACHE}" \
#   --csv "${CSV}" \
#   --text-embs "${TEXT_EMBS}" \
#   --out-dir "${REPORT_ROOT}/cqin/pareto_sweep" \
#   --epochs "${CQIN_EXTRA_EPOCHS}"

# run_stage cqin_head_analysis \
#   python -B scripts/run_head_analysis.py \
#   --cache "${CACHE}" \
#   --csv "${CSV}" \
#   --text-embs "${TEXT_EMBS}" \
#   --out-dir "${REPORT_ROOT}/cqin/head_analysis" \
#   --epochs "${CQIN_EXTRA_EPOCHS}"

# run_stage cqin_calibration_sweep \
#   python -B scripts/run_calibration_sweep.py \
#   --cache "${CACHE}" \
#   --csv "${CSV}" \
#   --text-embs "${TEXT_EMBS}" \
#   --out-dir "${REPORT_ROOT}/cqin/calibration_sweep" \
#   --epochs "${CQIN_EXTRA_EPOCHS}"

# run_stage cqin_training_sensitivity \
#   python -B scripts/run_training_sensitivity.py \
#   --cache "${CACHE}" \
#   --csv "${CSV}" \
#   --text-embs "${TEXT_EMBS}" \
#   --out-dir "${REPORT_ROOT}/cqin/training_sensitivity" \
#   --epochs "${CQIN_EXTRA_EPOCHS}" \
#   --n-seeds 3

run_stage cser_phase1 \
  python -B -m cser.run_phase1 \
  --videos "${CSER_GALLERY}" \
  --csv "${CSV}" \
  --real-models \
  --out-dir "${REPORT_ROOT}/cser/phase1" \
  --epochs "${CSER_EPOCHS}"

run_stage cser_phase2 \
  python -B -m cser.run_phase2 \
  --videos "${CSER_GALLERY}" \
  --csv "${CSV}" \
  --real-models \
  --out-dir "${REPORT_ROOT}/cser/phase2" \
  --epochs "${CSER_EPOCHS}"

run_stage cser_phase3 \
  python -B -m cser.run_phase3 \
  --videos "${CSER_GALLERY}" \
  --csv "${CSV}" \
  --real-models \
  --out-dir "${REPORT_ROOT}/cser/phase3" \
  --epochs "${CSER_EPOCHS}"

CURRENT_STAGE="complete"
write_status complete all_experiments_complete
printf '\n[done] CQIN and CSER full MSR-VTT experiments: %s\n' "${REPORT_ROOT}"
