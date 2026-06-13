#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPORT_DIR="${REPO_ROOT}/reports/setup"
STATUS_FILE="${REPORT_DIR}/cser_download_status.txt"
LOG_FILE="${REPORT_DIR}/cser_download_tmux.log"
INSIGHTFACE_ROOT="/hpc2hdd/home/yyan047/.insightface"
TORCH_CACHE="/hpc2hdd/home/yyan047/.cache/torch/hub/checkpoints"
CLIP_CACHE="/hpc2hdd/home/yyan047/.cache/clip"
DOWNLOAD_FAILED=0

mkdir -p "${REPORT_DIR}" "${REPO_ROOT}/models/mobileclip2" \
  "${INSIGHTFACE_ROOT}/models" "${TORCH_CACHE}" "${CLIP_CACHE}"

exec >>"${LOG_FILE}" 2>&1

write_status() {
  local state="$1"
  local detail="$2"
  {
    printf 'updated_at=%s\n' "$(date --iso-8601=seconds)"
    printf 'state=%s\n' "${state}"
    printf 'detail=%s\n' "${detail}"
    printf 'tmux_session=cser-downloads\n'
    printf 'tmux_window=experts\n'
    printf 'log_file=%s\n' "${LOG_FILE}"
    printf 'mobileclip2_s0=%s\n' "$([[ -s "${REPO_ROOT}/models/mobileclip2/mobileclip2_s0.pt" ]] && printf ready || printf missing)"
    printf 'moment_detr=%s\n' "$([[ -s "${REPO_ROOT}/models/moment_detr/run_on_video/moment_detr_ckpt/model_best.ckpt" ]] && printf ready || printf missing)"
    printf 'scrfd=%s\n' "$([[ -s "${INSIGHTFACE_ROOT}/models/buffalo_l/det_10g.onnx" ]] && printf ready || printf missing)"
    printf 'arcface=%s\n' "$([[ -s "${INSIGHTFACE_ROOT}/models/buffalo_l/w600k_r50.onnx" ]] && printf ready || printf missing)"
    printf 'mobilenet_v3=%s\n' "$([[ -s "${TORCH_CACHE}/mobilenet_v3_small-047dcff4.pth" ]] && printf ready || printf missing)"
    printf 'openai_clip_vit_b32=%s\n' "$([[ -s "${CLIP_CACHE}/ViT-B-32.pt" ]] && printf ready || printf missing)"
  } >"${STATUS_FILE}"
}

run_step() {
  local label="$1"
  shift
  write_status running "${label}"
  printf '\n[%s] START %s\n' "$(date --iso-8601=seconds)" "${label}"
  if "$@"; then
    printf '[%s] DONE  %s\n' "$(date --iso-8601=seconds)" "${label}"
    return 0
  fi
  printf '[%s] FAIL  %s\n' "$(date --iso-8601=seconds)" "${label}"
  write_status failed "${label}"
  DOWNLOAD_FAILED=1
  return 0
}

download_if_missing() {
  local target="$1"
  local url="$2"
  if [[ -s "${target}" ]]; then
    printf 'Already present: %s\n' "${target}"
    return 0
  fi
  curl -fL --retry 3 -C - "${url}" -o "${target}"
}

download_with_sha256() {
  local target="$1"
  local url="$2"
  local expected_sha256="$3"
  if [[ -s "${target}" ]] &&
    printf '%s  %s\n' "${expected_sha256}" "${target}" | sha256sum -c -; then
    printf 'Already present with valid SHA256: %s\n' "${target}"
    return 0
  fi
  curl -fL --retry 3 -C - "${url}" -o "${target}"
}

write_status running initialization
printf '\n[%s] CSER expert download started\n' "$(date --iso-8601=seconds)"

run_step mobileclip2_s0 \
  hf download apple/MobileCLIP2-S0 mobileclip2_s0.pt \
  --local-dir "${REPO_ROOT}/models/mobileclip2"

if [[ ! -d "${REPO_ROOT}/models/moment_detr/.git" ]]; then
  run_step moment_detr_clone \
    git clone https://github.com/jayleicn/moment_detr.git \
    "${REPO_ROOT}/models/moment_detr"
fi

run_step insightface_buffalo_l_zip \
  download_if_missing \
  "${INSIGHTFACE_ROOT}/models/buffalo_l.zip" \
  https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip

run_step insightface_buffalo_l_extract \
  unzip -o "${INSIGHTFACE_ROOT}/models/buffalo_l.zip" \
  -d "${INSIGHTFACE_ROOT}/models/buffalo_l"

run_step mobilenet_v3_small \
  download_if_missing \
  "${TORCH_CACHE}/mobilenet_v3_small-047dcff4.pth" \
  https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth

run_step openai_clip_vit_b32 \
  download_with_sha256 \
  "${CLIP_CACHE}/ViT-B-32.pt" \
  https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt \
  40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af

run_step openai_clip_vit_b32_sha256 \
  bash -c "printf '%s  %s\n' \
    40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af \
    '${CLIP_CACHE}/ViT-B-32.pt' | sha256sum -c -"

run_step cser_expert_static_check \
  /hpc2hdd/home/yyan047/miniconda3/bin/conda run -n cser \
  python "${REPO_ROOT}/scripts/check_cser_experts.py" \
  --out "${REPORT_DIR}/cser_expert_status.json"

if [[ "${DOWNLOAD_FAILED}" -eq 0 ]]; then
  write_status complete all_downloads_complete
  printf '[%s] CSER expert download completed\n' "$(date --iso-8601=seconds)"
else
  write_status failed one_or_more_steps_failed
  printf '[%s] CSER expert download finished with failures\n' "$(date --iso-8601=seconds)"
  exit 1
fi
