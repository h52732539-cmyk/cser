#!/usr/bin/env bash
#SBATCH --job-name=cser-cuda-probe
#SBATCH -p debug
#SBATCH -o reports/setup/slurm-%x-%j.out
#SBATCH -e reports/setup/slurm-%x-%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH -D /hpc2hdd/home/yyan047/xuhaodong/cser

set -Eeuo pipefail

REPO_ROOT="/hpc2hdd/home/yyan047/xuhaodong/cser"
cd "${REPO_ROOT}"

source /hpc2hdd/home/yyan047/miniconda3/etc/profile.d/conda.sh
conda activate cser

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
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

nvidia-smi --query-gpu=name,driver_version,memory.total \
  --format=csv,noheader

python - <<'PY'
import json
import time

import numpy as np
import onnxruntime as ort
import torch

from cser.set_value_network import SetValueNetwork
from cser.svn import SubmodularValueNetwork
from cser.train_set_value import SetValueTrainConfig, train_set_value
from cser.train_svn import SVNTrainConfig, train_svn
from cser.value_oracle import OracleLabels


def subset(labels, n=512):
    return OracleLabels(
        metric=labels.metric,
        query_feats=labels.query_feats[:n],
        value_matrix=labels.value_matrix[:n],
        marginal=labels.marginal[:n],
        optional_experts=labels.optional_experts,
    )


labels = subset(OracleLabels.load(
    "reports/cser_only/9814419/cser/phase1/oracle_train.npz"))
if not torch.cuda.is_available():
    raise SystemExit("torch.cuda.is_available() is false")

report = {
    "torch_version": torch.__version__,
    "torch_file": torch.__file__,
    "cuda_runtime": torch.version.cuda,
    "cuda_device": torch.cuda.get_device_name(0),
    "onnxruntime_version": ort.__version__,
    "onnxruntime_file": ort.__file__,
    "onnxruntime_providers": ort.get_available_providers(),
    "n_training_queries": labels.n_queries,
}

svn_seen = set()
svn_forward = SubmodularValueNetwork.forward


def checked_svn_forward(self, query_feat, selected_mask):
    parameter_device = next(self.parameters()).device
    assert parameter_device.type == "cuda", parameter_device
    assert query_feat.is_cuda, query_feat.device
    assert selected_mask.is_cuda, selected_mask.device
    svn_seen.add(str(parameter_device))
    return svn_forward(self, query_feat, selected_mask)


SubmodularValueNetwork.forward = checked_svn_forward
torch.cuda.reset_peak_memory_stats()
t0 = time.perf_counter()
svn, _ = train_svn(
    labels,
    SVNTrainConfig(
        epochs=1, patience=1, seed=42, device="cuda", batch_size=256),
    verbose=True,
)
report["svn"] = {
    "forward_devices": sorted(svn_seen),
    "train_seconds": time.perf_counter() - t0,
    "peak_cuda_memory_mb": torch.cuda.max_memory_allocated() / 1024**2,
    "returned_model_device": str(next(svn.parameters()).device),
}

set_value_seen = set()
set_value_forward = SetValueNetwork.forward


def checked_set_value_forward(self, query_feat):
    parameter_device = next(self.parameters()).device
    assert parameter_device.type == "cuda", parameter_device
    assert query_feat.is_cuda, query_feat.device
    set_value_seen.add(str(parameter_device))
    return set_value_forward(self, query_feat)


SetValueNetwork.forward = checked_set_value_forward
torch.cuda.reset_peak_memory_stats()
t0 = time.perf_counter()
set_value, _ = train_set_value(
    labels,
    SetValueTrainConfig(
        epochs=1, patience=1, seed=42, device="cuda", batch_size=256),
    verbose=True,
)
report["set_value"] = {
    "forward_devices": sorted(set_value_seen),
    "train_seconds": time.perf_counter() - t0,
    "peak_cuda_memory_mb": torch.cuda.max_memory_allocated() / 1024**2,
    "returned_model_device": str(next(set_value.parameters()).device),
}

print("CUDA_PROBE_RESULT=" + json.dumps(report, sort_keys=True))
PY
