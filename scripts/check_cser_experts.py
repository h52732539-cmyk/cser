"""Report CSER real-expert dependencies and local weight availability.

This check is intentionally static: it never downloads weights and never
constructs a model. Run it before ``--real-models`` experiments.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_ROOT = Path(os.environ.get("LITEVTR_MODEL_DIR", PROJECT_ROOT / "models"))


def _module_error(name: str) -> str | None:
    if importlib.util.find_spec(name) is None:
        return f"python package: {name}"
    try:
        importlib.import_module(name)
    except Exception as exc:
        return f"python package import failed: {name} ({type(exc).__name__}: {exc})"
    return None


def _entry(name: str, files=(), dirs=(), modules=(), note: str = "") -> dict:
    missing = []
    for module in modules:
        error = _module_error(module)
        if error:
            missing.append(error)
    for path in files:
        if not Path(path).is_file():
            missing.append(f"file: {path}")
    for path in dirs:
        if not Path(path).is_dir():
            missing.append(f"directory: {path}")
    return {
        "expert": name,
        "status": "ready" if not missing else "missing",
        "missing": missing,
        "note": note,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/setup/cser_expert_status.json")
    args = ap.parse_args()

    mobileclip = Path(os.environ.get(
        "MOBILECLIP_CKPT", MODEL_ROOT / "mobileclip2" / "mobileclip2_s0.pt"
    ))
    moment_repo = Path(os.environ.get("MOMENT_DETR_REPO", MODEL_ROOT / "moment_detr"))
    moment_ckpt = Path(os.environ.get(
        "MOMENT_DETR_CKPT",
        moment_repo / "run_on_video" / "moment_detr_ckpt" / "model_best.ckpt",
    ))
    insight_root = Path(os.environ.get("INSIGHTFACE_ROOT", Path.home() / ".insightface"))
    buffalo = insight_root / "models" / "buffalo_l"
    torch_cache = Path(os.environ.get(
        "TORCH_HOME", Path.home() / ".cache" / "torch"
    )) / "hub" / "checkpoints"
    mobilenet = torch_cache / "mobilenet_v3_small-047dcff4.pth"

    experts = [
        _entry("e0 semantic MobileCLIP2-S0", files=[mobileclip],
               modules=["open_clip"],
               note="Set MOBILECLIP_CKPT after downloading the checkpoint."),
        _entry("e1 highlight MomentDETR", files=[moment_ckpt], dirs=[moment_repo],
               note="Set MOMENT_DETR_REPO and MOMENT_DETR_CKPT."),
        _entry("e2 face SCRFD", files=[buffalo / "det_10g.onnx"],
               modules=["insightface", "onnxruntime"],
               note="Install InsightFace dependencies and unpack buffalo_l under INSIGHTFACE_ROOT."),
        _entry("e3 face_id ArcFace", files=[buffalo / "w600k_r50.onnx"],
               modules=["insightface", "onnxruntime"],
               note="ArcFace is included in the InsightFace buffalo_l model pack."),
        _entry("e4 scene MobileNetV3", files=[mobilenet], modules=["torchvision"],
               note="The ImageNet checkpoint may be placed in the Torch hub checkpoint cache."),
    ]
    report = {
        "all_ready": all(x["status"] == "ready" for x in experts),
        "model_root": str(MODEL_ROOT),
        "experts": experts,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    for expert in experts:
        print(f"[{expert['status'].upper():7}] {expert['expert']}")
        for missing in expert["missing"]:
            print(f"          - {missing}")
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
