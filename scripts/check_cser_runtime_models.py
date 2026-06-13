"""Construct CSER's five experts and fail if any adapter falls back to mocks."""
from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cser.expert_features import build_model_bundle


def main() -> None:
    bundle = build_model_bundle(use_real=True)
    models = {
        "e0_semantic_mobileclip2_s0": bundle.clip,
        "e1_highlight_moment_detr": bundle.highlight,
        "e2_face_scrfd": bundle.face_det,
        "e3_face_id_arcface": bundle.face_emb,
        "e4_scene_mobilenet_v3": bundle.scene,
    }
    report = {
        name: {
            "module": model.__class__.__module__,
            "class": model.__class__.__name__,
        }
        for name, model in models.items()
    }
    bad = [
        name
        for name, model in models.items()
        if model.__class__.__module__ != "tasks.real_models"
    ]
    print(json.dumps(report, indent=2))
    if bad:
        raise RuntimeError(f"real-model preflight fell back to mocks: {bad}")

    out = Path("reports/setup/cser_runtime_model_status.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"all_real": True, "models": report}, indent=2),
        encoding="utf-8",
    )
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
