"""Init for routing tests — includes shared test fixtures."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from core.offline_index import OfflineIndex, VideoIndexEntry, build_protos
from core.metadata import VideoMetadata


def _mk_test_index():
    """Create a tiny 3-video index for route executor tests."""
    D = 8

    def _entry(vid, idx, geo, motion):
        emb = np.zeros(D, dtype=np.float32)
        emb[idx] = 1.0
        emb = emb / (np.linalg.norm(emb) + 1e-9)
        fe = np.tile(emb, (4, 1)).astype(np.float32)
        meta = VideoMetadata(
            creation_time=100.0 + idx * 100,
            geo_category=geo,
            motion_class=motion,
            motion_confidence=0.9,
        )
        return VideoIndexEntry(
            video_id=vid, video_path="", duration=10.0,
            frame_embs=fe,
            protos={2: build_protos(fe, 2)},
            metadata=meta,
        )

    entries = [
        _entry("v0", 0, "coast", "running"),
        _entry("v1", 1, "mountain", "walking"),
        _entry("v2", 2, "urban", "stationary"),
    ]
    idx = OfflineIndex(entries=entries)
    gt_map = {"v0": 0, "v1": 1, "v2": 2}
    return idx, gt_map
