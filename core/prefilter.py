"""Metadata prefilter: produces a candidate-frame mask and scene hints.

Sources supported:
  1. Optional sensor stream dict (gyroscope / AF events).
  2. Lightweight content fingerprint: 1fps coarse decode, 64x64 frame-diff.

No dependencies on proprietary Huawei APIs; sensor stream is pluggable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class PrefilterResult:
    candidate_mask: np.ndarray  # [T] bool at 100ms resolution
    scene_boundaries: List[float] = field(default_factory=list)
    static_segments: List[Tuple[float, float]] = field(default_factory=list)

    def num_candidates(self) -> int:
        return int(self.candidate_mask.sum())

    def mask_resolution_hz(self) -> float:
        return 10.0  # 100 ms buckets


class MetadataPrefilter:
    """Fast metadata-based frame pruning.

    Args:
        use_sensor: consume the optional sensor_stream dict when present.
        use_content_fingerprint: run a 1fps coarse decode to detect static
            segments and scene boundaries from frame diffs.
        static_threshold: mean-abs-diff below which a frame is 'static'.
        min_static_duration: static segments shorter than this are ignored.
        boundary_threshold: mean-abs-diff above which a frame boundary is
            flagged as a scene change.
    """

    def __init__(
        self,
        use_sensor: bool = True,
        use_content_fingerprint: bool = True,
        static_threshold: float = 0.015,
        min_static_duration: float = 2.0,
        boundary_threshold: float = 0.15,
    ) -> None:
        self.use_sensor = use_sensor
        self.use_content_fingerprint = use_content_fingerprint
        self.static_threshold = static_threshold
        self.min_static_duration = min_static_duration
        self.boundary_threshold = boundary_threshold

    # ---- public API ---------------------------------------------------

    def analyze(
        self,
        video_path: str,
        duration: float,
        sensor_stream: Optional[dict] = None,
    ) -> PrefilterResult:
        T = max(1, int(duration * 10))  # 100ms buckets
        candidate_mask = np.ones(T, dtype=bool)
        static_segments: List[Tuple[float, float]] = []
        scene_boundaries: List[float] = []

        if self.use_sensor and sensor_stream is not None:
            s_static = self._detect_static_from_sensor(sensor_stream, duration)
            s_bnd = self._detect_scenes_from_af(sensor_stream, duration)
            static_segments.extend(s_static)
            scene_boundaries.extend(s_bnd)

        if self.use_content_fingerprint:
            fp_static, fp_bnd = self._content_fingerprint(video_path, duration)
            static_segments.extend(fp_static)
            scene_boundaries.extend(fp_bnd)

        for start, end in static_segments:
            if end - start < self.min_static_duration:
                continue
            si = max(0, int(start * 10))
            ei = min(T, int(end * 10))
            if ei > si + 1:
                # keep the first bucket as representative, drop the rest
                candidate_mask[si + 1:ei] = False

        for t in scene_boundaries:
            idx = int(t * 10)
            if 0 <= idx < T:
                candidate_mask[idx] = True

        return PrefilterResult(
            candidate_mask=candidate_mask,
            scene_boundaries=sorted(set(round(x, 2) for x in scene_boundaries)),
            static_segments=static_segments,
        )

    # ---- sensor heuristics -------------------------------------------

    def _detect_static_from_sensor(
        self, stream: dict, duration: float
    ) -> List[Tuple[float, float]]:
        gyro = stream.get("gyro")
        if gyro is None:
            return []
        gyro = np.asarray(gyro, dtype=np.float32)
        fps = float(stream.get("gyro_fps", 200.0))
        if gyro.ndim != 2 or gyro.shape[0] < 2:
            return []
        mag = np.linalg.norm(gyro, axis=1)
        win = max(4, int(fps * 0.5))
        step = max(1, win // 2)
        var = np.array([
            mag[i:i + win].var()
            for i in range(0, max(1, len(mag) - win), step)
        ])
        static_thr = float(stream.get("gyro_static_var", 0.01))
        static_mask = var < static_thr

        segs: List[Tuple[float, float]] = []
        cur_start: Optional[float] = None
        for i, s in enumerate(static_mask):
            t = i * step / fps
            if s and cur_start is None:
                cur_start = t
            elif (not s) and cur_start is not None:
                segs.append((cur_start, t))
                cur_start = None
        if cur_start is not None:
            segs.append((cur_start, duration))
        return segs

    def _detect_scenes_from_af(
        self, stream: dict, duration: float
    ) -> List[float]:
        events = stream.get("af_events")
        if not events:
            return []
        return [float(t) for t in events if 0.0 <= float(t) <= duration]

    # ---- content fingerprint ------------------------------------------

    def _content_fingerprint(
        self, video_path: str, duration: float
    ) -> Tuple[List[Tuple[float, float]], List[float]]:
        try:
            import cv2
        except Exception:
            return [], []

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return [], []

        try:
            prev_small: Optional[np.ndarray] = None
            static_segs: List[Tuple[float, float]] = []
            boundaries: List[float] = []
            static_start: Optional[float] = None

            t = 0.0
            while t < duration:
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                small = cv2.resize(frame, (64, 64)).astype(np.float32) / 255.0
                if prev_small is not None:
                    diff = float(np.abs(small - prev_small).mean())
                    if diff < self.static_threshold:
                        if static_start is None:
                            static_start = max(0.0, t - 1.0)
                    else:
                        if static_start is not None:
                            static_segs.append((static_start, t))
                            static_start = None
                        if diff > self.boundary_threshold:
                            boundaries.append(t)
                prev_small = small
                t += 1.0

            if static_start is not None:
                static_segs.append((static_start, duration))
            return static_segs, boundaries
        finally:
            cap.release()
