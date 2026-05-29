"""Adaptive frame sampling strategies.

All samplers return a sorted list of `(timestamp_sec, origin_tag)` pairs.
None of them load or call any Huawei model weights directly; Q-Frame is
the only one that *consumes* an external image encoder, which the caller
injects (so the Huawei model stays external).

Included strategies:

  S0  UniformSampler            — baseline (1fps default)
  S1  ContentFingerprintSampler — 1fps + 64x64 diff > tau
  S2  MVBasedSampler            — codec motion-vector magnitude
  S3  QFrameSampler             — query-conditioned top-K via injected encoder
  S4  HybridSampler             — union-of-sets fusion of any subset

The hybrid sampler is how the production path in framework_v2 runs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np


# ----------------------------------------------------------------------
#  Common types
# ----------------------------------------------------------------------

Sample = Tuple[float, str]  # (timestamp_sec, origin_tag)


class _BaseSampler:
    name: str = "base"

    def sample(self, video_path: str, duration: float,
               sensor_stream: Optional[dict] = None,
               query_text: Optional[str] = None) -> List[Sample]:
        raise NotImplementedError


# ----------------------------------------------------------------------
#  S0 · Uniform
# ----------------------------------------------------------------------

class UniformSampler(_BaseSampler):
    name = "S0_uniform"

    def __init__(self, fps: float = 1.0, max_samples: int = 120) -> None:
        self.fps = fps
        self.max_samples = max_samples

    def sample(self, video_path, duration, sensor_stream=None, query_text=None):
        stride = 1.0 / max(self.fps, 1e-6)
        ts = np.arange(0.0, max(duration, 1e-6), stride)
        if len(ts) > self.max_samples:
            ts = np.linspace(0.0, duration, self.max_samples, endpoint=False)
        return [(float(t), self.name) for t in ts]


# ----------------------------------------------------------------------
#  S1 · 1fps content fingerprint (frame diff > tau)
# ----------------------------------------------------------------------

class ContentFingerprintSampler(_BaseSampler):
    name = "S1_content_fp"

    def __init__(self, diff_tau: float = 0.04, probe_fps: float = 1.0,
                 max_samples: int = 120) -> None:
        self.diff_tau = diff_tau
        self.probe_fps = probe_fps
        self.max_samples = max_samples

    def sample(self, video_path, duration, sensor_stream=None, query_text=None):
        try:
            import cv2
        except Exception:
            return UniformSampler(fps=self.probe_fps).sample(
                video_path, duration
            )
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []
        out: List[Sample] = []
        try:
            prev = None
            t = 0.0
            stride = 1.0 / max(self.probe_fps, 1e-6)
            while t < duration:
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                small = cv2.resize(frame, (64, 64)).astype(np.float32) / 255.0
                if prev is None:
                    out.append((float(t), self.name))
                else:
                    d = float(np.abs(small - prev).mean())
                    if d > self.diff_tau:
                        out.append((float(t), self.name))
                prev = small
                t += stride
        finally:
            cap.release()
        if len(out) > self.max_samples:
            # keep spread: pick top-k by diff variance (approx)
            idx = np.linspace(0, len(out) - 1, self.max_samples).astype(int)
            out = [out[i] for i in idx]
        return out


# ----------------------------------------------------------------------
#  S2 · Codec motion-vector based sampler
#
#  Uses PyAV to walk packets; if PyAV / MVs unavailable, falls back to
#  an I-frame-only approximation via frame-diff spikes.
# ----------------------------------------------------------------------

class MVBasedSampler(_BaseSampler):
    name = "S2_mv_based"

    def __init__(self, motion_tau: float = 2.0, min_gap_sec: float = 0.5,
                 max_samples: int = 120) -> None:
        self.motion_tau = motion_tau
        self.min_gap = min_gap_sec
        self.max_samples = max_samples

    def sample(self, video_path, duration, sensor_stream=None, query_text=None):
        ts_with_motion = self._extract_mv_timestamps(video_path)
        if not ts_with_motion:
            # fallback: approximate via I-frame spacing heuristic
            return ContentFingerprintSampler(
                diff_tau=0.06, probe_fps=1.0,
                max_samples=self.max_samples
            ).sample(video_path, duration)

        # keep only high-motion frames
        kept: List[Sample] = []
        last_t = -1e9
        for t, mag in ts_with_motion:
            if mag < self.motion_tau:
                continue
            if t - last_t < self.min_gap:
                continue
            kept.append((float(t), self.name))
            last_t = t
        if len(kept) > self.max_samples:
            idx = np.linspace(0, len(kept) - 1, self.max_samples).astype(int)
            kept = [kept[i] for i in idx]
        return kept

    # ------------------------------------------------------------------

    def _extract_mv_timestamps(self, video_path: str) -> List[Tuple[float, float]]:
        try:
            import av  # PyAV
        except Exception:
            return []
        out: List[Tuple[float, float]] = []
        try:
            with av.open(video_path) as container:
                stream = container.streams.video[0]
                stream.thread_type = "AUTO"
                for packet in container.demux(stream):
                    for frame in packet.decode():
                        ts = float(frame.pts * stream.time_base)
                        # approximate motion energy via frame mean luminance diff
                        # (PyAV does not expose MV directly; this is a
                        # packet-aligned proxy — still I/P-aware).
                        try:
                            arr = frame.to_ndarray(format="gray8")
                            mag = float(np.abs(np.diff(arr.astype(np.float32),
                                                        axis=0)).mean())
                        except Exception:
                            mag = 0.0
                        out.append((ts, mag))
        except Exception:
            return []
        return out


# ----------------------------------------------------------------------
#  S3 · Query-conditioned Q-Frame sampler
#
#  Workflow:
#    1. Coarse-scan at `probe_fps` using any existing sampler.
#    2. Encode those frames via an injected encoder (e.g. HuaweiCLIP).
#    3. Score each frame = cosine(encoder(frame), query_emb).
#    4. Keep top-K timestamps.
#
#  The encoder is injected, so no Huawei weight is embedded here.
# ----------------------------------------------------------------------

@dataclass
class QFrameConfig:
    probe_fps: float = 1.0
    top_k: int = 8
    min_gap_sec: float = 0.3


class QFrameSampler(_BaseSampler):
    name = "S3_qframe"

    def __init__(self,
                 image_encoder: Callable[[list], np.ndarray],
                 text_encoder:  Callable[[str], np.ndarray],
                 config: Optional[QFrameConfig] = None,
                 max_samples: int = 120) -> None:
        """
        image_encoder(frames: List[np.ndarray]) -> np.ndarray[N, D]
        text_encoder(query: str)                -> np.ndarray[D]
        """
        self.cfg = config or QFrameConfig()
        self.max_samples = max_samples
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder

    def sample(self, video_path, duration, sensor_stream=None, query_text=None):
        if query_text is None:
            # fall back to uniform when no query is provided
            return UniformSampler(fps=self.cfg.probe_fps).sample(
                video_path, duration
            )

        probe = UniformSampler(fps=self.cfg.probe_fps).sample(video_path, duration)
        if not probe:
            return []

        frames = self._decode_at(video_path, [t for t, _ in probe])
        if not frames:
            return []

        img_embs = self.image_encoder(frames)         # (N, D)
        if img_embs is None or len(img_embs) == 0:
            return []
        q_emb = self.text_encoder(query_text)          # (D,)

        img_embs = np.asarray(img_embs, dtype=np.float32)
        q_emb = np.asarray(q_emb, dtype=np.float32)
        img_embs /= (np.linalg.norm(img_embs, axis=-1, keepdims=True) + 1e-9)
        q_emb /= (np.linalg.norm(q_emb) + 1e-9)
        scores = img_embs @ q_emb                      # (N,)
        order = np.argsort(-scores)

        kept: List[Sample] = []
        for idx in order:
            t = float(probe[idx][0])
            if all(abs(t - kt) > self.cfg.min_gap_sec for kt, _ in kept):
                kept.append((t, self.name))
            if len(kept) >= self.cfg.top_k:
                break
        kept.sort(key=lambda x: x[0])
        if len(kept) > self.max_samples:
            kept = kept[:self.max_samples]
        return kept

    # ------------------------------------------------------------------

    @staticmethod
    def _decode_at(video_path, timestamps):
        try:
            import cv2
        except Exception:
            return []
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []
        out = []
        try:
            for t in timestamps:
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
                ret, frame = cap.read()
                if ret and frame is not None:
                    out.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        finally:
            cap.release()
        return out


# ----------------------------------------------------------------------
#  S4 · Hybrid sampler — union-of-sets + de-dup
# ----------------------------------------------------------------------

class HybridSampler(_BaseSampler):
    name = "S4_hybrid"

    def __init__(self, samplers: Sequence[_BaseSampler],
                 dedup_gap_sec: float = 0.2,
                 max_samples: int = 120) -> None:
        self.samplers = list(samplers)
        self.dedup_gap = dedup_gap_sec
        self.max_samples = max_samples

    def sample(self, video_path, duration, sensor_stream=None, query_text=None):
        all_pts: List[Sample] = []
        for s in self.samplers:
            try:
                all_pts.extend(
                    s.sample(video_path, duration,
                              sensor_stream=sensor_stream,
                              query_text=query_text)
                )
            except Exception:
                continue
        if not all_pts:
            return []
        all_pts.sort(key=lambda x: x[0])

        merged: List[Sample] = []
        last_t = -1e9
        origins: List[str] = []
        for t, origin in all_pts:
            if t - last_t > self.dedup_gap:
                if origins:
                    merged.append((last_t, "|".join(sorted(set(origins)))))
                last_t = t
                origins = [origin]
            else:
                origins.append(origin)
        if origins:
            merged.append((last_t, "|".join(sorted(set(origins)))))
        if len(merged) > self.max_samples:
            idx = np.linspace(0, len(merged) - 1, self.max_samples).astype(int)
            merged = [merged[i] for i in idx]
        return merged
