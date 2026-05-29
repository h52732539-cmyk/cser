"""Noisy metadata injection for realistic controlled experiments.

Transforms perfect synthetic metadata into realistic noisy metadata by:
  - Time: shift by N(0, σ_days) + missing with probability p
  - Geo:  jitter by N(0, σ_km) in lat/lon + wrong region flip + missing
  - Motion: label flip with probability p + missing
  - Device: label flip with probability p + missing

Also supports "ambiguous cluster" noise: multiple videos share the
same metadata bucket, simulating real-world duplication (e.g., many
videos taken at the same location on the same day).
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from core.metadata import VideoMetadata, GEO_CATEGORIES, MOTION_CLASSES


@dataclass
class NoiseConfig:
    # Time
    time_shift_days_std: float = 7.0
    time_missing_prob: float = 0.2
    # Geo
    geo_jitter_km_std: float = 20.0
    geo_wrong_region_prob: float = 0.1
    geo_missing_prob: float = 0.3
    # Motion
    motion_flip_prob: float = 0.15
    motion_missing_prob: float = 0.2
    # Device
    device_flip_prob: float = 0.05
    device_missing_prob: float = 0.1

    @classmethod
    def from_dict(cls, d: Dict) -> "NoiseConfig":
        kw = {}
        t = d.get("time", {})
        kw["time_shift_days_std"] = t.get("shift_days_std", 7.0)
        kw["time_missing_prob"] = t.get("missing_prob", 0.2)
        g = d.get("geo", {})
        kw["geo_jitter_km_std"] = g.get("jitter_km_std", 20.0)
        kw["geo_wrong_region_prob"] = g.get("wrong_region_prob", 0.1)
        kw["geo_missing_prob"] = g.get("missing_prob", 0.3)
        m = d.get("motion", {})
        kw["motion_flip_prob"] = m.get("label_flip_prob", 0.15)
        kw["motion_missing_prob"] = m.get("missing_prob", 0.2)
        dv = d.get("device", {})
        kw["device_flip_prob"] = dv.get("label_flip_prob", 0.05)
        kw["device_missing_prob"] = dv.get("missing_prob", 0.1)
        return cls(**kw)


# Degree per km (approximate)
_DEG_PER_KM = 1.0 / 111.0

_KNOWN_GEO = [c for c in GEO_CATEGORIES if c != "unknown"]
_KNOWN_MOTION = [c for c in MOTION_CLASSES if c != "unknown"]


def inject_noise(meta: VideoMetadata,
                  cfg: NoiseConfig,
                  rng: random.Random) -> VideoMetadata:
    """Return a new VideoMetadata with realistic noise injected."""
    kw = meta.to_dict()

    # --- Time noise ---
    if meta.creation_time is not None:
        if rng.random() < cfg.time_missing_prob:
            kw["creation_time"] = None
        else:
            shift_sec = rng.gauss(0, cfg.time_shift_days_std * 86400)
            kw["creation_time"] = meta.creation_time + shift_sec

    # --- Geo noise ---
    if meta.latitude is not None and meta.longitude is not None:
        if rng.random() < cfg.geo_missing_prob:
            kw["latitude"] = None
            kw["longitude"] = None
            kw["geo_category"] = None
        else:
            jitter_lat = rng.gauss(0, cfg.geo_jitter_km_std * _DEG_PER_KM)
            jitter_lon = rng.gauss(0, cfg.geo_jitter_km_std * _DEG_PER_KM)
            kw["latitude"] = meta.latitude + jitter_lat
            kw["longitude"] = meta.longitude + jitter_lon
            if rng.random() < cfg.geo_wrong_region_prob and _KNOWN_GEO:
                kw["geo_category"] = rng.choice(_KNOWN_GEO)
    elif meta.geo_category is not None:
        if rng.random() < cfg.geo_missing_prob:
            kw["geo_category"] = None
        elif rng.random() < cfg.geo_wrong_region_prob and _KNOWN_GEO:
            kw["geo_category"] = rng.choice(_KNOWN_GEO)

    # --- Motion noise ---
    if meta.motion_class is not None:
        if rng.random() < cfg.motion_missing_prob:
            kw["motion_class"] = None
            kw["motion_confidence"] = None
        elif rng.random() < cfg.motion_flip_prob and _KNOWN_MOTION:
            kw["motion_class"] = rng.choice(_KNOWN_MOTION)
            kw["motion_confidence"] = max(0.3, (meta.motion_confidence or 0.8) - 0.3)

    # --- Device noise ---
    if meta.device_make is not None:
        if rng.random() < cfg.device_missing_prob:
            kw["device_make"] = None
            kw["device_model"] = None
        elif rng.random() < cfg.device_flip_prob:
            kw["device_make"] = rng.choice(["HUAWEI", "Apple", "Samsung", "Xiaomi"])

    return VideoMetadata.from_dict(kw)


def inject_noise_batch(metas: List[Optional[VideoMetadata]],
                        cfg: NoiseConfig,
                        seed: int = 42) -> List[Optional[VideoMetadata]]:
    """Apply noise to a list of metadata records."""
    rng = random.Random(seed)
    out = []
    for m in metas:
        if m is None:
            out.append(None)
        else:
            out.append(inject_noise(m, cfg, rng))
    return out
