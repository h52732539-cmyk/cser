"""Metadata structures for time/geo/motion-aware retrieval.

Each video in the OfflineIndex can carry a `VideoMetadata` record. The
values here are extracted at indexing time from:

  1. MP4 container atoms (`©xyz` GPS, `creation_time`, `©mak/©mod` device)
  2. Sensor sidecar streams (gyroscope, accelerometer, AF events)
  3. File-system attributes (mtime fallback)

All fields are optional — the retrieval path gracefully skips absent
constraints. No Huawei model is invoked to build metadata.
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ----------------------------------------------------------------------
#  Motion classes (coarse — matches what most phone IMU streams expose)
# ----------------------------------------------------------------------

MOTION_CLASSES = [
    "stationary",   # |a| ~ 1g, gyro ~ 0
    "walking",      # periodic 1-2 Hz vertical oscillation
    "running",      # 2-4 Hz strong oscillation
    "cycling",      # sustained forward motion + mild periodic
    "vehicle",      # high translational, low gyro
    "unknown",
]


# ----------------------------------------------------------------------
#  Geo coarse categorisation
# ----------------------------------------------------------------------

GEO_CATEGORIES = [
    "indoor_home",     # low speed, no GPS drift, short residence
    "indoor_public",   # low speed, clustered
    "urban",           # dense commerce area (POI heuristic)
    "suburban",        # lower density
    "rural",           # sparse
    "coast",           # near ocean boundary
    "mountain",        # elevation > threshold
    "road",            # high linear speed
    "unknown",
]


# ----------------------------------------------------------------------
#  Data classes
# ----------------------------------------------------------------------

@dataclass
class VideoMetadata:
    """Per-video metadata record used by the retrieval layer.

    Every field is optional; missing metadata means "no constraint
    applies from this axis" at query time.
    """

    # Time
    creation_time: Optional[float] = None      # POSIX timestamp, UTC
    timezone_offset: Optional[float] = None    # minutes from UTC

    # Location
    latitude: Optional[float] = None           # decimal degrees
    longitude: Optional[float] = None
    altitude: Optional[float] = None           # metres
    geo_category: Optional[str] = None         # ∈ GEO_CATEGORIES

    # Device
    device_make: Optional[str] = None          # e.g. "HUAWEI"
    device_model: Optional[str] = None         # e.g. "Mate-60-Pro"

    # Motion (from IMU / gyro)
    motion_class: Optional[str] = None         # ∈ MOTION_CLASSES
    motion_confidence: Optional[float] = None  # 0..1
    avg_gyro_mag: Optional[float] = None       # mean |ω|
    avg_accel_mag: Optional[float] = None      # mean |a| minus 1g

    # Video intrinsics
    duration: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[float] = None
    rotation: Optional[int] = None             # 0/90/180/270

    # Derived / free-form tags (caller can attach scene tokens, etc.)
    tags: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "VideoMetadata":
        return cls(**{k: v for k, v in d.items()
                       if k in cls.__dataclass_fields__})


# ----------------------------------------------------------------------
#  Extraction — ffprobe-based (works for any MP4 with standard atoms)
# ----------------------------------------------------------------------

_ISO_RE = re.compile(
    r"(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2}):(\d{2})"
)
_ISO6709_RE = re.compile(r"([+-]\d+\.?\d*)([+-]\d+\.?\d*)([+-]\d+\.?\d*)?")


def _ffprobe_json(video_path: str) -> Dict:
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_format", "-show_streams", video_path],
            stderr=subprocess.DEVNULL, timeout=5.0,
        )
        return json.loads(out.decode("utf-8", errors="ignore"))
    except Exception:
        return {}


def _parse_iso_timestamp(s: str) -> Optional[float]:
    if not s:
        return None
    m = _ISO_RE.search(s)
    if not m:
        return None
    try:
        dt = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                       int(m.group(4)), int(m.group(5)), int(m.group(6)),
                       tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return None


def _parse_iso6709(s: str) -> Tuple[Optional[float], Optional[float],
                                     Optional[float]]:
    if not s:
        return None, None, None
    m = _ISO6709_RE.search(s.replace("/", ""))
    if not m:
        return None, None, None
    try:
        lat = float(m.group(1))
        lon = float(m.group(2))
        alt = float(m.group(3)) if m.group(3) else None
        return lat, lon, alt
    except Exception:
        return None, None, None


def extract_metadata(video_path: str,
                      sensor_sidecar: Optional[Dict] = None,
                      mtime_fallback: bool = True) -> VideoMetadata:
    """Extract VideoMetadata from an MP4 file + optional sensor sidecar."""
    meta = VideoMetadata()
    info = _ffprobe_json(video_path)
    fmt = info.get("format", {})
    tags = {k.lower(): v for k, v in fmt.get("tags", {}).items()}
    streams = info.get("streams", [])
    vstream = next((s for s in streams if s.get("codec_type") == "video"),
                   None)

    # Time
    t = (_parse_iso_timestamp(tags.get("creation_time", ""))
         or _parse_iso_timestamp(tags.get("date", "")))
    if t is None and mtime_fallback:
        try:
            t = os.path.getmtime(video_path)
        except Exception:
            t = None
    meta.creation_time = t

    # GPS (Apple / most Androids write '©xyz' as ISO-6709)
    loc = (tags.get("location") or tags.get("com.apple.quicktime.location.iso6709")
           or tags.get("location-eng") or "")
    lat, lon, alt = _parse_iso6709(loc)
    meta.latitude = lat
    meta.longitude = lon
    meta.altitude = alt

    # Device
    meta.device_make = tags.get("com.apple.quicktime.make") or tags.get("make")
    meta.device_model = tags.get("com.apple.quicktime.model") or tags.get("model")

    # Video intrinsics
    if vstream:
        try:
            meta.duration = float(vstream.get("duration") or fmt.get("duration", 0))
        except Exception:
            meta.duration = None
        meta.width = int(vstream.get("width") or 0) or None
        meta.height = int(vstream.get("height") or 0) or None
        fps_str = vstream.get("avg_frame_rate", "0/1")
        try:
            a, b = fps_str.split("/")
            meta.fps = float(a) / float(b) if float(b) else None
        except Exception:
            meta.fps = None
        try:
            meta.rotation = int(vstream.get("tags", {}).get("rotate", 0))
        except Exception:
            meta.rotation = None

    # Sensor-derived motion class
    if sensor_sidecar:
        motion = classify_motion_from_sensor(sensor_sidecar)
        meta.motion_class = motion["class"]
        meta.motion_confidence = motion["confidence"]
        meta.avg_gyro_mag = motion.get("gyro_mag")
        meta.avg_accel_mag = motion.get("accel_mag")

    return meta


# ----------------------------------------------------------------------
#  Motion classifier (rule-based, from IMU stream)
# ----------------------------------------------------------------------

def classify_motion_from_sensor(sensor: Dict) -> Dict:
    """Coarse motion classification from IMU stream.

    Expected sensor dict keys:
      - 'gyro'     : (N, 3) rad/s
      - 'gyro_fps' : float (sampling rate)
      - 'accel'    : (N, 3) m/s²  (optional)
      - 'accel_fps': float        (optional)

    Returns dict with 'class' ∈ MOTION_CLASSES, 'confidence' ∈ [0,1], and
    descriptive statistics.
    """
    try:
        import numpy as np
    except Exception:
        return {"class": "unknown", "confidence": 0.0}

    g = sensor.get("gyro")
    if g is None:
        return {"class": "unknown", "confidence": 0.0}
    g = np.asarray(g, dtype=float)
    if g.ndim != 2 or g.size < 6:
        return {"class": "unknown", "confidence": 0.0}

    gm = float(np.linalg.norm(g, axis=-1).mean())
    # Accel magnitude minus gravity (if available)
    a = sensor.get("accel")
    am: Optional[float] = None
    dominant_freq: Optional[float] = None
    if a is not None:
        a = np.asarray(a, dtype=float)
        if a.ndim == 2 and a.shape[0] > 4:
            mag = np.linalg.norm(a, axis=-1) - 9.80
            am = float(np.abs(mag).mean())
            fps_a = float(sensor.get("accel_fps", 50.0))
            # Identify dominant frequency by FFT of vertical component
            try:
                vert = mag - mag.mean()
                spec = np.abs(np.fft.rfft(vert))
                freqs = np.fft.rfftfreq(len(vert), d=1.0 / fps_a)
                if len(spec) > 1:
                    dominant_freq = float(freqs[1 + int(np.argmax(spec[1:]))])
            except Exception:
                pass

    # Decision tree
    if gm < 0.05 and (am is None or am < 0.3):
        return {"class": "stationary", "confidence": 0.9,
                "gyro_mag": gm, "accel_mag": am,
                "dominant_freq": dominant_freq}
    if dominant_freq is not None:
        if 2.2 <= dominant_freq <= 4.0 and (am or 0) > 1.5:
            return {"class": "running", "confidence": 0.85,
                    "gyro_mag": gm, "accel_mag": am,
                    "dominant_freq": dominant_freq}
        if 1.2 <= dominant_freq < 2.2 and (am or 0) > 0.5:
            return {"class": "walking", "confidence": 0.8,
                    "gyro_mag": gm, "accel_mag": am,
                    "dominant_freq": dominant_freq}
        if dominant_freq < 1.0 and gm < 0.2 and (am or 0) < 0.5:
            return {"class": "vehicle", "confidence": 0.6,
                    "gyro_mag": gm, "accel_mag": am,
                    "dominant_freq": dominant_freq}
    if 0.05 <= gm <= 0.5:
        return {"class": "walking", "confidence": 0.5,
                "gyro_mag": gm, "accel_mag": am,
                "dominant_freq": dominant_freq}
    if gm > 1.0:
        return {"class": "running", "confidence": 0.5,
                "gyro_mag": gm, "accel_mag": am,
                "dominant_freq": dominant_freq}
    return {"class": "unknown", "confidence": 0.3,
            "gyro_mag": gm, "accel_mag": am,
            "dominant_freq": dominant_freq}


# ----------------------------------------------------------------------
#  Geo categorisation (coarse rule + optional city bbox)
# ----------------------------------------------------------------------

# A tiny built-in table; real deployment would use a reverse-geocoding
# cache or an offline POI table.
_COAST_BBOXES = [
    # (min_lat, max_lat, min_lon, max_lon, label)
    (18.0, 20.0, 108.5, 111.0, "coast"),   # Hainan
    (36.5, 38.5, 120.0, 122.5, "coast"),   # Qingdao
    (22.0, 23.5, 113.5, 114.7, "coast"),   # Shenzhen bay
]
_MOUNTAIN_ALT_M = 800.0


def classify_geo(lat: Optional[float], lon: Optional[float],
                 alt: Optional[float] = None) -> str:
    if lat is None or lon is None:
        return "unknown"
    for (a, b, c, d, lbl) in _COAST_BBOXES:
        if a <= lat <= b and c <= lon <= d:
            return lbl
    if alt is not None and alt >= _MOUNTAIN_ALT_M:
        return "mountain"
    return "unknown"


def fill_derived_fields(meta: VideoMetadata) -> VideoMetadata:
    """Populate derived fields (geo_category) in-place and return."""
    if meta.geo_category is None:
        meta.geo_category = classify_geo(
            meta.latitude, meta.longitude, meta.altitude
        )
    return meta
