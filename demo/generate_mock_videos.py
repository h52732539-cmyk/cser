"""Generate synthetic demo videos with varying content + sensor streams.

Each video contains alternating 'static' and 'active' segments so the
prefilter and two-stage logic can be exercised without needing real
footage.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np


def _draw_active(frame: np.ndarray, t: float) -> np.ndarray:
    """Fill frame with animated content."""
    h, w, _ = frame.shape
    # moving gradient background
    x = np.linspace(0, 1, w, dtype=np.float32)
    y = np.linspace(0, 1, h, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    r = (np.sin(xx * 6 + t * 1.3) * 0.5 + 0.5) * 255
    g = (np.sin(yy * 6 + t * 0.7) * 0.5 + 0.5) * 255
    b = (np.sin((xx + yy) * 4 + t) * 0.5 + 0.5) * 255
    frame[..., 0] = r.astype(np.uint8)
    frame[..., 1] = g.astype(np.uint8)
    frame[..., 2] = b.astype(np.uint8)

    # moving shape
    cx = int(w * (0.3 + 0.4 * np.sin(t * 1.5)))
    cy = int(h * (0.5 + 0.2 * np.cos(t * 1.1)))
    cv2.circle(frame, (cx, cy), 50, (255, 255, 255), -1)

    # periodic "face-like" warm patch (for MockFaceDetector)
    if int(t) % 8 in (2, 3, 4):
        fx, fy = int(w * 0.7), int(h * 0.35)
        cv2.ellipse(
            frame, (fx, fy), (60, 80), 0, 0, 360,
            (220, 170, 140), -1,
        )
    return frame


def _draw_static(frame: np.ndarray, t: float) -> np.ndarray:
    frame[:] = (80, 80, 80)
    cv2.putText(frame, "STATIC", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
    return frame


def synthesize_video(
    path: str,
    duration: float = 30.0,
    fps: int = 25,
    width: int = 320,
    height: int = 240,
    seed: int = 0,
) -> dict:
    """Write a synthetic .mp4 and return metadata including sensor stream."""
    rng = np.random.default_rng(seed)
    # build alternating segments: active [8s], static [4s], ...
    segments = []
    t = 0.0
    while t < duration:
        active_len = float(rng.uniform(4.0, 9.0))
        static_len = float(rng.uniform(2.0, 5.0))
        segments.append(("active", t, min(t + active_len, duration)))
        t += active_len
        if t >= duration:
            break
        segments.append(("static", t, min(t + static_len, duration)))
        t += static_len

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open writer for {path}")

    try:
        n_frames = int(duration * fps)
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        for i in range(n_frames):
            ts = i / fps
            kind = next((k for k, s, e in segments if s <= ts < e), "active")
            if kind == "active":
                _draw_active(frame, ts)
            else:
                _draw_static(frame, ts)
            writer.write(frame)
    finally:
        writer.release()

    # Build synthetic sensor stream matching the segment structure.
    gyro_fps = 200
    n_gyro = int(duration * gyro_fps)
    gyro = rng.normal(scale=0.30, size=(n_gyro, 3)).astype(np.float32)
    for kind, s, e in segments:
        if kind == "static":
            gs = int(s * gyro_fps)
            ge = int(e * gyro_fps)
            gyro[gs:ge] = rng.normal(scale=0.01, size=(ge - gs, 3))

    af_events = [float(e) for _, _, e in segments[:-1]]

    return {
        "path": path,
        "duration": duration,
        "fps": fps,
        "width": width,
        "height": height,
        "segments": [
            {"kind": k, "start": float(s), "end": float(e)}
            for k, s, e in segments
        ],
        "sensor": {
            "gyro": gyro.tolist(),
            "gyro_fps": gyro_fps,
            "af_events": af_events,
            "gyro_static_var": 0.01,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="demo/sample_videos")
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--min-duration", type=float, default=20.0)
    parser.add_argument("--max-duration", type=float, default=60.0)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)
    manifest = []
    for i in range(args.count):
        duration = float(rng.uniform(args.min_duration, args.max_duration))
        vid_path = str(out_dir / f"synth_{i:02d}.mp4")
        meta_path = str(out_dir / f"synth_{i:02d}.meta.json")
        print(f"[gen] {vid_path}  ({duration:.1f}s)")
        meta = synthesize_video(vid_path, duration=duration, seed=i)
        # Save sensor / segments separately (to keep videos small).
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "path": os.path.basename(vid_path),
                "duration": meta["duration"],
                "fps": meta["fps"],
                "width": meta["width"],
                "height": meta["height"],
                "segments": meta["segments"],
                "sensor": meta["sensor"],
            }, f)
        manifest.append({
            "id": f"synth_{i:02d}",
            "path": vid_path,
            "meta": meta_path,
            "duration": meta["duration"],
        })

    manifest_path = out_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\n[manifest] {manifest_path}  ({len(manifest)} videos)")


if __name__ == "__main__":
    main()
