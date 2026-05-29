"""Video decoding utilities (OpenCV-based)."""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from .types import Frame, FrameRequest


def decode_frames(
    video_path: str,
    requests: List[FrameRequest],
    cache=None,
) -> List[Frame]:
    """Decode the requested frames using OpenCV, optionally via cache.

    Returns frames in the same order as requests. Failed decodes are skipped.
    """
    if not requests:
        return []
    try:
        import cv2
    except Exception as e:
        raise RuntimeError("opencv-python is required for decoding") from e

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    frames: List[Frame] = []
    try:
        for req in requests:
            if cache is not None:
                cached = cache.get(req.frame_idx)
                if cached is not None:
                    frames.append(Frame(
                        video_id=req.video_id,
                        frame_idx=req.frame_idx,
                        timestamp=req.timestamp,
                        image=cached,
                        stage=req.stage,
                    ))
                    continue

            cap.set(cv2.CAP_PROP_POS_MSEC, req.timestamp * 1000.0)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if cache is not None:
                cache.put(req.frame_idx, rgb)
            frames.append(Frame(
                video_id=req.video_id,
                frame_idx=req.frame_idx,
                timestamp=req.timestamp,
                image=rgb,
                stage=req.stage,
            ))
    finally:
        cap.release()

    return frames


def probe_video(video_path: str) -> dict:
    """Return {'fps', 'n_frames', 'duration', 'width', 'height'}."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"fps": 0.0, "n_frames": 0, "duration": 0.0,
                "width": 0, "height": 0}
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = n / fps if fps > 0 else 0.0
        return {
            "fps": float(fps),
            "n_frames": n,
            "duration": float(duration),
            "width": w,
            "height": h,
        }
    finally:
        cap.release()
