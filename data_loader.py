from typing import List, Optional, Tuple, Union
import os
from pathlib import Path
import cv2
import numpy as np

PathLike = Union[str, Path]

def get_video_reader(
    video_path: PathLike,
    width: Optional[int] = None,
    height: Optional[int] = None,
    *,
    backend: str = "auto",
    **kwargs,
) -> cv2.VideoCapture:
    """Creates a video reader with a specific backend."""
    if backend == "auto":
        try:
            return cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
        except ImportError:
            return cv2.VideoCapture(str(video_path))
    elif backend == "ffmpeg":
        return cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
    elif backend == "opencv":
        return cv2.VideoCapture(str(video_path))
    else:
        raise ValueError(f"Unknown backend: {backend}")

def extract_frames(
    video_path: PathLike,
    *,
    max_frames: int,
    resize: Optional[Tuple[int, int]] = None,
    interpolation: int = cv2.INTER_AREA,
    backend: str = "auto",
) -> List[np.ndarray]:
    """
    Extracts frames from a video.

    Args:
        video_path: Path to the video file.
        max_frames: The maximum number of frames to extract.
        resize: A (width, height) tuple to resize frames.
        interpolation: OpenCV interpolation method (e.g., cv2.INTER_AREA).
        backend: The OpenCV backend to use.

    Returns:
        A list of extracted frames (as RGB numpy arrays).
    """
    cap = get_video_reader(video_path, backend=backend)

    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return []

    frames: List[np.ndarray] = []
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # If total_frames is 0 or less, or > max_frames, read frame by frame
    if total_frames <= 0 or total_frames > max_frames * 2:
        # Fallback for streams or incorrectly encoded videos
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if resize:
                frame = cv2.resize(frame, resize, interpolation=interpolation)

            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_count += 1
    else:
        # Efficiently sample frames if we know the total count
        indices = np.linspace(0, total_frames - 1, num=max_frames, dtype=int)

        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                continue

            if resize:
                frame = cv2.resize(frame, resize, interpolation=interpolation)

            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_count += 1

    cap.release()
    return frames