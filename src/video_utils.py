from pathlib import Path
from typing import List, Sequence, Tuple, Union

import cv2
import numpy as np


PathLike = Union[str, Path]


def extract_frames(video_path: PathLike, output_dir: PathLike) -> Tuple[int, float]:
	"""Extract all video frames to PNG files and return (frame_count, fps)."""
	video_path = Path(video_path)
	output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	cap = cv2.VideoCapture(str(video_path))
	if not cap.isOpened():
		raise ValueError(f"Could not open video: {video_path}")

	fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
	frame_count = 0

	try:
		while True:
			ok, frame = cap.read()
			if not ok:
				break

			frame_count += 1
			frame_path = output_dir / f"frame_{frame_count:05d}.png"
			if not cv2.imwrite(str(frame_path), frame):
				raise OSError(f"Failed to write frame file: {frame_path}")
	finally:
		cap.release()

	return frame_count, fps


def write_video(frames_list: Sequence[np.ndarray], output_path: PathLike, fps: float) -> None:
	"""Write a sequence of frames to an MP4 file using OpenCV and mp4v codec."""
	if not frames_list:
		raise ValueError("frames_list must contain at least one frame")

	output_path = Path(output_path)
	output_path.parent.mkdir(parents=True, exist_ok=True)

	first_frame = np.asarray(frames_list[0])
	if first_frame.ndim not in {2, 3}:
		raise ValueError("Each frame must be a 2D grayscale or 3D color array")

	height, width = first_frame.shape[:2]
	is_color = first_frame.ndim == 3 and first_frame.shape[2] == 3
	fourcc = cv2.VideoWriter_fourcc(*"mp4v")
	writer = cv2.VideoWriter(str(output_path), fourcc, float(fps), (width, height), isColor=is_color)

	if not writer.isOpened():
		raise ValueError(f"Could not open VideoWriter for output: {output_path}")

	try:
		for index, frame in enumerate(frames_list, start=1):
			arr = np.asarray(frame)

			if arr.shape[:2] != (height, width):
				raise ValueError(
					f"Frame {index} shape mismatch. "
					f"Expected {(height, width)}, got {arr.shape[:2]}"
				)

			if arr.ndim == 3 and arr.shape[2] == 1:
				arr = arr[:, :, 0]

			if arr.ndim == 2 and is_color:
				arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)

			if arr.ndim == 3 and not is_color:
				arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)

			if arr.dtype != np.uint8:
				arr = np.clip(arr, 0, 255).astype(np.uint8)

			writer.write(arr)
	finally:
		writer.release()
