"""Temporary prediction stub for UI development.

Replace this with the real frame interpolation implementation after training.
"""

from __future__ import annotations

from typing import Callable


def interpolate_video(
	input_video: str,
	output_video: str,
	model_path: str,
	factor: int = 2,
	progress_cb: Callable[[int, int], None] | None = None,
	cancel_flag=None,
) -> None:
	import time

	import cv2

	# Keep these args in signature for drop-in compatibility with the real implementation.
	_ = (output_video, model_path, factor)

	cap = cv2.VideoCapture(input_video)
	total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	cap.release()

	for i in range(total):
		time.sleep(0.01)
		if cancel_flag and cancel_flag.is_set():
			raise InterruptedError
		if progress_cb:
			progress_cb(i + 1, total)
