import shutil
from pathlib import Path
from typing import Dict, List

import cv2


MIN_FPS = 24.0
MIN_WIDTH = 1280
MIN_HEIGHT = 720
MIN_DURATION_SECONDS = 5.0


def iter_video_files(raw_dir: Path) -> List[Path]:
    return sorted(
        p
        for p in raw_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    )


def validate_video(video_path: Path) -> Dict:
    result = {
        "filename": video_path.name,
        "fps": 0.0,
        "resolution": "N/A",
        "duration": 0.0,
        "pass": False,
        "reason": "unknown",
    }

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        result["reason"] = "cannot_open"
        return result

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0

    ok, _ = cap.read()
    cap.release()

    duration = (frame_count / fps) if fps > 0 else 0.0

    result["fps"] = float(fps)
    result["resolution"] = f"{width}x{height}"
    result["duration"] = float(duration)

    if not ok:
        result["reason"] = "corrupt_or_empty"
        return result
    if fps < MIN_FPS:
        result["reason"] = "low_fps"
        return result
    if width < MIN_WIDTH or height < MIN_HEIGHT:
        result["reason"] = "low_resolution"
        return result
    if duration < MIN_DURATION_SECONDS:
        result["reason"] = "short_duration"
        return result

    result["pass"] = True
    result["reason"] = "ok"
    return result


def move_to_rejected(video_path: Path, rejected_dir: Path) -> Path:
    rejected_dir.mkdir(parents=True, exist_ok=True)
    target = rejected_dir / video_path.name

    if not target.exists():
        shutil.move(str(video_path), str(target))
        return target

    stem = video_path.stem
    suffix = video_path.suffix
    index = 1
    while True:
        candidate = rejected_dir / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            shutil.move(str(video_path), str(candidate))
            return candidate
        index += 1


def print_summary(results: List[Dict]) -> None:
    header = (
        f"{'filename':<36} {'fps':>8} {'resolution':>12} {'duration_s':>12} {'status':>8}"
    )
    print(header)
    print("-" * len(header))

    for row in results:
        status = "PASS" if row["pass"] else "FAIL"
        print(
            f"{row['filename']:<36} "
            f"{row['fps']:>8.2f} "
            f"{row['resolution']:>12} "
            f"{row['duration']:>12.2f} "
            f"{status:>8}"
        )


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    raw_dir = project_root / "data" / "raw_videos"
    rejected_dir = raw_dir / "rejected"

    raw_dir.mkdir(parents=True, exist_ok=True)
    video_files = iter_video_files(raw_dir)

    if not video_files:
        print(f"No video files found in: {raw_dir}")
        return 0

    results: List[Dict] = []
    valid_count = 0
    rejected_count = 0

    for video_path in video_files:
        result = validate_video(video_path)
        results.append(result)

        if result["pass"]:
            valid_count += 1
        else:
            move_to_rejected(video_path, rejected_dir)
            rejected_count += 1

    print_summary(results)
    print("\nValidation complete.")
    print(f"Total valid: {valid_count}")
    print(f"Total rejected: {rejected_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
