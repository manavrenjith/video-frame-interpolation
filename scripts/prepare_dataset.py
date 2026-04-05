from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from typing import List

import cv2


VALID_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def list_video_files(raw_videos_dir: Path) -> List[Path]:
    return sorted(
        p for p in raw_videos_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    )


def save_triplet(frame_a: Path, frame_b: Path, frame_c: Path, triplet_dir: Path) -> None:
    triplet_dir.mkdir(parents=True, exist_ok=True)

    image_a = cv2.imread(str(frame_a), cv2.IMREAD_COLOR)
    image_b = cv2.imread(str(frame_b), cv2.IMREAD_COLOR)
    image_c = cv2.imread(str(frame_c), cv2.IMREAD_COLOR)

    if image_a is None or image_b is None or image_c is None:
        raise ValueError(f"Could not read one or more frame files for triplet: {triplet_dir.name}")

    if not cv2.imwrite(str(triplet_dir / "A.png"), image_a):
        raise OSError(f"Failed writing A.png in {triplet_dir}")
    if not cv2.imwrite(str(triplet_dir / "B.png"), image_b):
        raise OSError(f"Failed writing B.png in {triplet_dir}")
    if not cv2.imwrite(str(triplet_dir / "C.png"), image_c):
        raise OSError(f"Failed writing C.png in {triplet_dir}")


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.video_utils import extract_frames

    raw_videos_dir = project_root / "data" / "raw_videos"
    triplets_dir = project_root / "data" / "triplets"

    raw_videos_dir.mkdir(parents=True, exist_ok=True)
    triplets_dir.mkdir(parents=True, exist_ok=True)

    videos = list_video_files(raw_videos_dir)
    if not videos:
        print(f"No valid videos found in {raw_videos_dir}")
        return 0

    total_triplets_created = 0
    total_triplets_skipped = 0

    print(f"Found {len(videos)} videos in {raw_videos_dir}")

    for video_index, video_path in enumerate(videos, start=1):
        video_tag = f"video{video_index:03d}"
        print(f"[{video_index}/{len(videos)}] Processing {video_path.name} -> {video_tag}")

        with TemporaryDirectory(prefix=f"{video_tag}_frames_") as temp_dir:
            frame_output_dir = Path(temp_dir)

            try:
                frame_count, fps = extract_frames(video_path, frame_output_dir)
            except Exception as exc:
                print(f"  [ERROR] Failed extracting frames from {video_path.name}: {exc}")
                continue

            print(f"  Extracted {frame_count} frames at {fps:.2f} FPS")

            frame_files = sorted(frame_output_dir.glob("frame_*.png"))
            if len(frame_files) < 3:
                print("  Not enough frames to form triplets; skipping video.")
                continue

            video_created = 0
            video_skipped = 0

            for i in range(len(frame_files) - 2):
                triplet_name = f"{video_tag}_triplet_{i + 1:04d}"
                triplet_dir = triplets_dir / triplet_name

                if triplet_dir.exists():
                    video_skipped += 1
                    total_triplets_skipped += 1
                    continue

                try:
                    save_triplet(frame_files[i], frame_files[i + 1], frame_files[i + 2], triplet_dir)
                    video_created += 1
                    total_triplets_created += 1
                except Exception as exc:
                    print(f"  [ERROR] Failed creating {triplet_name}: {exc}")

            print(f"  Triplets created: {video_created}, skipped existing: {video_skipped}")

    print("\nDataset preparation complete.")
    print(f"Total triplets created: {total_triplets_created}")
    print(f"Total triplets skipped: {total_triplets_skipped}")
    print(f"Triplets directory: {triplets_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())