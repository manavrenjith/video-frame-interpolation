import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Set


PEXELS_SEARCH_URL = "https://api.pexels.com/videos/search"
CATEGORIES = ["nature", "city", "people", "sports", "animals"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download free stock videos from Pexels into data/raw_videos/."
    )
    parser.add_argument(
        "--count",
        type=int,
        default=30,
        help="Total number of videos to download (default: 30)",
    )
    return parser.parse_args()


def pick_mp4_url(video: Dict) -> Optional[str]:
    files = video.get("video_files", [])
    mp4_files = [f for f in files if str(f.get("file_type", "")).lower() == "video/mp4"]
    if not mp4_files:
        return None

    # Prefer the highest-resolution MP4 variant when multiple are available.
    best = max(mp4_files, key=lambda f: (int(f.get("width") or 0), int(f.get("height") or 0)))
    return best.get("link")


def fetch_videos(api_key: str, query: str, page: int, per_page: int) -> List[Dict]:
    params = urllib.parse.urlencode({"query": query, "per_page": per_page, "page": page})
    url = f"{PEXELS_SEARCH_URL}?{params}"
    request = urllib.request.Request(url, headers={"Authorization": api_key})

    with urllib.request.urlopen(request, timeout=30) as response:
        payload = response.read().decode("utf-8")
    data = json.loads(payload)
    return data.get("videos", [])


def download_file(api_key: str, url: str, destination: Path) -> None:
    request = urllib.request.Request(url, headers={"Authorization": api_key})
    with urllib.request.urlopen(request, timeout=120) as response:
        destination.write_bytes(response.read())


def main() -> int:
    args = parse_args()
    if args.count <= 0:
        print("Error: --count must be a positive integer.")
        return 1

    api_key = os.getenv("PEXELS_API_KEY")
    if not api_key:
        print("Error: PEXELS_API_KEY environment variable is required.")
        return 1

    project_root = Path(__file__).resolve().parents[1]
    raw_videos_dir = project_root / "data" / "raw_videos"
    raw_videos_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    skipped = 0
    seen_video_ids: Set[int] = set()
    category_page = {category: 1 for category in CATEGORIES}
    category_index = 0

    print(f"Target video count: {args.count}")
    print(f"Download directory: {raw_videos_dir}")

    while downloaded + skipped < args.count:
        category = CATEGORIES[category_index % len(CATEGORIES)]
        page = category_page[category]
        category_page[category] += 1
        category_index += 1

        try:
            videos = fetch_videos(api_key, category, page=page, per_page=15)
        except urllib.error.HTTPError as exc:
            print(f"[ERROR] Pexels API HTTP error for '{category}' page {page}: {exc}")
            continue
        except urllib.error.URLError as exc:
            print(f"[ERROR] Network error while querying Pexels: {exc}")
            continue
        except json.JSONDecodeError:
            print("[ERROR] Failed to parse API response from Pexels.")
            continue

        if not videos:
            print(f"[WARN] No videos returned for '{category}' page {page}; trying next category.")
            continue

        for video in videos:
            if downloaded + skipped >= args.count:
                break

            video_id = int(video.get("id", 0))
            if video_id <= 0 or video_id in seen_video_ids:
                continue
            seen_video_ids.add(video_id)

            mp4_url = pick_mp4_url(video)
            if not mp4_url:
                print(f"[SKIP] Video {video_id}: no MP4 source available.")
                skipped += 1
                continue

            file_path = raw_videos_dir / f"pexels_{video_id}.mp4"
            progress = downloaded + skipped + 1

            if file_path.exists():
                print(f"[{progress}/{args.count}] Skip existing: {file_path.name}")
                skipped += 1
                continue

            try:
                print(f"[{progress}/{args.count}] Downloading: {file_path.name} ({category})")
                download_file(api_key, mp4_url, file_path)
                downloaded += 1
            except urllib.error.HTTPError as exc:
                print(f"[ERROR] HTTP error downloading {file_path.name}: {exc}")
                skipped += 1
            except urllib.error.URLError as exc:
                print(f"[ERROR] Network error downloading {file_path.name}: {exc}")
                skipped += 1
            except OSError as exc:
                print(f"[ERROR] Could not write {file_path.name}: {exc}")
                skipped += 1

    print("\nDone.")
    print(f"Downloaded: {downloaded}")
    print(f"Skipped/failed: {skipped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
