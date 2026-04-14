from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def format_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{num_bytes} B"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    spec_path = root / "vfi.spec"
    pyinstaller_work = root / "build" / "pyinstaller"
    pyinstaller_dist = root / "build" / "pyinstaller-dist"
    final_dist = root / "dist"

    if not spec_path.exists():
        raise FileNotFoundError(f"Spec file not found: {spec_path}")

    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--clean",
        "--workpath",
        str(pyinstaller_work),
        "--distpath",
        str(pyinstaller_dist),
        str(spec_path),
    ]

    print("Running PyInstaller...")
    subprocess.run(command, check=True)

    source_exe = pyinstaller_dist / "VFI.exe"
    if not source_exe.exists():
        raise FileNotFoundError(f"Build output not found: {source_exe}")

    final_dist.mkdir(parents=True, exist_ok=True)
    final_exe = final_dist / "VFI.exe"
    shutil.copy2(source_exe, final_exe)

    size = final_exe.stat().st_size
    print(f"Built executable: {final_exe}")
    print(f"File size: {format_size(size)}")


if __name__ == "__main__":
    main()