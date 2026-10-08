"""Headless --record smoke test: demo footage lands as a readable mp4."""

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).parents[1]


def test_demo_record_writes_valid_mp4(tmp_path):
    out = str(tmp_path / "footage.mp4")
    cmd = [
        sys.executable,
        "demo.py",
        "--headless",
        "--opponent",
        "random",
        "--points",
        "1",
        "--episodes",
        "1",
        "--fps",
        "100000",  # skip the render clock delay; playback fps unchanged
        "--record",
        out,
    ]
    subprocess.run(cmd, cwd=REPO, check=True, timeout=180, capture_output=True)
    assert os.path.getsize(out) > 10_000

    import imageio

    with imageio.get_reader(out) as reader:
        n_frames = reader.count_frames()
        first = reader.get_data(0)
    assert n_frames > 30  # a point takes at least a few dozen steps
    # ffmpeg pads the height to a multiple of 16 (600 -> 608).
    assert first.shape[0] in (600, 608) and first.shape[1] == 800 and first.shape[2] == 3
