# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SAOS-3624
# Copyright 2026 BangLiGL
"""Bounded decoding adapted from dsh-video-storyboard v0.6.2 scan/frame concepts.

Python subprocess lifecycle, PTS matching and AstrBot resource limits are local
adaptations. See NOTICE and LICENSES/Apache-2.0.txt.
"""

import contextlib
import math
import re
import subprocess
import threading
import time

import cv2
import imageio_ffmpeg
import numpy as np

from .bilibili import VideoError
from .sampling import hsv_distance, normalize_changes, scene_cuts


@contextlib.contextmanager
def process(arguments, cancel, timeout=90, timestamps=None):
    kwargs = (
        {"creationflags": subprocess.CREATE_NO_WINDOW}
        if hasattr(subprocess, "CREATE_NO_WINDOW")
        else {}
    )
    child = subprocess.Popen(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-nostdin", *arguments],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE if timestamps is not None else subprocess.DEVNULL,
        **kwargs,
    )
    done = threading.Event()
    deadline = time.monotonic() + timeout
    expired = threading.Event()

    def watch():
        while not done.wait(0.05):
            if cancel.is_set() or time.monotonic() >= deadline:
                expired.set()
                with contextlib.suppress(OSError):
                    child.kill()
                return

    def read_times():
        # Drain diagnostics, retaining only selected-frame timestamps, never paths/metadata.
        for line in iter(child.stderr.readline, b""):
            match = re.search(rb"\bpts_time:([\d.eE+-]+)", line[:8192])
            if match and len(timestamps) < 32:
                timestamps.append(float(match[1]))

    watcher = threading.Thread(target=watch, daemon=True)
    reader = threading.Thread(target=read_times, daemon=True) if timestamps is not None else None
    watcher.start()
    if reader:
        reader.start()
    try:
        yield child
        code = child.wait(timeout=2)
        if reader:
            reader.join(timeout=2)
        if cancel.is_set():
            raise VideoError("视频处理已取消")
        if expired.is_set():
            raise VideoError("视频解码超时")
        if code:
            raise VideoError("视频解码失败")
    finally:
        done.set()
        if child.poll() is None:
            with contextlib.suppress(OSError):
                child.kill()
        child.wait()
        watcher.join(timeout=1)
        child.stdout.close()
        if reader:
            reader.join(timeout=2)
            child.stderr.close()


def frame_bytes(stream, size):
    chunks, length = [], 0
    while length < size:
        block = stream.read(size - length)
        if not block:
            break
        chunks.append(block)
        length += len(block)
    if not length:
        return None
    if length != size:
        raise VideoError("视频解码返回了不完整的画面")
    return b"".join(chunks)


def scan_video(path, fps, duration, cancel):
    # At most 14,400 tiny frames; long/high-FPS sources cannot grow memory unboundedly.
    rate = min(fps, 24.0, 14400 / duration)
    means, rows, peaks = [], np.zeros(36), np.zeros(36)
    scores, long_scores, ring = [], [], []
    lag = max(1, round(0.4 * rate))
    previous = None
    with process(
        [
            "-v",
            "error",
            "-threads",
            "2",
            "-skip_loop_filter",
            "all",
            "-i",
            str(path),
            "-t",
            str(duration),
            "-an",
            "-sn",
            "-filter_threads",
            "1",
            "-vf",
            f"fps={rate:.9f}:start_time=0,scale=64:36:flags=area,format=bgr24",
            "-frames:v",
            "14400",
            "-f",
            "rawvideo",
            "-",
        ],
        cancel,
    ) as child:
        for index in range(14400):
            raw = frame_bytes(child.stdout, 64 * 36 * 3)
            if raw is None:
                break
            if cancel.is_set():
                raise VideoError("视频处理已取消")
            frame = np.frombuffer(raw, np.uint8).reshape(36, 64, 3)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
            value = hsv[..., 2]
            means.append(float(value.mean()))
            rows += value.mean(axis=1)
            peaks = np.maximum(peaks, value.max(axis=1))
            at = index / rate
            if previous is not None:
                scores.append((at, hsv_distance(previous, hsv)))
            if len(ring) == lag:
                long_scores.append((at, hsv_distance(ring[index % lag], hsv)))
                ring[index % lag] = hsv
            else:
                ring.append(hsv)
            previous = hsv
    if not means:
        raise VideoError("无法解码视频画面")
    dark = (rows / len(means) < 8) & (peaks < 24)
    top = next((i for i, v in enumerate(dark) if not v), 36)
    bottom = next((i for i, v in enumerate(dark[::-1]) if not v), 36)
    bars = (top + bottom) / 36
    bars = bars if 0 < bars < 0.5 else 0
    normalized = normalize_changes(scores, means, bars, offset=1)
    normalized_long = normalize_changes(long_scores, means, bars, offset=lag)
    cuts = scene_cuts(scores, duration, normalized=normalized, long_scores=normalized_long)
    return {
        "scores": normalized,
        "cuts": cuts,
        "interval": 1 / rate,
        "frames": len(means),
        "bars": round(bars, 3),
        "rate": rate,
    }


def extract_frames(path, targets, size, aspect, cancel):
    """Decode sparse time windows in bounded batches, matching actual decoded PTS."""
    width, height = size
    width = max(1, min(width, int(height * aspect)))
    height = max(1, min(height, int(width / aspect)))
    # At most 32 MiB of decoded pixels per batch, and no more than 8 targets.
    capacity = max(1, min(8, (32 * 1048576) // (width * height * 3)))
    batches = []
    for target in sorted(set(targets)):
        if not batches or len(batches[-1]) >= capacity or target - batches[-1][0] > 12:
            batches.append([])
        batches[-1].append(target)
    found = {}
    for batch in batches:
        start = max(0, batch[0] - 0.1)
        selectors = [
            f"gte(t,{at - start:.9f})*(isnan(prev_t)+lt(prev_t,{at - start:.9f}))" for at in batch
        ]
        stamps, decoded = [], []
        with process(
            [
                "-v",
                "info",
                "-threads",
                "2",
                "-ss",
                f"{start:.9f}",
                "-i",
                str(path),
                "-t",
                f"{batch[-1] - start + 1:.9f}",
                "-an",
                "-sn",
                "-filter_threads",
                "1",
                "-vf",
                f"select='{'+'.join(selectors)}',scale={width}:{height}:flags=area,showinfo",
                "-fps_mode",
                "passthrough",
                "-pix_fmt",
                "bgr24",
                "-f",
                "rawvideo",
                "-",
            ],
            cancel,
            timestamps=stamps,
        ) as child:
            for _ in range(capacity + 1):
                raw = frame_bytes(child.stdout, width * height * 3)
                if raw is None:
                    break
                decoded.append(np.frombuffer(raw, np.uint8).reshape(height, width, 3).copy())
            else:
                raise VideoError("视频取帧超过批量上限")
        if len(stamps) != len(decoded):
            raise VideoError("视频帧时间与画面未能对应，不能错配时间线")
        for target in batch:
            selected = next(
                (
                    (start + t, frame)
                    for t, frame in zip(stamps, decoded)
                    if math.isfinite(t) and start + t >= target - 0.0001
                ),
                None,
            )
            if selected:
                found[target] = selected
    return found
