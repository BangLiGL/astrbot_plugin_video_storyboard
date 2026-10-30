"""Internal scene sampling, storyboard rendering and AstrBot STT adaptation."""

import asyncio
import math
import subprocess
import threading
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageOps

from .bilibili import VideoError


def storyboard(video_path, directory, cells=12, cancel=None):
    """Detect scene changes and sample representative frames on a fixed budget."""
    cancel = cancel or threading.Event()
    capture = cv2.VideoCapture(str(video_path))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if not math.isfinite(fps) or fps <= 0 or count <= 0:
            raise VideoError("视频解码失败，无法取得有效时长")
        duration = count / fps
        if not math.isfinite(duration) or duration > 3600:
            raise VideoError("视频实际时长超过处理范围")
        cells = max(4, min(16, int(cells)))
        step = max(0.5, duration / 480)
        previous = None
        scores = []
        times = np.arange(0, max(0.1, duration - 0.05), step)
        for at in times:
            if cancel.is_set():
                raise VideoError("视频处理已取消")
            capture.set(cv2.CAP_PROP_POS_MSEC, float(at) * 1000)
            ok, frame = capture.read()
            if not ok:
                continue
            hsv = cv2.cvtColor(cv2.resize(frame, (64, 36)), cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv /= np.array([180.0, 255.0, 255.0], dtype=np.float32)
            score = float(np.mean(np.abs(hsv - previous))) if previous is not None else 0
            scores.append((float(at), score))
            previous = hsv
        if not scores:
            raise VideoError("无法解码视频画面")
        values = np.array([score for _, score in scores])
        threshold = max(
            0.07, float(np.median(values) + 3 * np.median(np.abs(values - np.median(values))))
        )
        cuts = []
        for at, score in scores:
            if score > threshold and (not cuts or at - cuts[-1] >= 0.8):
                cuts.append(at)
        # Uniform coverage is retained; cut times move boundaries within their bin.
        boundaries = [0.0]
        for index in range(1, cells):
            ideal = index * duration / cells
            nearby = [t for t in cuts if abs(t - ideal) < duration / cells * 0.3]
            boundaries.append(min(nearby, key=lambda t: abs(t - ideal)) if nearby else ideal)
        boundaries.append(duration)
        columns = 4 if cells > 9 else 3
        rows = math.ceil(cells / columns)
        width, height = 384, 240
        sheet = Image.new("RGB", (columns * width, rows * height), "#121720")
        draw = ImageDraw.Draw(sheet)
        observations = []
        for index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
            if cancel.is_set():
                raise VideoError("视频处理已取消")
            candidates = []
            for ratio in (0.25, 0.5, 0.75):
                at = min(duration - 0.05, start + (end - start) * ratio)
                capture.set(cv2.CAP_PROP_POS_MSEC, at * 1000)
                ok, frame = capture.read()
                if ok:
                    sharp = cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
                    candidates.append((sharp, at, frame))
            if not candidates:
                continue
            _, at, frame = max(candidates, key=lambda item: item[0])
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            image = ImageOps.contain(image, (width - 6, height - 34))
            x, y = (index % columns) * width, (index // columns) * height
            sheet.paste(image, (x + (width - image.width) // 2, y + 3))
            draw.rectangle((x + 5, y + 5, x + 36, y + 27), fill="#161c24")
            draw.text((x + 12, y + 8), str(index + 1), fill="#ffd465")
            draw.text(
                (x + 8, y + height - 24),
                f"{start:.1f}-{end:.1f}s | frame {at:.1f}s",
                fill="#d5dce8",
            )
            observations.append(
                {
                    "cell": index + 1,
                    "from": round(start, 2),
                    "to": round(end, 2),
                    "frame_at": round(at, 2),
                }
            )
        output = Path(directory) / "storyboard.jpg"
        sheet.save(output, "JPEG", quality=88)
        return {"image": str(output), "cells": observations, "cuts": cuts, "duration": duration}
    finally:
        capture.release()


async def transcribe(provider, media_path, directory, duration, timeout=60):
    """Use AstrBot's provider API; chunk boundaries are coarse, not word timestamps."""
    segments = []
    executable = imageio_ffmpeg.get_ffmpeg_exe()
    directory = Path(directory)
    for index, start in enumerate(range(0, math.ceil(duration), 30)):
        end = min(duration, start + 30)
        path = directory / f"asr-{index}.wav"
        proc = None
        try:
            kwargs = (
                {"creationflags": subprocess.CREATE_NO_WINDOW}
                if hasattr(subprocess, "CREATE_NO_WINDOW")
                else {}
            )
            proc = await asyncio.create_subprocess_exec(
                executable,
                "-nostdin",
                "-v",
                "error",
                "-y",
                "-ss",
                str(start),
                "-i",
                str(media_path),
                "-t",
                str(end - start),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                str(path),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
                **kwargs,
            )
            await asyncio.wait_for(proc.communicate(), timeout=30)
            if proc.returncode:
                raise VideoError("视频没有可转写的音轨或音频解码失败")
            text = await asyncio.wait_for(provider.get_text(str(path)), timeout=timeout)
            if str(text or "").strip():
                segments.append(
                    {
                        "from": start,
                        "to": end,
                        "text": str(text).strip()[:10000],
                        "timing": "chunk_approximate",
                    }
                )
        finally:
            if proc and proc.returncode is None:
                proc.kill()
                await proc.wait()
            path.unlink(missing_ok=True)
    return segments
