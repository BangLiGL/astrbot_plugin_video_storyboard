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
from .sampling import density_boundaries, hsv_distance, plan_images, readability, scene_cuts


def storyboard(video_path, directory, cells=12, cancel=None, options=None):
    """Read a bounded chronological storyboard; each cell is one sampled instant."""
    cancel = cancel or threading.Event()
    options = options or {}
    capture = cv2.VideoCapture(str(video_path))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if not math.isfinite(fps) or not math.isfinite(count) or fps <= 0 or count <= 0:
            raise VideoError("视频解码失败，无法取得有效时长")
        duration = count / fps
        if not math.isfinite(duration) or duration > 3600:
            raise VideoError("视频实际时长超过处理范围")
        source_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        source_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if source_w <= 0 or source_h <= 0:
            raise VideoError("视频解码失败，无法取得画面尺寸")
        plan = plan_images(
            cells,
            source_w / source_h,
            options.get("storyboard_pixel_budget", 1200000),
            options.get("storyboard_readability", "scene"),
            options.get("storyboard_max_per_sheet", 0),
            options.get("storyboard_max_images", 4),
        )
        step = max(1 / fps, 0.2, duration / 900)
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
            score = hsv_distance(previous, hsv) if previous is not None else 0
            scores.append((float(at), score))
            previous = hsv
        if not scores:
            raise VideoError("无法解码视频画面")
        cuts = scene_cuts(scores, duration)
        boundaries = density_boundaries(scores, duration, plan["actual_cells"])
        observations = []
        images, sheets = [], []
        index = 0
        for page, (number, canvas) in enumerate(zip(plan["counts"], plan["layouts"]), 1):
            sheet = Image.new("RGB", (canvas["width"], canvas["height"]), "#121720")
            draw = ImageDraw.Draw(sheet)
            width, height = canvas["cell_width"] + 6, canvas["cell_height"] + 34
            for local in range(number):
                if cancel.is_set():
                    raise VideoError("视频处理已取消")
                start, end = boundaries[index : index + 2]
                candidates = []
                for ratio in (0.2, 0.5, 0.8):
                    at = min(max(0, duration - 1 / fps), start + (end - start) * ratio)
                    capture.set(cv2.CAP_PROP_POS_MSEC, at * 1000)
                    ok, frame = capture.read()
                    if ok:
                        position = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
                        if math.isfinite(position) and (position > 0 or at == 0):
                            at = position
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        dark = bool(np.mean(gray < 12) > 0.995)
                        sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
                        candidates.append((not dark, sharp, at, frame, dark))
                if not candidates:
                    plan["warnings"].append(f"第{index + 1}格解码失败，不代表该时段没有内容")
                    index += 1
                    continue
                _, _, at, frame, dark = max(candidates, key=lambda item: item[:2])
                image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                image = ImageOps.contain(
                    image,
                    (
                        min(canvas["cell_width"], image.width),
                        min(canvas["cell_height"], image.height),
                    ),
                )
                x, y = local % canvas["columns"] * width, local // canvas["columns"] * height
                sheet.paste(image, (x + (width - image.width) // 2, y + 3))
                draw.rectangle((x + 5, y + 5, x + 36, y + 27), fill="#161c24")
                draw.text((x + 12, y + 8), str(index + 1), fill="#ffd465")
                draw.text(
                    (x + 5, y + height - 24), f"{start:.1f}-{end:.1f}s | @{at:.1f}s", fill="#d5dce8"
                )
                level = readability(image.width, image.height)
                observations.append(
                    {
                        "cell": index + 1,
                        "image": page,
                        "from": round(start, 2),
                        "to": round(end, 2),
                        "frame_at": round(at, 2),
                        "readability": level,
                        "pixels": [image.width, image.height],
                        "dark": dark,
                    }
                )
                index += 1
            output = Path(directory) / f"storyboard-{page}.jpg"
            sheet.save(output, "JPEG", quality=88)
            images.append(str(output))
            sheets.append(
                {"image": page, "width": sheet.width, "height": sheet.height, "cells": number}
            )
        if not observations:
            raise VideoError("无法取得可读取的代表帧")
        levels = {"composition": 0, "shot": 1, "scene": 2, "text": 3}
        if any(levels[cell["readability"]] < levels[plan["need"]] for cell in observations):
            plan["warnings"].append(
                "部分实际画面低于目标可读性，源视频清晰度也可能不足；不能据此辨认小字"
            )
        if any(cell["dark"] for cell in observations):
            plan["warnings"].append("有近黑代表帧；这不表示相应时段没有剧情或台词")
        return {
            "image": images[0],
            "images": images,
            "cells": observations,
            "cuts": cuts,
            "duration": duration,
            "scan_interval": round(step, 3),
            "sheets": sheets,
            "plan": {k: v for k, v in plan.items() if k not in {"layouts", "counts"}},
        }
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
