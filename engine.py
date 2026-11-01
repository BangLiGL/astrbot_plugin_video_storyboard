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
from .decode import extract_frames, scan_video
from .sampling import density_boundaries, plan_images, readability


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
        capture.release()
        scan = scan_video(video_path, fps, duration, cancel)
        cuts = scan["cuts"]
        boundaries = density_boundaries(scan["scores"], duration, plan["actual_cells"])
        # Avoid selecting on a detected transition: use the largest shot overlap per interval.
        requests = []
        for start, end in zip(boundaries, boundaries[1:]):
            marks = [start, *(at for at in cuts if start < at < end), end]
            a, b = max(zip(marks, marks[1:]), key=lambda pair: pair[1] - pair[0])
            requests.append(
                [min(max(0, duration - 1 / fps), a + (b - a) * r) for r in (0.25, 0.5, 0.75)]
            )
        observations = []
        images, sheets = [], []
        index = 0
        for page, (number, canvas) in enumerate(zip(plan["counts"], plan["layouts"]), 1):
            page_start, page_end = boundaries[index], boundaries[index + number]
            targets = [at for group in requests[index : index + number] for at in group]
            try:
                frames = extract_frames(
                    video_path,
                    targets,
                    (min(source_w, canvas["cell_width"]), min(source_h, canvas["cell_height"])),
                    source_w / source_h,
                    cancel,
                )
            except VideoError:
                if cancel.is_set():
                    raise
                plan["warnings"].append(f"第{page}张图取帧失败，未取得的画面不能当作已看过")
                frames = {}
            sheet = Image.new("RGB", (canvas["width"], canvas["height"]), "#121720")
            draw = ImageDraw.Draw(sheet)
            draw.text(
                (6, 5),
                f"{page_start:.1f}-{page_end:.1f}s  |  {page}/{len(plan['counts'])}",
                fill="#ffd465",
            )
            width, height = canvas["cell_width"] + 6, canvas["cell_height"] + 34
            for local in range(number):
                if cancel.is_set():
                    raise VideoError("视频处理已取消")
                start, end = boundaries[index : index + 2]
                candidates = []
                for target in requests[index]:
                    if target in frames:
                        at, frame = frames[target]
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
                x, y = local % canvas["columns"] * width, 24 + local // canvas["columns"] * height
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
                        "timing": "decoded_pts",
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
                {
                    "image": page,
                    "width": sheet.width,
                    "height": sheet.height,
                    "cells": number,
                    "from": round(page_start, 2),
                    "to": round(page_end, 2),
                }
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
        if scan["rate"] < min(fps, 24):
            plan["warnings"].append("长视频扫描已按资源上限降采样，极短镜头可能遗漏")
        return {
            "image": images[0],
            "images": images,
            "cells": observations,
            "cuts": cuts,
            "duration": duration,
            "scan_interval": round(scan["interval"], 3),
            "scan_frames": scan["frames"],
            "letterbox_fraction": scan["bars"],
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
