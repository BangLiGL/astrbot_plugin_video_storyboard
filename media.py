"""Borrow quoted local videos or fetch a bounded public attachment."""

import asyncio
import hashlib
import math
from pathlib import Path
from urllib.parse import unquote, urlsplit
from urllib.request import url2pathname

import cv2

from .bilibili import VideoError, check_url


def resolve_sources(sources, data_root, max_bytes):
    root = Path(data_root).resolve()
    urls = []
    for source in sources[:6]:
        source = str(source)[:8192]
        parsed = urlsplit(source)
        if parsed.scheme in {"http", "https"}:
            try:
                check_url(source)
                urls.append(source)
            except VideoError:
                pass
            continue
        if parsed.scheme == "file":
            if parsed.netloc:
                continue
            source = url2pathname(unquote(parsed.path))
        elif parsed.scheme and len(parsed.scheme) != 1:  # Windows drive letter is valid.
            continue
        path = Path(source).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            continue
        stat = path.stat()
        if stat.st_size > max_bytes:
            raise VideoError("引用的视频超过插件的文件大小上限")
        fingerprint = hashlib.sha256(
            f"{path}\0{stat.st_size}\0{stat.st_mtime_ns}".encode()
        ).hexdigest()
        return {"local_video": str(path), "fingerprint": fingerprint}
    if urls:
        return {"video_url": urls[0]}
    raise VideoError("引用视频未提供可读取的本地缓存或公网媒体地址，请同时附上原视频链接或BV号")


def duration_of(path):
    capture = cv2.VideoCapture(str(path))
    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        frames = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frames / fps if fps > 0 else 0
        if not math.isfinite(duration) or duration <= 0:
            raise VideoError("引用视频无法解码，未获得有效时长")
        return duration
    finally:
        capture.release()


async def prepare_video(ref, directory, config, client, on_info):
    path = ref.get("local_video")
    if not path:
        path = await client._get(
            ref["video_url"],
            max(1, int(config.get("max_file_mb", 150))) * 1048576,
            Path(directory) / "quoted.mp4",
        )
    duration = await asyncio.to_thread(duration_of, path)
    if duration > min(3600, int(config.get("max_duration_seconds", 600))):
        raise VideoError("引用的视频超过允许分析的时长")
    media = {
        "title": "用户引用的视频",
        "part_title": "",
        "uploader": "",
        "duration": duration,
        "url": "",
        "segments": [],
        "transcript_source": "none",
        "warnings": [],
        "video_path": str(path),
        "audio_path": str(path),
    }
    if on_info:
        await on_info(media)
    return media
