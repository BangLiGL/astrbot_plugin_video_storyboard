"""Read local audio/subtitle tracks without extra services or FFprobe dependencies."""

import asyncio
import contextlib
import re
import subprocess

import imageio_ffmpeg

from .bilibili import VideoError


async def run_ffmpeg(arguments, limit=4 * 1048576, timeout=30):
    kwargs = (
        {"creationflags": subprocess.CREATE_NO_WINDOW}
        if hasattr(subprocess, "CREATE_NO_WINDOW")
        else {}
    )
    child = await asyncio.create_subprocess_exec(
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-hide_banner",
        "-nostdin",
        "-v",
        "error",
        *arguments,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        **kwargs,
    )

    async def read(stream, cap):
        chunks, total = [], 0
        while block := await stream.read(65536):
            total += len(block)
            if total > cap:
                raise VideoError("媒体轨道信息超过处理上限")
            chunks.append(block)
        return b"".join(chunks)

    readers = [
        asyncio.create_task(read(child.stdout, limit)),
        asyncio.create_task(read(child.stderr, 65536)),
    ]
    try:
        stdout, stderr = await asyncio.wait_for(asyncio.gather(*readers), timeout=timeout)
        return await child.wait(), stdout, stderr
    finally:
        if child.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                child.kill()
        for task in readers:
            if not task.done():
                task.cancel()
        await asyncio.gather(*readers, return_exceptions=True)
        await child.wait()


def missing_stream(stderr):
    return b"matches no streams" in stderr.lower()


async def audio_status(path):
    code, _, error = await run_ffmpeg(
        [
            "-threads",
            "1",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-t",
            "0.01",
            "-f",
            "null",
            "-",
        ]
    )
    return "present" if code == 0 else "absent" if missing_stream(error) else "unavailable"


def parse_srt(text, duration):
    def seconds(match):
        h, m, s, ms = map(int, match)
        return h * 3600 + m * 60 + s + ms / 1000

    rows = []
    pattern = r"(\d{1,3}):(\d{2}):(\d{2})[,.](\d{3})"
    for block in re.split(r"\r?\n\s*\r?\n", text.lstrip("\ufeff"))[:12000]:
        lines = block.splitlines()
        marker = next((i for i, line in enumerate(lines) if "-->" in line), None)
        if marker is None:
            continue
        times = re.findall(pattern, lines[marker])
        if len(times) != 2:
            continue
        start, end = map(seconds, times)
        words = re.sub(r"<[^>]{0,200}>", "", " ".join(lines[marker + 1 :])).strip()[:2000]
        if 0 <= start < end and start < duration and words:
            rows.append(
                {
                    "from": start,
                    "to": min(end, duration),
                    "text": words,
                    "timing": "embedded_subtitle",
                }
            )
    return sorted(rows, key=lambda row: row["from"])


async def embedded_subtitles(path, duration):
    code, output, error = await run_ffmpeg(
        [
            "-threads",
            "1",
            "-i",
            str(path),
            "-t",
            str(duration),
            "-map",
            "0:s:0",
            "-f",
            "srt",
            "-",
        ]
    )
    if code:
        return [], "absent" if missing_stream(error) else "unreadable"
    rows = parse_srt(output.decode("utf-8", errors="replace"), duration)
    return rows, "available" if rows else "empty"
