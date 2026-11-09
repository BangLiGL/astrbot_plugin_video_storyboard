"""Bilibili identifiers, subtitles and bounded media downloads."""

import asyncio
import errno
import ipaddress
import json
import math
import re
import shutil
import socket
from pathlib import Path
from urllib.parse import parse_qs, urljoin, urlsplit

import aiohttp
from aiohttp.abc import AbstractResolver
from bilibili_api import Credential, exceptions, video

BV = re.compile(r"(?<![A-Za-z0-9])BV[0-9A-Za-z]{10}(?![A-Za-z0-9])", re.I)
URL = re.compile(r"https?://[^\s<>\"\\]+", re.I)
HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.bilibili.com/"}


class VideoError(Exception):
    """An operational error safe to describe to the model."""


class DiskSpaceError(VideoError):
    """A local capacity failure; trying another CDN cannot fix it."""


def subtitle_failure(error):
    """Only safe categories/codes; exception strings may contain cookies or signed URLs."""
    if isinstance(error, exceptions.CredentialNoSessdataException) or (
        isinstance(error, exceptions.ResponseCodeException) and error.code == -101
    ):
        return {"status": "login_required", "reason": "authentication"}
    if isinstance(error, exceptions.ResponseCodeException):
        return {"status": "failed", "reason": "api_error", "code": int(error.code)}
    if isinstance(error, aiohttp.ClientResponseError):
        return {"status": "failed", "reason": "http_error", "code": int(error.status)}
    if isinstance(error, (aiohttp.ClientError, exceptions.NetworkException, OSError, TimeoutError)):
        return {"status": "failed", "reason": "network_error"}
    if isinstance(error, (ValueError, TypeError, KeyError, UnicodeError)):
        return {"status": "failed", "reason": "invalid_response"}
    return {"status": "failed", "reason": "request_failed"}


def subtitle_warning(state):
    if state["status"] == "login_required":
        return "站点字幕需要有效登录，当前未取得；检查B站登录凭据后可重试，不等于视频没有字幕"
    detail = {
        "api_error": "站点接口返回错误",
        "http_error": "字幕下载返回HTTP错误",
        "network_error": "网络连接或请求超时",
        "invalid_response": "字幕响应格式无效",
        "request_failed": "字幕请求未完成",
    }.get(state.get("reason"), "字幕请求未完成")
    code = f"（代码{state['code']}）" if "code" in state else ""
    return f"站点字幕获取失败：{detail}{code}；继续尝试内嵌字幕或ASR，不能当作视频没有字幕"


def ensure_space(destination, required):
    """Check the destination filesystem, including directories not created yet."""
    probe = Path(destination).resolve()
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    try:
        free = shutil.disk_usage(probe).free
    except OSError:
        return  # The actual write still reports permission/filesystem failures.
    if free < max(0, required) + 16 * 1048576:
        raise DiskSpaceError("视频临时目录所在磁盘空间不足，请先释放空间后重试")


def extract_refs(text):
    """Find BV IDs and Bilibili URLs while retaining explicit part numbers."""
    refs = []
    seen = set()
    covered = []
    for match in URL.finditer(str(text)[:24000]):
        url = match.group().rstrip("。，、；！？）,.)]")
        parsed = urlsplit(url)
        host = (parsed.hostname or "").lower()
        if host == "b23.tv":
            key = url.split("?")[0]
            if key not in seen:
                refs.append({"short": key})
                seen.add(key)
            covered.append(match.span())
        elif host == "bilibili.com" or host.endswith(".bilibili.com"):
            bvid = BV.search(parsed.path)
            if bvid:
                try:
                    page = max(1, int(parse_qs(parsed.query).get("p", [1])[0]))
                except (TypeError, ValueError):
                    page = 1
                value = "BV" + bvid.group()[2:]
                key = f"{value}:{page}"
                if key not in seen:
                    refs.append({"bvid": value, "page": page})
                    seen.add(key)
                covered.append(match.span())
    for match in BV.finditer(str(text)[:24000]):
        if any(a <= match.start() < b for a, b in covered):
            continue
        value = "BV" + match.group()[2:]
        key = f"{value}:1"
        if key not in seen:
            refs.append({"bvid": value, "page": 1})
            seen.add(key)
    return refs[:2]


def check_url(url):
    parsed = urlsplit(url)
    if parsed.scheme not in {"https", "http"} or not parsed.hostname or parsed.username:
        raise VideoError("视频服务返回了无效链接")
    try:
        address = ipaddress.ip_address(parsed.hostname)
    except ValueError:
        if parsed.hostname.lower() == "localhost":
            raise VideoError("不能请求内网媒体地址")
    else:
        if not address.is_global:
            raise VideoError("不能请求内网媒体地址")
    return parsed


class PublicResolver(AbstractResolver):
    def __init__(self):
        self.inner = aiohttp.resolver.ThreadedResolver()

    async def resolve(self, host, port=0, family=socket.AF_INET):
        answers = await self.inner.resolve(host, port, family)
        trusted_cdn = any(
            host == domain or host.endswith("." + domain)
            for domain in ("bilivideo.com", "bilivideo.cn", "hdslb.com")
        )
        for item in answers:
            address = ipaddress.ip_address(item["host"])
            # TUN proxies commonly map genuine CDN names into this synthetic IP pool.
            proxy_fake_ip = address.version == 4 and address in ipaddress.ip_network(
                "198.18.0.0/15"
            )
            if not address.is_global and not (trusted_cdn and proxy_fake_ip):
                raise OSError("Refusing a private media address")
        return answers

    async def close(self):
        await self.inner.close()


class BiliClient:
    def __init__(self, config):
        self.config = config

    async def _stream(self, stream, max_bytes, target):
        candidates = [stream.get("baseUrl") or stream.get("base_url") or stream.get("url")]
        candidates.extend(stream.get("backupUrl") or stream.get("backup_url") or [])
        candidates = list(dict.fromkeys(url for url in candidates if url))
        # Prefer ordinary HTTPS CDN nodes over region-dependent P2P/mcdn endpoints.
        candidates.sort(
            key=lambda url: (
                "mcdn" in (urlsplit(url).hostname or ""),
                urlsplit(url).port not in {None, 443},
            )
        )
        for url in candidates[:4]:
            try:
                return await asyncio.wait_for(self._get(url, max_bytes, target), timeout=40)
            except asyncio.CancelledError:
                raise
            except DiskSpaceError:
                raise
            except Exception:
                continue
        raise VideoError("B站主备媒体节点均暂时不可用")

    async def _get(self, url, max_bytes, target=None):
        resolver = PublicResolver()
        connector = aiohttp.TCPConnector(resolver=resolver)
        timeout = aiohttp.ClientTimeout(total=120, connect=15)
        part = target.with_suffix(".part") if target else None
        try:
            async with aiohttp.ClientSession(
                headers=HEADERS, timeout=timeout, connector=connector
            ) as client:
                for _ in range(6):
                    check_url(url)
                    async with client.get(url, allow_redirects=False) as response:
                        if response.status in {301, 302, 303, 307, 308}:
                            location = response.headers.get("Location")
                            if not location:
                                raise VideoError("视频服务重定向失败")
                            url = urljoin(url, location)
                            continue
                        response.raise_for_status()
                        if response.content_length and response.content_length > max_bytes:
                            raise VideoError("视频文件超过插件的大小上限")
                        size = 0
                        if target:
                            ensure_space(
                                target.parent,
                                response.content_length or min(max_bytes, 8 * 1048576),
                            )
                            with part.open("wb") as file:
                                checked = 0
                                async for chunk in response.content.iter_chunked(65536):
                                    size += len(chunk)
                                    if size > max_bytes:
                                        raise VideoError("视频文件超过插件的大小上限")
                                    if size - checked >= 1048576:
                                        ensure_space(target.parent, len(chunk))
                                        checked = size
                                    file.write(chunk)
                            if not size:
                                raise VideoError("视频下载结果为空")
                            part.replace(target)
                            return str(target)
                        raw = bytearray()
                        async for chunk in response.content.iter_chunked(65536):
                            raw.extend(chunk)
                            if len(raw) > max_bytes:
                                raise VideoError("字幕数据过大")
                        return bytes(raw)
                raise VideoError("视频服务重定向次数过多")
        except OSError as error:
            if error.errno == errno.ENOSPC:
                raise DiskSpaceError("视频临时目录所在磁盘空间不足，请先释放空间后重试") from error
            raise
        finally:
            if part:
                part.unlink(missing_ok=True)
            await resolver.close()

    async def resolve(self, ref):
        if "bvid" in ref:
            return ref
        url = ref["short"]
        timeout = aiohttp.ClientTimeout(total=15)
        async with aiohttp.ClientSession(timeout=timeout, headers=HEADERS) as client:
            for _ in range(5):
                host = (urlsplit(url).hostname or "").lower()
                if (
                    host != "b23.tv"
                    and host != "bilibili.com"
                    and not host.endswith(".bilibili.com")
                ):
                    raise VideoError("B站短链跳转到了站外地址")
                found = extract_refs(url)
                if found and "bvid" in found[0]:
                    return found[0]
                async with client.get(url, allow_redirects=False) as response:
                    target = response.headers.get("Location")
                    if not target:
                        break
                    url = urljoin(url, target)
        raise VideoError("无法解析这个 B站短链")

    async def fetch(self, ref, directory, on_info=None):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        credential = Credential(sessdata=self.config.get("sessdata") or None)
        item = video.Video(bvid=ref["bvid"], credential=credential)
        info = await item.get_info()
        pages = info.get("pages") or []
        page = ref.get("page", 1)
        if not pages or not 1 <= page <= len(pages):
            raise VideoError("视频分P不存在")
        part = pages[page - 1]
        duration = float(part.get("duration") or info.get("duration") or 0)
        if not 0 < duration <= int(self.config.get("max_duration_seconds", 600)):
            raise VideoError("这段视频时长未知或超过允许分析的时长")
        result = {
            "bvid": ref["bvid"],
            "page": page,
            "title": str(info.get("title", ""))[:300],
            "part_title": str(part.get("part", ""))[:200],
            "duration": duration,
            "uploader": str((info.get("owner") or {}).get("name", ""))[:100],
            "description": str(info.get("desc", ""))[:1500],
            "url": f"https://www.bilibili.com/video/{ref['bvid']}?p={page}",
            "segments": [],
            "transcript_source": "none",
            "warnings": [],
        }
        if on_info:
            await on_info(result)
        await self._subtitles(item, part["cid"], result)
        try:
            streams = await item.get_download_url(cid=part["cid"])
            dash = streams.get("dash") or {}
            videos = dash.get("video") or []
            audio = dash.get("audio") or []
            if audio and result["segments"]:
                result["audio_state"] = "available_not_downloaded"
            max_bytes = int(self.config.get("max_file_mb", 150)) * 1048576
            if videos:
                preferred = [
                    v for v in videos if str(v.get("codecs", "")).startswith("avc")
                ] or videos
                target_height = {"360p": 360, "720p": 720, "1080p": 1080}.get(
                    self.config.get("video_quality", "720p"), 720
                )
                bounded = [v for v in preferred if int(v.get("height", 9999)) <= target_height]
                chosen = (
                    max(bounded, key=lambda v: v.get("height", 0))
                    if bounded
                    else min(preferred, key=lambda v: v.get("height", 9999))
                )
                result["download_height"] = int(chosen.get("height") or 0)
                if result["download_height"] and result["download_height"] < target_height:
                    result["warnings"].append(
                        f"本次可取视频为{result['download_height']}p，低于所选{target_height}p；不能保证读清小字"
                    )
                result["video_path"] = await self._stream(
                    chosen, max_bytes, directory / "video.mp4"
                )
                if audio and not result["segments"]:
                    result["audio_state"] = "unavailable"
                    chosen_audio = min(audio, key=lambda a: a.get("bandwidth", 9999999))
                    result["audio_path"] = await self._stream(
                        chosen_audio,
                        min(max_bytes, 50 * 1048576),
                        directory / "audio.m4a",
                    )
                    result.pop(
                        "audio_state", None
                    )  # Probe the file before claiming it is readable.
            elif len(streams.get("durl") or []) == 1:
                result["video_path"] = await self._stream(
                    streams["durl"][0], max_bytes, directory / "video.mp4"
                )
                result["audio_path"] = result["video_path"]
            else:
                raise VideoError("视频服务没有返回可用的媒体流")
        except Exception:
            result["warnings"].append(
                "音轨下载失败，不能确认未取得的语音内容"
                if result.get("video_path")
                else "视频画面下载失败，当前只能依据已取得的字幕理解，不能声称看到了画面"
            )
        if not result.get("video_path") and not result["segments"]:
            if result["site_subtitle"]["status"] in {"failed", "login_required"}:
                raise VideoError(
                    subtitle_warning(result["site_subtitle"])
                    + "；本次视频画面也未取得，暂时无法理解内容"
                )
            raise VideoError("没能取得视频画面或字幕，可能需要登录、受到站点风控或视频已失效")
        return result

    async def _subtitles(self, item, cid, result):
        try:
            metadata = await item.get_subtitle(cid=cid)
            if metadata is None:
                metadata = {}
            if not isinstance(metadata, dict):
                raise ValueError("Invalid subtitle metadata")
            subtitles = metadata.get("subtitles", [])
            if not isinstance(subtitles, list) or any(not isinstance(s, dict) for s in subtitles):
                raise ValueError("Invalid subtitle list")
            if not subtitles:
                if metadata.get("need_login_subtitle"):
                    state = {"status": "login_required", "reason": "authentication"}
                    result["site_subtitle"] = state
                    result["warnings"].append(subtitle_warning(state))
                else:
                    result["site_subtitle"] = {"status": "none"}
                return
            subtitles.sort(
                key=lambda sub: (
                    not str(sub.get("lan", "")).startswith("zh"),
                    str(sub.get("lan", "")),
                )
            )
            selected = subtitles[0]
            url = str(selected.get("subtitle_url", ""))
            if url.startswith("//"):
                url = "https:" + url
            if not url:
                raise ValueError("Missing subtitle URL")
            data = json.loads(await self._get(url, 4 * 1048576))
            if not isinstance(data, dict) or not isinstance(data.get("body"), list):
                raise ValueError("Invalid subtitle body")
            rows = []
            for row in data["body"][:12000]:
                if not isinstance(row, dict) or not isinstance(row.get("content"), str):
                    raise ValueError("Invalid subtitle row")
                if not row["content"].strip():
                    continue
                start, end = float(row["from"]), float(row["to"])
                if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
                    raise ValueError("Invalid subtitle timing")
                rows.append(
                    {"from": start, "to": end, "text": row["content"][:2000], "timing": "subtitle"}
                )
            language = str(selected.get("lan", ""))
            result["site_subtitle"] = {"status": "available" if rows else "empty"}
            if re.fullmatch(r"[a-zA-Z0-9_-]{1,32}", language):
                result["site_subtitle"]["language"] = language
            result["segments"] = rows
            if rows:
                result["transcript_source"] = "bilibili_subtitle"
        except asyncio.CancelledError:
            raise
        except Exception as error:
            state = subtitle_failure(error)
            result["site_subtitle"] = state
            result["warnings"].append(subtitle_warning(state))
