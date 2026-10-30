"""Bilibili identifiers, subtitles and bounded media downloads."""

import asyncio
import ipaddress
import json
import re
import socket
from pathlib import Path
from urllib.parse import parse_qs, urljoin, urlsplit

import aiohttp
from aiohttp.abc import AbstractResolver
from bilibili_api import Credential, video

BV = re.compile(r"(?<![A-Za-z0-9])BV[0-9A-Za-z]{10}(?![A-Za-z0-9])", re.I)
URL = re.compile(r"https?://[^\s<>\"\\]+", re.I)
HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://www.bilibili.com/"}


class VideoError(Exception):
    """An operational error safe to describe to the model."""


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
                            with part.open("wb") as file:
                                async for chunk in response.content.iter_chunked(65536):
                                    size += len(chunk)
                                    if size > max_bytes:
                                        raise VideoError("视频文件超过插件的大小上限")
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
        try:
            subtitles = (await item.get_subtitle(cid=part["cid"]) or {}).get("subtitles") or []
            subtitles.sort(
                key=lambda sub: (not str(sub.get("lan", "")).startswith("zh"), sub.get("lan", ""))
            )
            if subtitles:
                url = str(subtitles[0].get("subtitle_url", ""))
                if url.startswith("//"):
                    url = "https:" + url
                data = json.loads(await self._get(url, 4 * 1048576))
                result["segments"] = [
                    {
                        "from": float(row["from"]),
                        "to": float(row["to"]),
                        "text": str(row["content"])[:2000],
                        "timing": "subtitle",
                    }
                    for row in data.get("body", [])
                    if row.get("content")
                ][:12000]
                result["transcript_source"] = "bilibili_subtitle"
        except Exception:
            result["warnings"].append("站点字幕不可用；有配置语音识别时尝试转写")
        try:
            streams = await item.get_download_url(cid=part["cid"])
            dash = streams.get("dash") or {}
            videos = dash.get("video") or []
            audio = dash.get("audio") or []
            max_bytes = int(self.config.get("max_file_mb", 150)) * 1048576
            if videos:
                preferred = [
                    v for v in videos if str(v.get("codecs", "")).startswith("avc")
                ] or videos
                bounded = [v for v in preferred if int(v.get("height", 9999)) <= 480]
                chosen = (
                    max(bounded, key=lambda v: v.get("height", 0))
                    if bounded
                    else min(preferred, key=lambda v: v.get("height", 9999))
                )
                result["video_path"] = await self._stream(
                    chosen, max_bytes, directory / "video.mp4"
                )
                if audio and not result["segments"]:
                    chosen_audio = min(audio, key=lambda a: a.get("bandwidth", 9999999))
                    result["audio_path"] = await self._stream(
                        chosen_audio,
                        min(max_bytes, 50 * 1048576),
                        directory / "audio.m4a",
                    )
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
            raise VideoError("没能取得视频画面或字幕，可能需要登录、受到站点风控或视频已失效")
        return result
