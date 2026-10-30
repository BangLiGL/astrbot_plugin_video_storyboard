"""Real video/audio processing plus AstrBot chat lifecycle contract tests."""

import asyncio
import copy
import importlib
import json
import logging
import subprocess
import sys
import tempfile
import types
import unittest
import wave
from pathlib import Path
from unittest.mock import AsyncMock, patch

import imageio_ffmpeg
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = "bili_chat_test_package"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(ROOT)]
sys.modules[PACKAGE] = package
bili = importlib.import_module(PACKAGE + ".bilibili")
engine = importlib.import_module(PACKAGE + ".engine")
analysis = importlib.import_module(PACKAGE + ".analysis")
knowledge = importlib.import_module(PACKAGE + ".knowledge")


def load_main():
    def identity(*args, **kwargs):
        return lambda function: function

    class Star:
        def __init__(self, context):
            self.context = context

    modules = {
        name: types.ModuleType(name)
        for name in (
            "astrbot",
            "astrbot.api",
            "astrbot.api.event",
            "astrbot.api.provider",
            "astrbot.api.star",
        )
    }
    modules["astrbot.api"].AstrBotConfig = dict
    modules["astrbot.api"].logger = logging.getLogger("test")
    modules["astrbot.api.event"].AstrMessageEvent = object
    modules["astrbot.api.event"].filter = types.SimpleNamespace(
        event_message_type=identity,
        on_llm_request=identity,
        EventMessageType=types.SimpleNamespace(ALL=1),
    )
    modules["astrbot.api.provider"].ProviderRequest = object
    modules["astrbot.api.star"].Star = Star
    modules["astrbot.api.star"].Context = object
    modules["astrbot.api.star"].StarTools = types.SimpleNamespace(get_data_dir=lambda _: ROOT)
    with patch.dict(sys.modules, modules):
        return importlib.import_module(PACKAGE + ".main")


main = load_main()


class Event:
    def __init__(self, text, group="100", bot="bot1", messages=None):
        self.text = text
        self.group = group
        self.messages = messages or []
        self.unified_msg_origin = f"test:group:{group}"
        self.message_obj = types.SimpleNamespace(self_id=bot)
        self.extras = {}
        self.requests = []
        self.call_llm = False

    def get_group_id(self):
        return self.group

    def get_platform_name(self):
        return "test"

    def get_message_str(self):
        return self.text

    def get_messages(self):
        return self.messages

    def get_extra(self, key):
        return self.extras.get(key)

    def set_extra(self, key, value):
        self.extras[key] = value

    def should_call_llm(self, value):
        self.call_llm = value

    def request_llm(self, **kwargs):
        self.requests.append(kwargs)
        return types.SimpleNamespace(**kwargs)

    def plain_result(self, *args):
        raise AssertionError("The plugin must not send a progress/report message")

    def image_result(self, *args):
        raise AssertionError("The plugin must not send a storyboard into the group")


class Context:
    def __init__(self):
        self.conversation = types.SimpleNamespace(
            cid="current",
            persona_id="group_persona",
            history='[{"role":"user","content":"群里之前的话题"}]',
        )
        self.conversation_manager = types.SimpleNamespace(
            get_curr_conversation_id=AsyncMock(return_value="current"),
            get_conversation=AsyncMock(return_value=self.conversation),
            new_conversation=AsyncMock(return_value="new"),
        )
        self.stt = types.SimpleNamespace(
            provider_config={"id": "stt"}, get_text=AsyncMock(return_value="这是一段语音里的观点")
        )
        self.providers = {
            "stt": self.stt,
            "vision": types.SimpleNamespace(provider_config={"modalities": ["text", "image"]}),
            "chat": types.SimpleNamespace(provider_config={"modalities": ["text"]}),
        }
        self.get_using_stt_provider_async = AsyncMock(return_value=self.stt)
        self.get_current_chat_provider_id = AsyncMock(return_value="chat")
        self.llm_generate = AsyncMock(
            return_value=types.SimpleNamespace(
                completion_text="画面先出现红色场景，随后切换到蓝色场景。说话者提出一个观点，情绪有反差。"
            )
        )

    def get_provider_by_id(self, name):
        return self.providers.get(name)

    def get_all_stt_providers(self):
        return [provider for provider in self.providers.values() if hasattr(provider, "get_text")]

    def get_config(self, origin):
        return {"provider_settings": {}}


class ParsingTests(unittest.TestCase):
    def test_bv_links_short_links_cards_and_parts(self):
        self.assertEqual(
            bili.extract_refs("看这个 BV1GJ411x7h7 哈哈"), [{"bvid": "BV1GJ411x7h7", "page": 1}]
        )
        refs = bili.extract_refs("https://www.bilibili.com/video/BV1GJ411x7h7?p=2&spm=123")
        self.assertEqual(refs, [{"bvid": "BV1GJ411x7h7", "page": 2}])
        self.assertEqual(
            bili.extract_refs("https://b23.tv/abcd"), [{"short": "https://b23.tv/abcd"}]
        )
        self.assertFalse(bili.extract_refs("BV1abc XBV1GJ411x7h7Z"))
        card = types.SimpleNamespace(
            type="ComponentType.Json",
            data={"meta": {"news": {"jumpUrl": "https://www.bilibili.com/video/BV1GJ411x7h7"}}},
        )
        instance = object.__new__(main.VideoStoryboardPlugin)
        self.assertTrue(bili.extract_refs(instance.message_text(Event("", messages=[card]))))

    def test_no_dependency_or_asr_endpoint_configuration(self):
        schema = json.loads((ROOT / "_conf_schema.json").read_text(encoding="utf-8"))
        self.assertEqual(schema["stt_provider_id"]["_special"], "select_provider_stt")
        self.assertEqual(schema["vision_provider_id"]["_special"], "select_provider")
        for key in schema:
            self.assertNotIn("path", key)
            self.assertNotIn("base_url", key)
            self.assertNotIn("api_key", key)
        requirements = (ROOT / "requirements.txt").read_text()
        self.assertIn("imageio-ffmpeg", requirements)
        self.assertNotIn("yt-dlp", requirements)

    def test_private_media_targets_rejected(self):
        for url in (
            "http://127.0.0.1/a",
            "http://[::1]/a",
            "file:///etc/passwd",
            "http://169.254.169.254/x",
        ):
            with self.subTest(url=url), self.assertRaises(bili.VideoError):
                bili.check_url(url)


class KnowledgeTests(unittest.TestCase):
    def test_durable_group_isolation_and_retrieval(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "kb.sqlite3"
            kb = knowledge.Knowledge(path)
            record = {
                "bvid": "BV1GJ411x7h7",
                "title": "红蓝场景",
                "understanding": "记忆管理的观点与结论",
                "visual": True,
                "secret": "must-not-store",
                "video_path": "private",
            }
            clean = kb.save("key", record)
            self.assertNotIn("secret", clean)
            self.assertNotIn("video_path", clean)
            kb.remember("platform:bot:group1", "key", clean)
            reopened = knowledge.Knowledge(path)
            self.assertEqual(reopened.cached("key")["title"], "红蓝场景")
            self.assertTrue(reopened.retrieve("platform:bot:group1", "刚才那个视频"))
            self.assertTrue(reopened.retrieve("platform:bot:group1", "记忆管理为什么这样？"))
            self.assertFalse(reopened.retrieve("platform:bot:group2", "记忆管理为什么这样？"))
            self.assertFalse(reopened.retrieve("platform:bot2:group1", "刚才那个视频"))
            self.assertFalse(reopened.retrieve("platform:bot:group1", "今天晚饭吃什么？"))

    def test_memory_bound(self):
        with tempfile.TemporaryDirectory() as folder:
            kb = knowledge.Knowledge(Path(folder) / "kb.sqlite3")
            for index in range(105):
                kb.remember("group", str(index), {"title": str(index)})
            with kb.connect() as db:
                count = db.execute("SELECT count(*) FROM memories").fetchone()[0]
            self.assertEqual(count, 100)


class BiliClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_selected_part_subtitles_and_stream_selection(self):
        sdk = types.SimpleNamespace(
            get_info=AsyncMock(
                return_value={
                    "title": "测试",
                    "pages": [{"cid": 1, "duration": 5}, {"cid": 2, "duration": 8}],
                }
            ),
            get_subtitle=AsyncMock(
                return_value={
                    "subtitles": [
                        {"lan": "en", "subtitle_url": "https://example.com/en"},
                        {"lan": "zh-CN", "subtitle_url": "//example.com/zh"},
                    ]
                }
            ),
            get_download_url=AsyncMock(
                return_value={
                    "dash": {
                        "video": [
                            {
                                "height": 1080,
                                "codecs": "avc",
                                "baseUrl": "https://example.com/high",
                            },
                            {"height": 480, "codecs": "avc", "baseUrl": "https://example.com/low"},
                        ],
                        "audio": [{"baseUrl": "https://example.com/audio"}],
                    }
                }
            ),
        )
        client = bili.BiliClient({})
        client._get = AsyncMock(
            return_value=json.dumps({"body": [{"from": 0, "to": 2, "content": "字幕"}]}).encode()
        )
        client._stream = AsyncMock(return_value="video.mp4")
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.object(bili.video, "Video", return_value=sdk),
        ):
            result = await client.fetch({"bvid": "BV1GJ411x7h7", "page": 2}, folder)
        sdk.get_subtitle.assert_awaited_once_with(cid=2)
        self.assertEqual(result["transcript_source"], "bilibili_subtitle")
        self.assertEqual(client._get.call_args.args[0], "https://example.com/zh")
        self.assertEqual(client._stream.await_count, 1)  # Subtitles mean no audio download.
        self.assertEqual(client._stream.call_args.args[0]["height"], 480)

    async def test_duration_bound_before_download(self):
        sdk = types.SimpleNamespace(
            get_info=AsyncMock(return_value={"pages": [{"cid": 1, "duration": 9999}]})
        )
        client = bili.BiliClient({})
        client._get = AsyncMock()
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.object(bili.video, "Video", return_value=sdk),
            self.assertRaises(bili.VideoError),
        ):
            await client.fetch({"bvid": "BV1GJ411x7h7", "page": 1}, folder)
        client._get.assert_not_called()

    async def test_stream_fallback_prefers_normal_cdn(self):
        client = bili.BiliClient({})
        client._get = AsyncMock(side_effect=[OSError("first node failed"), "video.mp4"])
        stream = {
            "baseUrl": "https://test.mcdn.bilivideo.cn:8082/video",
            "backupUrl": ["https://backup.bilivideo.com/video"],
        }
        result = await client._stream(stream, 1000, Path("video.mp4"))
        self.assertEqual(result, "video.mp4")
        self.assertEqual(
            client._get.call_args_list[0].args[0], "https://backup.bilivideo.com/video"
        )

    async def test_dns_guard_allows_known_cdn_tun_fake_ip_only(self):
        resolver = bili.PublicResolver()
        try:
            with patch.object(
                resolver.inner, "resolve", AsyncMock(return_value=[{"host": "198.18.0.10"}])
            ):
                self.assertTrue(await resolver.resolve("backup.bilivideo.com"))
                with self.assertRaises(OSError):
                    await resolver.resolve("untrusted.example")
            with patch.object(
                resolver.inner, "resolve", AsyncMock(return_value=[{"host": "10.0.0.10"}])
            ):
                with self.assertRaises(OSError):
                    await resolver.resolve("backup.bilivideo.com")
        finally:
            await resolver.close()


class ChatTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.context = Context()
        self.config = {"vision_provider_id": "vision", "stt_provider_id": "stt"}
        self.instance = object.__new__(main.VideoStoryboardPlugin)
        self.instance.context = self.context
        self.instance.config = self.config
        self.instance.analyzer = analysis.Analyzer(self.context, self.config, self.folder.name)

    async def asyncTearDown(self):
        await self.instance.terminate()
        self.folder.cleanup()

    async def test_bv_starts_one_normal_conversation_request_without_any_report(self):
        event = Event("看看 BV1GJ411x7h7，笑死了")
        results = [item async for item in self.instance.on_message(event)]
        self.assertEqual(len(results), 1)
        self.assertEqual(event.requests[0]["prompt"], event.text)
        self.assertIs(event.requests[0]["conversation"], self.context.conversation)
        self.assertEqual(event.requests[0]["conversation"].persona_id, "group_persona")
        self.assertTrue(event.call_llm)
        self.context.llm_generate.assert_not_called()

    async def test_knowledge_injected_without_replacing_persona_or_history(self):
        event = Event("BV1GJ411x7h7 你怎么看")
        record = {"title": "视频内容", "understanding": "作者在讲一个观点", "visual": True}
        self.instance.analyzer.understand = AsyncMock(return_value=record)
        request = types.SimpleNamespace(
            prompt=event.text,
            system_prompt="原来的人设",
            contexts=[{"role": "assistant", "content": "原来的聊天"}],
            image_urls=[],
        )
        history = copy.deepcopy(request.contexts)
        await self.instance.inject_video_knowledge(event, request)
        self.assertIn("作者在讲一个观点", request.prompt)
        self.assertEqual(request.system_prompt, "原来的人设")
        self.assertEqual(request.contexts, history)
        self.assertEqual(request.image_urls, [])
        once = request.prompt
        await self.instance.inject_video_knowledge(event, request)
        self.assertEqual(request.prompt, once)

    async def test_failure_is_context_for_honest_persona_reply_not_fake_summary(self):
        self.instance.analyzer.understand = AsyncMock(
            side_effect=bili.VideoError("没有读到视频内容，不能猜剧情")
        )
        event = Event("BV1GJ411x7h7")
        request = types.SimpleNamespace(prompt=event.text, system_prompt="人设")
        await self.instance.inject_video_knowledge(event, request)
        self.assertIn("没有读到视频内容", request.prompt)
        self.assertIn('"video_knowledge": []', request.prompt)
        self.assertEqual(request.system_prompt, "人设")

    async def test_ordinary_chat_and_group_allowlist(self):
        self.assertEqual([item async for item in self.instance.on_message(Event("今天聊点啥"))], [])
        self.config["enabled_groups"] = ["200"]
        self.assertEqual(
            [item async for item in self.instance.on_message(Event("BV1GJ411x7h7", group="100"))],
            [],
        )
        self.assertEqual(
            [item async for item in self.instance.on_message(Event("BV1GJ411x7h7", group=""))], []
        )
        a = self.instance.scope(Event("", group="200", bot="bot1"))
        b = self.instance.scope(Event("", group="200", bot="bot2"))
        self.assertNotEqual(a, b)

    async def test_followup_uses_group_memory_after_restart(self):
        event = Event("刚才那个视频为什么这样")
        self.instance.analyzer.knowledge.remember(
            self.instance.scope(event), "key", {"title": "视频", "understanding": "已读的事实"}
        )
        results = [item async for item in self.instance.on_message(event)]
        self.assertEqual(len(results), 1)
        request = types.SimpleNamespace(prompt=event.text)
        await self.instance.inject_video_knowledge(event, request)
        self.assertIn("已读的事实", request.prompt)
        other = Event(event.text, group="other")
        self.assertEqual([item async for item in self.instance.on_message(other)], [])


class MediaTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = tempfile.TemporaryDirectory()
        cls.clip = Path(cls.fixture.name) / "测试 video.mp4"
        subprocess.run(
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "color=c=red:s=320x180:d=4:r=12",
                "-f",
                "lavfi",
                "-i",
                "color=c=blue:s=320x180:d=4:r=12",
                "-f",
                "lavfi",
                "-i",
                "sine=frequency=440:duration=8",
                "-filter_complex",
                "[0:v][1:v]concat=n=2:v=1:a=0[v]",
                "-map",
                "[v]",
                "-map",
                "2:a",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-shortest",
                str(cls.clip),
            ],
            check=True,
            capture_output=True,
        )

    @classmethod
    def tearDownClass(cls):
        cls.fixture.cleanup()

    async def asyncSetUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.context = Context()
        self.config = {
            "vision_provider_id": "vision",
            "stt_provider_id": "stt",
            "storyboard_cells": 4,
        }
        self.analyzer = analysis.Analyzer(self.context, self.config, self.folder.name)
        self.metadata = {
            "bvid": "BV1GJ411x7h7",
            "page": 1,
            "title": "红蓝视频",
            "part_title": "第一P",
            "uploader": "作者",
            "duration": 8,
            "url": "https://www.bilibili.com/video/BV1GJ411x7h7",
            "segments": [],
            "warnings": [],
            "transcript_source": "none",
            "video_path": str(self.clip),
            "audio_path": str(self.clip),
        }
        self.analyzer.client.resolve = AsyncMock(side_effect=lambda ref: ref)

        async def fetch(ref, directory, on_info=None):
            if on_info:
                await on_info(self.metadata)
            await asyncio.sleep(0.02)
            return copy.deepcopy(self.metadata)

        self.analyzer.client.fetch = AsyncMock(side_effect=fetch)

    async def asyncTearDown(self):
        await self.analyzer.close()
        self.folder.cleanup()

    async def test_scene_detection_and_hidden_storyboard(self):
        result = await asyncio.to_thread(engine.storyboard, self.clip, self.folder.name, 4)
        self.assertTrue(any(abs(t - 4) < 0.8 for t in result["cuts"]))
        self.assertEqual(len(result["cells"]), 4)
        with Image.open(result["image"]) as image:
            self.assertGreater(image.width, 500)
            self.assertGreater(image.height, 400)
        self.assertGreater(result["cells"][-1]["frame_at"], 4)

    async def test_astrbot_stt_receives_real_wav_and_coarse_timestamps(self):
        async def read_wav(path):
            with wave.open(path, "rb") as audio:
                self.assertEqual(audio.getframerate(), 16000)
                self.assertEqual(audio.getnchannels(), 1)
            return "供应商转写出来的文字"

        provider = types.SimpleNamespace(get_text=AsyncMock(side_effect=read_wav))
        segments = await engine.transcribe(provider, self.clip, self.folder.name, 8)
        self.assertEqual(segments[0]["timing"], "chunk_approximate")
        self.assertEqual(segments[0]["from"], 0)
        self.assertEqual(segments[0]["to"], 8)
        self.assertEqual(segments[0]["text"], "供应商转写出来的文字")
        self.assertFalse(list(Path(self.folder.name).glob("*.wav")))

    async def test_pipeline_cache_dedup_group_knowledge_and_cleanup(self):
        ref = {"bvid": "BV1GJ411x7h7", "page": 1}
        first, second = await asyncio.gather(
            self.analyzer.understand(ref, "group1", "scope1"),
            self.analyzer.understand(ref, "group2", "scope2"),
        )
        self.assertEqual(first, second)
        self.assertTrue(first["visual"])
        self.assertEqual(self.analyzer.client.fetch.await_count, 1)
        self.context.stt.get_text.assert_awaited_once()
        self.context.llm_generate.assert_awaited_once()
        kwargs = self.context.llm_generate.call_args.kwargs
        self.assertEqual(kwargs["chat_provider_id"], "vision")
        self.assertEqual(len(kwargs["image_urls"]), 1)
        self.assertNotIn("group_persona", kwargs["system_prompt"])
        self.assertFalse(list((Path(self.folder.name) / "media").rglob("*.jpg")))
        self.assertFalse(list((Path(self.folder.name) / "media").rglob("*.wav")))
        await self.analyzer.understand(ref, "group1", "scope1")
        self.assertEqual(self.analyzer.client.fetch.await_count, 1)
        self.assertTrue(self.analyzer.knowledge.retrieve("scope1", "刚才那个视频"))
        self.assertFalse(self.analyzer.knowledge.retrieve("scope3", "刚才那个视频"))

    async def test_site_subtitle_skips_asr(self):
        self.metadata["segments"] = [{"from": 0, "to": 3, "text": "站点字幕", "timing": "subtitle"}]
        self.metadata["transcript_source"] = "bilibili_subtitle"
        record = await self.analyzer.understand(
            {"bvid": "BV1GJ411x7h7", "page": 1}, "group", "scope"
        )
        self.context.stt.get_text.assert_not_awaited()
        self.assertEqual(record["transcript_source"], "bilibili_subtitle")

    async def test_subtitle_cache_survives_stt_switch_without_download(self):
        self.metadata["segments"] = [{"from": 0, "to": 3, "text": "站点字幕"}]
        self.metadata["transcript_source"] = "bilibili_subtitle"
        ref = {"bvid": "BV1GJ411x7h7", "page": 1}
        first = await self.analyzer.understand(ref, "group", "scope")
        self.context.providers["new_stt"] = types.SimpleNamespace(
            provider_config={"id": "new_stt"}, get_text=AsyncMock()
        )
        self.config["stt_provider_id"] = "new_stt"
        second = await self.analyzer.understand(ref, "group", "other_scope")
        self.assertEqual(first, second)
        self.analyzer.client.fetch.assert_awaited_once()
        self.context.llm_generate.assert_awaited_once()
        self.context.providers["new_stt"].get_text.assert_not_awaited()
        self.assertTrue(self.analyzer.knowledge.retrieve("other_scope", "刚才那个视频"))

    async def test_asr_cache_is_not_reused_across_stt_switch(self):
        ref = {"bvid": "BV1GJ411x7h7", "page": 1}
        await self.analyzer.understand(ref, "group", "scope")
        replacement = types.SimpleNamespace(
            provider_config={"id": "new_stt"}, get_text=AsyncMock(return_value="重新转写")
        )
        self.context.providers["new_stt"] = replacement
        self.config["stt_provider_id"] = "new_stt"
        await self.analyzer.understand(ref, "group", "scope")
        self.assertEqual(self.analyzer.client.fetch.await_count, 2)
        replacement.get_text.assert_awaited_once()

    async def test_subtitle_cache_still_respects_expiry_and_visual_settings(self):
        self.metadata["segments"] = [{"from": 0, "to": 3, "text": "站点字幕"}]
        self.metadata["transcript_source"] = "bilibili_subtitle"
        ref = {"bvid": "BV1GJ411x7h7", "page": 1}
        await self.analyzer.understand(ref, "group", "scope")
        self.config["stt_provider_id"] = ""
        self.config["storyboard_cells"] = 8
        await self.analyzer.understand(ref, "group", "scope")
        self.assertEqual(self.analyzer.client.fetch.await_count, 2)
        with self.analyzer.knowledge.connect() as db:
            db.execute("UPDATE videos SET created=0")
        self.config["storyboard_cells"] = 4
        await self.analyzer.understand(ref, "group", "scope")
        self.assertEqual(self.analyzer.client.fetch.await_count, 3)

    async def test_text_only_is_not_marked_as_having_seen_frames(self):
        self.config["vision_provider_id"] = "chat"
        self.metadata["segments"] = [{"from": 0, "to": 3, "text": "站点字幕", "timing": "subtitle"}]
        record = await self.analyzer.understand(
            {"bvid": "BV1GJ411x7h7", "page": 1}, "group", "scope"
        )
        self.assertFalse(record["visual"])
        self.assertEqual(self.context.llm_generate.call_args.kwargs["image_urls"], [])

    async def test_model_failure_falls_back_only_to_actual_transcript(self):
        self.context.llm_generate.side_effect = RuntimeError("service unavailable")
        record = await self.analyzer.understand(
            {"bvid": "BV1GJ411x7h7", "page": 1}, "group", "scope"
        )
        self.assertFalse(record["visual"])
        self.assertIn("这是一段语音里的观点", record["understanding"])

    async def test_missing_both_modalities_does_not_fabricate_knowledge(self):
        self.config["vision_provider_id"] = "chat"
        self.context.stt.get_text.side_effect = RuntimeError("ASR unavailable")
        with self.assertRaises(bili.VideoError):
            await self.analyzer.understand({"bvid": "BV1GJ411x7h7", "page": 1}, "group", "scope")
        self.context.llm_generate.assert_not_called()

    async def test_timeout_and_unload_cancel_owned_jobs(self):
        async def slow(ref, directory, on_info=None):
            await asyncio.sleep(30)

        self.analyzer.client.fetch = AsyncMock(side_effect=slow)
        self.config["analysis_timeout"] = 1
        with self.assertRaisesRegex(bili.VideoError, "超时"):
            await self.analyzer.understand({"bvid": "BV1GJ411x7h7", "page": 1}, "group", "scope")
        self.config["analysis_timeout"] = 30
        task = asyncio.create_task(
            self.analyzer.understand({"bvid": "BV1GJ411x7h7", "page": 1}, "group", "scope")
        )
        await asyncio.sleep(0.1)
        await self.analyzer.close()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertFalse(list((Path(self.folder.name) / "media").glob("*")))


if __name__ == "__main__":
    unittest.main()
