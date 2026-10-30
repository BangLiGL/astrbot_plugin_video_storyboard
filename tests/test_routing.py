"""Ownership, quoted cache reuse and optional persona notice contracts."""

import asyncio
import copy
import hashlib
import importlib
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

import test_chat as base

routing = importlib.import_module(base.PACKAGE + ".routing")
media = importlib.import_module(base.PACKAGE + ".media")


def segment(type_, **values):
    return types.SimpleNamespace(type=type_, **values)


def parser_entry(name="on_message", module="data.plugins.astrbot_plugin_parser.main"):
    return types.SimpleNamespace(handler_name=name, handler_module_path=module)


class RoutingTests(unittest.TestCase):
    def test_share_and_conversation_matrix(self):
        passive = [
            "BV1GJ411x7h7",
            "  BV1GJ411x7h7  ",
            "https://www.bilibili.com/video/BV1GJ411x7h7?p=2&spm_id_from=1.2",
            "https://b23.tv/abc123",
            "（BV1GJ411x7h7）",
        ]
        active = [
            "BV1GJ411x7h7 你怎么看",
            "看看 BV1GJ411x7h7 哈哈",
            "你看https://www.bilibili.com/video/BV1GJ411x7h7这个",
            "https://www.bilibili.com/video/BV1GJ411x7h7 好好笑",
            "BV1GJ411x7h7 😂",
        ]
        for text in passive:
            with self.subTest(text=text):
                decision = routing.decide(base.Event(text), True)
                self.assertTrue(decision["passive"])
                self.assertFalse(decision["refs"])
        for text in active:
            with self.subTest(text=text):
                decision = routing.decide(base.Event(text), True)
                self.assertFalse(decision["passive"])
                self.assertTrue(decision["takeover"])
                self.assertTrue(decision["refs"])

    def test_card_and_mention_do_not_become_comments(self):
        card = segment(
            "Json", data=json.dumps({"url": "https://www.bilibili.com/video/BV1GJ411x7h7"})
        )
        event = base.Event("[JSON分享卡片]", messages=[segment("At", qq="bot1"), card])
        self.assertTrue(routing.decide(event, True)["passive"])
        self.assertTrue(routing.decide(event, False)["refs"])

    def test_reply_to_video_or_original_link_triggers(self):
        video = segment("Video", file="/cache/video.mp4", url="https://example.com/v.mp4")
        quoted = segment("Reply", chain=[video])
        result = routing.decide(base.Event("你怎么看", messages=[quoted]), True)
        self.assertEqual(result["refs"][0]["video_sources"][0], "/cache/video.mp4")
        quoted.chain = [segment("Plain", text="BV1GJ411x7h7")]
        self.assertEqual(
            routing.decide(base.Event("", messages=[quoted]), True)["refs"],
            [{"bvid": "BV1GJ411x7h7", "page": 1}],
        )
        self.assertFalse(routing.decide(base.Event("", messages=[video]), True)["refs"])

    def test_skip_only_parser_listener_in_current_queue(self):
        own = parser_entry("on_message", "data.plugins.astrbot_plugin_video_storyboard.main")
        parser = parser_entry()
        command = parser_entry("open_parser")
        other = parser_entry("on_message", "data.plugins.other.main")
        event = base.Event("")
        queue = [own, parser, command, other]
        event.set_extra("activated_handlers", queue)
        visited = []
        for entry in queue:
            visited.append(entry)
            if entry is own:
                routing.skip_parser(event)
        self.assertEqual(visited, [own, command, other])
        self.assertIs(event.get_extra("activated_handlers"), queue)


class CompatibilityChatTests(base.ChatTests):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        self.context.get_registered_star = lambda name: types.SimpleNamespace(activated=True)
        self.persona = {"prompt": "本轮已经加载的群聊人设", "_begin_dialogs_processed": []}
        self.context.persona_manager = types.SimpleNamespace(
            resolve_selected_persona=AsyncMock(
                return_value=("group_persona", self.persona, None, False)
            )
        )

    # Inherit only the setup/teardown, not duplicate the original suite.
    test_bv_starts_one_normal_conversation_request_without_any_report = None
    test_knowledge_injected_without_replacing_persona_or_history = None
    test_failure_is_context_for_honest_persona_reply_not_fake_summary = None
    test_ordinary_chat_and_group_allowlist = None
    test_followup_uses_group_memory_after_restart = None

    async def test_bare_share_never_analyses_even_when_another_plugin_calls_llm(self):
        event = base.Event("BV1GJ411x7h7")
        self.instance.analyzer.understand = AsyncMock()
        self.assertEqual([item async for item in self.instance.on_message(event)], [])
        request = types.SimpleNamespace(prompt=event.text, system_prompt="persona")
        await self.instance.inject_video_knowledge(event, request)
        self.instance.analyzer.understand.assert_not_called()
        self.assertEqual(request.prompt, event.text)
        self.assertTrue(event.call_llm)

    async def test_comment_takes_over_only_parser_and_copies_tools(self):
        event = base.Event("BV1GJ411x7h7 你怎么看")
        parser, other = parser_entry(), parser_entry("other", "data.plugins.other.main")
        event.set_extra("activated_handlers", [parser, other])
        self.assertEqual(len([item async for item in self.instance.on_message(event)]), 1)
        self.assertEqual(event.get_extra("activated_handlers"), [other])
        self.instance.analyzer.understand = AsyncMock(return_value={"understanding": "事实"})

        class Tools:
            names = ["parse_media_url", "weather"]

            def remove_tool(self, name):
                self.names = [value for value in self.names if value != name]

        original = Tools()
        req = types.SimpleNamespace(prompt=event.text, system_prompt="人设", func_tool=original)
        await self.instance.inject_video_knowledge(event, req)
        self.assertEqual(req.func_tool.names, ["weather"])
        self.assertEqual(original.names, ["parse_media_url", "weather"])
        self.assertEqual(req.system_prompt, "人设")

    async def test_compatibility_off_restores_original_bare_bv_trigger(self):
        self.config["parser_compatibility"] = False
        self.assertEqual(
            len([item async for item in self.instance.on_message(base.Event("BV1GJ411x7h7"))]), 1
        )

    async def test_notice_uses_current_chat_persona_once_and_preserves_history(self):
        event = base.Event("看看视频")
        event.send = AsyncMock()
        event.plain_result = lambda text: text
        req = types.SimpleNamespace(
            system_prompt="本轮已经加载的群聊人设", contexts=[{"role": "user", "content": "历史"}]
        )
        original = copy.deepcopy(req)
        self.context.llm_generate.return_value.completion_text = "这段有点长，等我再多看会儿～"
        await self.instance.long_video_notice(event, req, 300)
        await self.instance.long_video_notice(event, req, 300)
        event.send.assert_awaited_once_with("这段有点长，等我再多看会儿～")
        call = self.context.llm_generate.call_args.kwargs
        self.assertEqual(call["chat_provider_id"], "chat")
        self.assertEqual(call["system_prompt"], req.system_prompt)
        self.assertNotIn("tools", call)
        self.assertEqual(vars(req), vars(original))

    async def test_notice_disabled_or_failed_does_not_send(self):
        event = base.Event("")
        event.send = AsyncMock()
        self.config["long_video_notice_enabled"] = False
        await self.instance.long_video_notice(event, types.SimpleNamespace(system_prompt=""), 300)
        self.context.llm_generate.assert_not_called()
        self.config["long_video_notice_enabled"] = True
        self.context.llm_generate.side_effect = RuntimeError("offline")
        await self.instance.long_video_notice(event, types.SimpleNamespace(system_prompt=""), 300)
        event.send.assert_not_called()

    async def test_notice_uses_resolved_current_persona_and_its_examples_not_stale_request(self):
        event = base.Event("")
        examples = [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好喵"}]
        selected = {
            "prompt": "当前强制生效的猫娘人格，说话带喵",
            "_begin_dialogs_processed": examples,
        }
        self.context.persona_manager.resolve_selected_persona.return_value = (
            "forced-persona",
            selected,
            "forced-persona",
            False,
        )
        req = types.SimpleNamespace(system_prompt="旧人格", conversation=self.context.conversation)
        response = await self.instance.generate_long_notice(event, req, 300)
        self.assertTrue(response.completion_text)
        self.context.persona_manager.resolve_selected_persona.assert_awaited_once_with(
            umo=event.unified_msg_origin,
            conversation_persona_id="group_persona",
            platform_name="test",
            provider_settings={},
        )
        call = self.context.llm_generate.call_args.kwargs
        self.assertEqual(call["system_prompt"], selected["prompt"])
        self.assertEqual(call["contexts"], examples)
        self.assertIsNot(call["contexts"], examples)
        self.assertEqual(req.system_prompt, "旧人格")

    async def test_unresolved_persona_never_generates_a_generic_notice(self):
        event = base.Event("")
        event.send = AsyncMock()
        self.context.persona_manager.resolve_selected_persona.return_value = (
            None,
            None,
            None,
            False,
        )
        await self.instance.long_video_notice(
            event, types.SimpleNamespace(system_prompt="通用提示"), 300
        )
        self.context.llm_generate.assert_not_called()
        event.send.assert_not_called()

    async def test_persona_lookup_failure_never_generates_a_generic_notice(self):
        event = base.Event("")
        event.send = AsyncMock()
        self.context.persona_manager.resolve_selected_persona.side_effect = RuntimeError(
            "unavailable"
        )
        await self.instance.long_video_notice(
            event, types.SimpleNamespace(system_prompt="通用提示"), 300
        )
        self.context.llm_generate.assert_not_called()
        event.send.assert_not_called()


class QuoteAndNoticeMediaTests(base.MediaTests):
    test_scene_detection_and_hidden_storyboard = None
    test_astrbot_stt_receives_real_wav_and_coarse_timestamps = None
    test_pipeline_cache_dedup_group_knowledge_and_cleanup = None
    test_site_subtitle_skips_asr = None
    test_subtitle_cache_survives_stt_switch_without_download = None
    test_asr_cache_is_not_reused_across_stt_switch = None
    test_subtitle_cache_still_respects_expiry_and_visual_settings = None
    test_text_only_is_not_marked_as_having_seen_frames = None
    test_model_failure_falls_back_only_to_actual_transcript = None
    test_missing_both_modalities_does_not_fabricate_knowledge = None
    test_timeout_and_unload_cancel_owned_jobs = None

    async def test_quoted_local_cache_borrowed_without_download_or_deletion(self):
        self.analyzer.data_root = Path(self.fixture.name)
        original = hashlib.sha256(self.clip.read_bytes()).hexdigest()
        self.analyzer.client._get = AsyncMock(side_effect=AssertionError("no download"))
        result = await self.analyzer.understand(
            {"video_sources": [str(self.clip)]}, "group", "scope"
        )
        self.assertTrue(result["visual"])
        self.assertEqual(result["transcript_source"], "astrbot_stt_chunks")
        self.assertNotIn("video_path", result)
        self.assertEqual(hashlib.sha256(self.clip.read_bytes()).hexdigest(), original)
        self.analyzer.client._get.assert_not_called()
        self.analyzer.client.fetch.assert_not_called()
        again = await self.analyzer.understand(
            {"video_sources": [str(self.clip)]}, "group", "scope"
        )
        self.assertEqual(result, again)
        self.context.llm_generate.assert_awaited_once()

    async def test_notice_starts_before_work_finishes_and_not_on_cache_hit(self):
        self.config["long_video_notice_threshold"] = 5
        seen = asyncio.Event()
        callback = AsyncMock(side_effect=lambda duration: seen.set())
        original = self.analyzer.client.fetch.side_effect

        async def held_fetch(ref, directory, on_info=None):
            await on_info(self.metadata)
            await asyncio.wait_for(seen.wait(), timeout=1)
            return await original(ref, directory, on_info=on_info)

        self.analyzer.client.fetch.side_effect = held_fetch
        ref = {"bvid": "BV1GJ411x7h7", "page": 1}
        await self.analyzer.understand(ref, "group", "scope", on_long=callback)
        callback.assert_awaited_once_with(8.0)
        await self.analyzer.understand(ref, "group", "scope", on_long=callback)
        callback.assert_awaited_once()

    async def test_short_video_does_not_notice(self):
        self.config["long_video_notice_threshold"] = 60
        callback = AsyncMock()
        await self.analyzer.understand(
            {"bvid": "BV1GJ411x7h7", "page": 1}, "group", "scope", on_long=callback
        )
        callback.assert_not_called()

    async def test_completed_work_cancels_a_slow_notice(self):
        self.config["long_video_notice_threshold"] = 5
        cancelled = asyncio.Event()

        async def slow_notice(duration):
            try:
                await asyncio.sleep(30)
            finally:
                cancelled.set()

        await self.analyzer.understand(
            {"bvid": "BV1GJ411x7h7", "page": 1}, "group", "scope", on_long=slow_notice
        )
        self.assertTrue(cancelled.is_set())


class SourceSafetyTests(unittest.TestCase):
    def test_local_source_is_limited_to_data_root_and_size(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "cache.mp4"
            path.write_bytes(b"sample")
            result = media.resolve_sources([path.as_uri()], root, 100)
            self.assertEqual(result["local_video"], str(path.resolve()))
            with self.assertRaises(Exception):
                media.resolve_sources([str(path)], root / "other", 100)
            with self.assertRaises(Exception):
                media.resolve_sources([str(path)], root, 1)
            with self.assertRaises(Exception):
                media.resolve_sources(["http://127.0.0.1/video"], root, 100)
