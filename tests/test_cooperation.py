"""Shared reply ownership, request-local evidence, and both long-notice modes."""

import asyncio
import copy
import importlib
import types
import unittest
from unittest.mock import AsyncMock

import test_chat as base

cooperation = importlib.import_module(base.PACKAGE + ".cooperation")


class CooperationTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        await base.ChatTests.asyncSetUp(self)
        self.instance.analyzer.understand = AsyncMock(
            return_value={"title": "video", "understanding": "Grounded video evidence"}
        )
        self.context.persona_manager = types.SimpleNamespace(
            resolve_selected_persona=AsyncMock(
                return_value=(
                    "group_persona",
                    {"prompt": "Current persona", "_begin_dialogs_processed": []},
                    None,
                    False,
                )
            )
        )

    async def asyncTearDown(self):
        await base.ChatTests.asyncTearDown(self)

    def event(self):
        return base.Event("看看 BV1GJ411x7h7，你怎么看？")

    def request(self):
        return types.SimpleNamespace(
            prompt="The companion's merged user turn",
            system_prompt="Current persona\nCompanion relationship and state",
            contexts=[{"role": "assistant", "content": "Previous conversation"}],
            extra_user_content_parts=[{"type": "text", "text": "Companion turn notes"}],
        )

    async def test_auto_mode_defers_to_companion_without_rewriting_its_flags(self):
        event = self.event()
        companion = types.SimpleNamespace(
            handler_module_path="data.plugins.astrbot_plugin_private_companion.main",
            handler_name="on_group_message",
        )
        event.set_extra("activated_handlers", [companion])
        await self.instance.on_message(event)
        self.assertFalse(event.is_at_or_wake_command)
        self.assertFalse(event.is_wake)
        self.assertFalse(event.call_llm)
        self.assertEqual(event.requests, [])
        self.assertIs(event.get_extra("activated_handlers")[0], companion)
        self.instance.analyzer.understand.assert_not_called()
        # A companion-admitted request gets the same evidence without replacing its content.
        event.is_at_or_wake_command = True
        request = self.request()
        original = copy.deepcopy(request)
        await self.instance.inject_video_knowledge(event, request)
        self.assertEqual(request.prompt, original.prompt)
        self.assertEqual(request.system_prompt, original.system_prompt)
        self.assertEqual(request.contexts, original.contexts)
        self.assertEqual(request.extra_user_content_parts[0], original.extra_user_content_parts[0])
        self.assertIn("Grounded video evidence", request.extra_user_content_parts[-1]["text"])

    async def test_stopped_or_suppressed_or_already_answered_events_are_not_revived(self):
        for state in ("stopped", "suppressed", "sent"):
            with self.subTest(state=state):
                event = self.event()
                if state == "stopped":
                    event.stop_event()
                elif state == "suppressed":
                    event.should_call_llm(True)
                else:
                    event._has_send_oper = True
                await self.instance.on_message(event)
                self.assertFalse(event.is_at_or_wake_command)
                self.assertEqual(event.call_llm, state == "suppressed")
                self.assertEqual(event.requests, [])
                if state == "stopped":
                    request = self.request()
                    await self.instance.inject_video_knowledge(event, request)
                    self.assertEqual(len(request.extra_user_content_parts), 1)
        self.instance.analyzer.understand.assert_not_called()

    async def test_passive_mode_supports_unknown_chat_controllers(self):
        self.config["chat_trigger_mode"] = "passive"
        event = self.event()
        await self.instance.on_message(event)
        self.assertFalse(event.is_at_or_wake_command)
        self.assertFalse(event.call_llm)
        await self.instance.inject_video_knowledge(event, self.request())
        self.instance.analyzer.understand.assert_awaited_once()

    async def test_companion_detection_respects_current_handler_queue_and_plugin_allowlist(self):
        self.context.get_registered_star = lambda _: types.SimpleNamespace(activated=True)
        event = self.event()
        self.assertTrue(cooperation.companion_active(self.context, event))
        event.set_extra("activated_handlers", [])
        self.assertFalse(cooperation.companion_active(self.context, event))
        event.extras.pop("activated_handlers")
        event.plugins_name = ["astrbot_plugin_video_storyboard"]
        self.assertFalse(cooperation.companion_active(self.context, event))
        event.plugins_name = ["*"]
        self.assertTrue(cooperation.companion_active(self.context, event))
        self.context.get_registered_star = lambda _: types.SimpleNamespace(activated=False)
        self.assertFalse(cooperation.companion_active(self.context, event))

    async def test_evidence_survives_companion_prompt_finalization_and_request_rebuild(self):
        event = self.event()
        request = self.request()
        # A companion's final injection-plan render has already materialized these fields.
        original_parts = request.extra_user_content_parts
        await self.instance.inject_video_knowledge(event, request)
        self.assertEqual(len(original_parts), 1)
        self.assertEqual(len(request.extra_user_content_parts), 2)
        await self.instance.inject_video_knowledge(event, request)
        self.assertEqual(len(request.extra_user_content_parts), 2)
        rebuilt = self.request()
        await self.instance.inject_video_knowledge(event, rebuilt)
        self.assertEqual(len(rebuilt.extra_user_content_parts), 2)
        self.instance.analyzer.understand.assert_awaited_once()
        self.assertEqual(rebuilt.prompt, "The companion's merged user turn")

    async def test_only_our_evidence_is_temporary_in_new_and_rebuilt_requests(self):
        event = self.event()
        for request in (self.request(), self.request()):
            permanent = request.extra_user_content_parts[0]
            await self.instance.inject_video_knowledge(event, request)
            self.assertIs(request.extra_user_content_parts[0], permanent)
            self.assertNotIn("_no_save", permanent)
            block = request.extra_user_content_parts[-1]
            self.assertTrue(block["_no_save"])
            self.assertEqual(block["type"], "text")
            self.assertIn("Grounded video evidence", block["text"])
        self.instance.analyzer.understand.assert_awaited_once()

    async def test_failed_video_evidence_is_also_temporary(self):
        self.instance.analyzer.understand.side_effect = base.bili.VideoError("video unavailable")
        event, request = self.event(), self.request()
        await self.instance.inject_video_knowledge(event, request)
        self.assertTrue(request.extra_user_content_parts[-1]["_no_save"])
        self.assertIn("video unavailable", request.extra_user_content_parts[-1]["text"])

    async def test_followup_retrieves_durable_facts_as_temporary_evidence(self):
        event = base.Event("刚才那个视频在讲什么？")
        self.instance.analyzer.knowledge.remember(
            self.instance.scope(event),
            "known",
            {"title": "已看视频", "understanding": "保存的事实"},
        )
        request = self.request()
        await self.instance.inject_video_knowledge(event, request)
        block = request.extra_user_content_parts[-1]
        self.assertTrue(block["_no_save"])
        self.assertIn("保存的事实", block["text"])
        self.instance.analyzer.understand.assert_not_called()
        unrelated = self.request()
        await self.instance.inject_video_knowledge(base.Event("今晚吃什么？"), unrelated)
        self.assertEqual(len(unrelated.extra_user_content_parts), 1)

    async def test_cancellation_during_analysis_never_attaches_partial_evidence(self):
        self.instance.analyzer.understand.side_effect = asyncio.CancelledError()
        event, request = self.event(), self.request()
        with self.assertRaises(asyncio.CancelledError):
            await self.instance.inject_video_knowledge(event, request)
        self.assertEqual(len(request.extra_user_content_parts), 1)
        self.assertIsNone(event.get_extra("bili_chat_evidence"))

    async def test_fixed_notice_is_literal_and_never_calls_model_or_persona(self):
        self.config.update(
            long_video_notice_mode="fixed",
            long_video_notice_text='还有{duration_minutes}分钟的内容，慢慢看。{保留} {"x":1}',
        )
        event = self.event()
        event.send = AsyncMock()
        event.plain_result = lambda text: text
        await self.instance.long_video_notice(event, self.request(), 300)
        await self.instance.long_video_notice(event, self.request(), 300)
        event.send.assert_awaited_once_with('还有5.0分钟的内容，慢慢看。{保留} {"x":1}')
        self.context.llm_generate.assert_not_called()
        self.context.persona_manager.resolve_selected_persona.assert_not_called()

    async def test_llm_custom_prompt_keeps_current_persona_examples_and_enrichments(self):
        self.config["long_video_notice_prompt"] = (
            "用俏皮的口吻提醒，需要看{duration_seconds}秒的视频。"
        )
        examples = [{"role": "assistant", "content": "Persona speech example"}]
        selected = self.context.persona_manager.resolve_selected_persona.return_value[1]
        selected["_begin_dialogs_processed"] = examples
        event, request = self.event(), self.request()
        before = copy.deepcopy(request)
        await self.instance.generate_long_notice(event, request, 301)
        call = self.context.llm_generate.call_args.kwargs
        self.assertIn("301秒", call["prompt"])
        self.assertIn("俏皮", call["prompt"])
        self.assertEqual(call["system_prompt"], request.system_prompt)
        self.assertEqual(call["contexts"], examples)
        self.assertEqual(vars(request), vars(before))

    async def test_empty_fixed_notice_and_unknown_mode_do_not_fall_back_to_llm(self):
        for mode in ("fixed", "unknown"):
            self.config.update(long_video_notice_mode=mode, long_video_notice_text="  ")
            event = self.event()
            event.send = AsyncMock()
            await self.instance.long_video_notice(event, self.request(), 300)
            event.send.assert_not_called()
        self.context.llm_generate.assert_not_called()

    async def test_stopped_notice_never_generates_or_sends(self):
        event = self.event()
        event.stop_event()
        event.send = AsyncMock()
        await self.instance.long_video_notice(event, self.request(), 300)
        self.context.llm_generate.assert_not_called()
        event.send.assert_not_called()

    async def test_stop_while_llm_notice_is_generating_prevents_delivery(self):
        event = self.event()
        event.send = AsyncMock()

        async def generate(**kwargs):
            event.stop_event()
            return types.SimpleNamespace(completion_text="Notice")

        self.context.llm_generate.side_effect = generate
        await self.instance.long_video_notice(event, self.request(), 300)
        event.send.assert_not_called()

    async def test_schema_defaults_preserve_existing_llm_notice_mode(self):
        import json

        schema = json.loads((base.ROOT / "_conf_schema.json").read_text(encoding="utf-8"))
        self.assertEqual(schema["long_video_notice_mode"]["default"], "llm")
        self.assertEqual(schema["chat_trigger_mode"]["default"], "auto")
        self.assertEqual(
            schema["long_video_notice_prompt"]["default"], base.main.DEFAULT_NOTICE_PROMPT
        )
