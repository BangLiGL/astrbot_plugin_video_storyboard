"""Subtitle diagnostics, privacy and recovery from temporarily degraded evidence."""

import asyncio
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import aiohttp
import test_chat as base


class SubtitleTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.client = base.bili.BiliClient({})
        self.item = type("SDK", (), {})()
        self.item.get_subtitle = AsyncMock()
        self.client._get = AsyncMock(return_value=b'{"body":[{"from":0,"to":2,"content":"CC"}]}')

    async def fetch(self):
        result = {"segments": [], "warnings": [], "transcript_source": "none"}
        await self.client._subtitles(self.item, 1, result)
        return result

    async def test_missing_and_login_gated_subtitles_have_distinct_states(self):
        for metadata, state in (
            ({"subtitles": []}, "none"),
            ({"subtitles": [], "need_login_subtitle": True}, "login_required"),
        ):
            self.item.get_subtitle.return_value = metadata
            result = await self.fetch()
            self.assertEqual(result["site_subtitle"]["status"], state)
            self.assertEqual(bool(result["warnings"]), state == "login_required")
        self.client._get.assert_not_called()

    async def test_safe_error_categories_without_exception_payloads(self):
        secret = "SESSDATA=private https://host/subtitle?token=private"
        cases = (
            (TimeoutError(secret), "failed", "network_error"),
            (base.bili.exceptions.NetworkException(500, secret), "failed", "network_error"),
            (
                aiohttp.ClientResponseError(None, (), status=503, message=secret),
                "failed",
                "http_error",
            ),
            (
                base.bili.exceptions.ResponseCodeException(-101, secret),
                "login_required",
                "authentication",
            ),
            (base.bili.exceptions.ResponseCodeException(-400, secret), "failed", "api_error"),
            (
                base.bili.exceptions.CredentialNoSessdataException(),
                "login_required",
                "authentication",
            ),
            (RuntimeError(secret), "failed", "request_failed"),
        )
        for error, status, reason in cases:
            with self.subTest(reason=reason):
                self.item.get_subtitle.side_effect = error
                result = await self.fetch()
                self.assertEqual(result["site_subtitle"]["status"], status)
                self.assertEqual(result["site_subtitle"]["reason"], reason)
                self.assertNotIn("private", json.dumps(result))

    async def test_invalid_subtitle_responses_are_not_reported_as_no_subtitles(self):
        metadata = {"subtitles": [{"lan": "ai-zh", "subtitle_url": "https://example.com/cc"}]}
        cases = (
            b"not json",
            b"{}",
            b'{"body":{}}',
            b'{"body":[{"from":0,"to":"NaN","content":"x"}]}',
        )
        self.item.get_subtitle.return_value = metadata
        for body in cases:
            with self.subTest(body=body):
                self.client._get.return_value = body
                result = await self.fetch()
                self.assertEqual(
                    result["site_subtitle"], {"status": "failed", "reason": "invalid_response"}
                )
                self.assertEqual(result["segments"], [])

    async def test_transient_failure_retries_and_success_keeps_source_language(self):
        self.item.get_subtitle.side_effect = [
            TimeoutError(),
            {"subtitles": [{"lan": "ai-zh", "subtitle_url": "//example.com/cc"}]},
        ]
        self.assertEqual((await self.fetch())["site_subtitle"]["status"], "failed")
        result = await self.fetch()
        self.assertEqual(result["site_subtitle"], {"status": "available", "language": "ai-zh"})
        self.assertEqual(result["transcript_source"], "bilibili_subtitle")
        self.assertEqual(result["segments"][0]["text"], "CC")
        self.assertEqual(result["warnings"], [])

    async def test_cancellation_is_not_downgraded_to_a_subtitle_failure(self):
        self.item.get_subtitle.side_effect = asyncio.CancelledError()
        with self.assertRaises(asyncio.CancelledError):
            await self.fetch()


class RecoveryCacheTests(unittest.TestCase):
    def test_degraded_cache_expires_in_five_minutes_without_deleting_memories(self):
        with tempfile.TemporaryDirectory() as folder:
            kb = base.knowledge.Knowledge(Path(folder) / "knowledge.sqlite3")
            record = {
                "understanding": "Partial visual evidence",
                "transcript_status": "subtitle_failed",
                "site_subtitle": {"status": "failed", "reason": "network_error"},
            }
            with patch.object(base.knowledge.time, "time", return_value=1000):
                kb.save("video", record)
                kb.remember("group", "video", record)
            with patch.object(base.knowledge.time, "time", return_value=1299):
                self.assertIsNotNone(kb.cached("video"))
                self.assertIsNone(kb.cached("video", hours=0))
            with patch.object(base.knowledge.time, "time", return_value=1300):
                self.assertIsNone(kb.cached("video"))
                self.assertTrue(kb.retrieve("group", "刚才那个视频"))
            with kb.connect() as db:
                self.assertEqual(db.execute("SELECT COUNT(*) FROM videos").fetchone()[0], 1)

    def test_successful_fallback_and_genuine_absence_keep_normal_ttl(self):
        with tempfile.TemporaryDirectory() as folder:
            kb = base.knowledge.Knowledge(Path(folder) / "knowledge.sqlite3")
            for source in ("embedded_subtitle", "astrbot_stt_chunks"):
                with patch.object(base.knowledge.time, "time", return_value=1000):
                    kb.save(
                        source,
                        {
                            "transcript_source": source,
                            "transcript_status": "available",
                            "site_subtitle": {"status": "failed"},
                        },
                    )
                with patch.object(base.knowledge.time, "time", return_value=2000):
                    self.assertIsNotNone(kb.cached(source))
            with patch.object(base.knowledge.time, "time", return_value=1000):
                kb.save(
                    "absent",
                    {"transcript_status": "not_configured", "site_subtitle": {"status": "none"}},
                )
            with patch.object(base.knowledge.time, "time", return_value=2000):
                self.assertIsNotNone(kb.cached("absent"))

    def test_legacy_failure_records_and_asr_failures_also_retry(self):
        with tempfile.TemporaryDirectory() as folder:
            kb = base.knowledge.Knowledge(Path(folder) / "knowledge.sqlite3")
            cases = (
                {
                    "warnings": ["站点字幕不可用；有配置语音识别时尝试转写"],
                    "transcript_status": "not_configured",
                },
                {"transcript_status": "failed"},
                {"embedded_subtitle": "unreadable", "transcript_status": "not_configured"},
            )
            for i, record in enumerate(cases):
                with patch.object(base.knowledge.time, "time", return_value=1000):
                    kb.save(str(i), record)
                with patch.object(base.knowledge.time, "time", return_value=1301):
                    self.assertIsNone(kb.cached(str(i)))


class PipelineRecoveryTests(unittest.IsolatedAsyncioTestCase):
    setUpClass = classmethod(base.MediaTests.setUpClass.__func__)
    tearDownClass = classmethod(base.MediaTests.tearDownClass.__func__)
    asyncSetUp = base.MediaTests.asyncSetUp
    asyncTearDown = base.MediaTests.asyncTearDown

    async def test_no_usable_modality_keeps_actual_subtitle_failure_in_error(self):
        self.config.update(vision_provider_id="chat", stt_provider_id="")
        self.context.get_using_stt_provider_async.return_value = None
        self.metadata["site_subtitle"] = {"status": "failed", "reason": "network_error"}
        with self.assertRaisesRegex(base.bili.VideoError, "网络连接或请求超时"):
            await self.analyzer.understand({"bvid": "BV1GJ411x7h7", "page": 1}, "group", "scope")
        self.context.llm_generate.assert_not_called()

    async def test_degraded_result_recovers_after_cooldown_without_clearing_knowledge(self):
        self.config["stt_provider_id"] = ""
        self.context.get_using_stt_provider_async.return_value = None
        count = 0

        async def fetch(ref, directory, on_info=None):
            nonlocal count
            count += 1
            media = copy.deepcopy(self.metadata)
            media["site_subtitle"] = {"status": "failed", "reason": "network_error"}
            if count > 1:
                media.update(
                    site_subtitle={"status": "available", "language": "zh-CN"},
                    segments=[{"from": 0, "to": 2, "text": "Recovered CC"}],
                    transcript_source="bilibili_subtitle",
                )
            return media

        self.analyzer.client.fetch.side_effect = fetch
        ref = {"bvid": "BV1GJ411x7h7", "page": 1}
        first = await self.analyzer.understand(ref, "group", "scope")
        self.assertEqual(first["transcript_status"], "subtitle_failed")
        await self.analyzer.understand(ref, "group", "scope")
        self.assertEqual(count, 1)
        with self.analyzer.knowledge.connect() as db:
            db.execute("UPDATE videos SET created=created-301")
        recovered = await self.analyzer.understand(ref, "group", "scope")
        self.assertEqual(count, 2)
        self.assertEqual(recovered["transcript_status"], "available")
        self.assertEqual(recovered["transcript_source"], "bilibili_subtitle")
        saved = self.analyzer.knowledge.retrieve("scope", "刚才那个视频")[0]
        self.assertEqual(saved["site_subtitle"]["language"], "zh-CN")
        self.context.stt.get_text.assert_not_called()

    async def test_asr_success_preserves_site_failure_but_keeps_successful_cache(self):
        self.metadata["site_subtitle"] = {"status": "login_required", "reason": "authentication"}
        ref = {"bvid": "BV1GJ411x7h7", "page": 1}
        record = await self.analyzer.understand(ref, "group", "scope")
        self.assertEqual(record["transcript_source"], "astrbot_stt_chunks")
        self.assertEqual(record["site_subtitle"]["status"], "login_required")
        payload = json.loads(self.context.llm_generate.call_args.kwargs["prompt"])
        self.assertEqual(payload["transcript_status"], "available")
        self.assertEqual(payload["site_subtitle"]["status"], "login_required")
        with self.analyzer.knowledge.connect() as db:
            db.execute("UPDATE videos SET created=created-301")
        await self.analyzer.understand(ref, "group", "scope")
        self.analyzer.client.fetch.assert_awaited_once()
