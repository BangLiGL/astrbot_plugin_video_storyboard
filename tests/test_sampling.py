"""Regression coverage for the upstream-inspired v0.4.0 sampling and cache changes."""

import asyncio
import importlib
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np
import test_chat as base
from PIL import Image

sampling = importlib.import_module(base.PACKAGE + ".sampling")


class PlanningTests(unittest.TestCase):
    def test_hue_wrap_and_nearly_grey_noise_do_not_become_scene_changes(self):
        a = np.array([[[0, 255, 180]]], dtype=np.float32)
        b = np.array([[[179, 255, 180]]], dtype=np.float32)
        self.assertLess(sampling.hsv_distance(a, b), 0.003)
        a[..., 1], b[..., 1] = 1, 1
        b[..., 0] = 90
        self.assertLess(sampling.hsv_distance(a, b), 0.001)
        b[..., 2] = 0
        self.assertGreater(sampling.hsv_distance(a, b), 0.2)

    def test_busy_interval_gets_more_samples_without_losing_quiet_timeline(self):
        scores = [(i / 10, 0.3 if 20 <= i / 10 < 40 else 0.01) for i in range(600)]
        boundaries = sampling.density_boundaries(scores, 60, 12)
        self.assertEqual((boundaries[0], boundaries[-1]), (0, 60))
        self.assertTrue(all(a < b for a, b in zip(boundaries, boundaries[1:])))
        centres = [(a + b) / 2 for a, b in zip(boundaries, boundaries[1:])]
        self.assertGreater(sum(20 <= t < 40 for t in centres), 4)
        self.assertTrue(any(t < 10 for t in centres))
        self.assertTrue(any(t > 50 for t in centres))

    def test_image_budget_caps_and_explicit_capacity_use_actual_render_geometry(self):
        for aspect in (16 / 9, 9 / 16, 1, 2.4, 100000, 0.00001):
            for need in sampling.NEEDS:
                plan = sampling.plan_images(32, aspect, 640000, need, max_images=4)
                self.assertLessEqual(len(plan["layouts"]), 4)
                self.assertEqual(sum(plan["counts"]), plan["actual_cells"])
                for canvas in plan["layouts"]:
                    self.assertLessEqual(canvas["width"] * canvas["height"], 640000)
        plan = sampling.plan_images(12, 16 / 9, max_per_sheet=2, max_images=6)
        self.assertEqual(plan["counts"], [2] * 6)

    def test_budget_adjustment_and_unreadable_override_are_visible(self):
        plan = sampling.plan_images(16, 16 / 9, 200000, "text", 1, 2)
        self.assertEqual(plan["actual_cells"], 2)
        self.assertTrue(plan["warnings"])
        plan = sampling.plan_images(16, 16 / 9, 200000, "text", 16, 1)
        self.assertTrue(plan["warnings"])

    def test_one_cell_stays_one_frame_and_not_a_hidden_four_frame_montage(self):
        plan = sampling.plan_images(1, 16 / 9)
        self.assertEqual(plan["counts"], [1])

    def test_transcript_expiry_does_not_slide_when_read_and_paths_are_not_saved(self):
        with tempfile.TemporaryDirectory() as folder:
            kb = base.knowledge.Knowledge(Path(folder) / "knowledge.sqlite3")
            rows = [
                {
                    "from": 3,
                    "to": 7,
                    "text": "words",
                    "timing": "chunk_approximate",
                    "path": "private-path",
                    "api_key": "private-key",
                }
            ]
            kb.save_transcript("key", rows)
            self.assertEqual(kb.cached_transcript("key")[0]["from"], 3)
            with kb.connect() as db:
                db.execute("UPDATE transcripts SET created=0")
            self.assertIsNone(kb.cached_transcript("key", 1))
            self.assertNotIn(b"private-path", kb.path.read_bytes())
            self.assertNotIn(b"private-key", kb.path.read_bytes())


class SamplingMediaTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        base.MediaTests.setUpClass.__func__(cls)

    @classmethod
    def tearDownClass(cls):
        cls.fixture.cleanup()

    async def asyncSetUp(self):
        await base.MediaTests.asyncSetUp(self)

    async def asyncTearDown(self):
        await base.MediaTests.asyncTearDown(self)

    async def test_real_multi_image_evidence_and_cleanup(self):
        self.config["storyboard_max_per_sheet"] = 1
        await self.analyzer.understand({"bvid": "BV1GJ411x7h7", "page": 1}, "g", "g")
        kwargs = self.context.llm_generate.call_args.kwargs
        self.assertEqual(len(kwargs["image_urls"]), 4)
        evidence = json.loads(kwargs["prompt"])
        self.assertEqual([row["image"] for row in evidence["visible_cells"]], [1, 2, 3, 4])
        self.assertEqual(evidence["visible_cells"][0]["from"], 0)
        self.assertEqual(evidence["visible_cells"][-1]["to"], 8)
        self.assertFalse(list((Path(self.folder.name) / "media").glob("*")))

    async def test_real_source_resolution_is_not_faked_by_upscaling(self):
        result = base.engine.storyboard(
            self.clip, self.folder.name, 4, options={"storyboard_readability": "text"}
        )
        self.assertTrue(result["plan"]["warnings"])
        for cell in result["cells"]:
            self.assertLessEqual(cell["pixels"][0], 320)
            self.assertLessEqual(cell["pixels"][1], 180)
            self.assertNotEqual(cell["readability"], "text")
        for path in result["images"]:
            with Image.open(path) as image:
                self.assertLessEqual(image.width * image.height, 1200000)

    async def test_visual_setting_changes_reuse_asr_but_new_hotwords_do_not(self):
        ref = {"bvid": "BV1GJ411x7h7", "page": 1}
        await self.analyzer.understand(ref, "g", "g")
        self.config["storyboard_readability"] = "text"
        await self.analyzer.understand(ref, "g", "g")
        self.assertEqual(self.context.stt.get_text.await_count, 1)
        self.assertEqual(self.context.llm_generate.await_count, 2)
        self.context.stt.provider_config.update({"language": "zh", "hotwords": ["新热词"]})
        await self.analyzer.understand(ref, "g", "g")
        self.assertEqual(self.context.stt.get_text.await_count, 2)

    async def test_expired_transcript_is_transcribed_again(self):
        ref = {"bvid": "BV1GJ411x7h7", "page": 1}
        await self.analyzer.understand(ref, "g", "g")
        with self.analyzer.knowledge.connect() as db:
            db.execute("UPDATE videos SET created=0")
            db.execute("UPDATE transcripts SET created=0")
        await self.analyzer.understand(ref, "g", "g")
        self.assertEqual(self.context.stt.get_text.await_count, 2)

    async def test_plain_text_model_does_not_decode_unused_storyboards(self):
        self.config["vision_provider_id"] = "chat"
        with patch.object(base.analysis, "storyboard") as render:
            await self.analyzer.understand({"bvid": "BV1GJ411x7h7", "page": 1}, "g", "g")
        render.assert_not_called()

    async def test_cancelled_sampling_stops_before_writing_images(self):
        cancel = threading.Event()
        cancel.set()
        with self.assertRaises(base.bili.VideoError):
            await asyncio.to_thread(base.engine.storyboard, self.clip, self.folder.name, 12, cancel)
        self.assertFalse(list(Path(self.folder.name).glob("*.jpg")))


class QualityTests(unittest.IsolatedAsyncioTestCase):
    async def test_requested_download_quality_is_independent_of_canvas_settings(self):
        import types

        sdk = types.SimpleNamespace(
            get_info=AsyncMock(return_value={"pages": [{"cid": 1, "duration": 5}]}),
            get_subtitle=AsyncMock(return_value={"subtitles": []}),
            get_download_url=AsyncMock(
                return_value={
                    "dash": {
                        "video": [
                            {
                                "height": height,
                                "codecs": "avc",
                                "baseUrl": "https://example.com/video",
                            }
                            for height in (360, 720, 1080)
                        ]
                    }
                }
            ),
        )
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.object(base.bili.video, "Video", return_value=sdk),
        ):
            for quality, expected in (("360p", 360), ("720p", 720), ("1080p", 1080)):
                client = base.bili.BiliClient(
                    {"video_quality": quality, "storyboard_pixel_budget": 640000}
                )
                client._stream = AsyncMock(return_value="video.mp4")
                result = await client.fetch({"bvid": "BV1GJ411x7h7"}, folder)
                self.assertEqual(result["download_height"], expected)
                self.assertEqual(client._stream.call_args.args[0]["height"], expected)
