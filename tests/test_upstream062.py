"""Motion-aware scan and destination-volume storage regressions (upstream v0.6.7)."""

import asyncio
import importlib
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import numpy as np
import test_chat as base
from test_decode import ffmpeg

sampling = importlib.import_module(base.PACKAGE + ".sampling")
decode = importlib.import_module(base.PACKAGE + ".decode")


class MotionTests(unittest.TestCase):
    def test_local_translation_is_absorbed_but_a_new_scene_remains(self):
        a = np.zeros((8, 12, 3), np.float32)
        a[..., 1:] = 200
        a[:, ::2, 0] = 90
        b = np.roll(a, 1, axis=1)
        self.assertGreater(sampling.motion_score(a, b, radius=0), 0.9)
        self.assertEqual(sampling.motion_score(a, b), 0)
        self.assertEqual(sampling.motion_score(a, a.copy()), 0)
        b[..., 1:] = 0
        self.assertEqual(sampling.motion_score(a, b), 1)

    def test_motion_search_respects_borders_and_grey_hue_noise(self):
        a = np.array([[[0, 1, 128]]], np.float32)
        b = np.array([[[90, 1, 128]]], np.float32)
        self.assertEqual(sampling.motion_score(a, b), 0)
        b[..., 2] = 0
        self.assertEqual(sampling.motion_score(a, b), 1)

    def test_noise_floor_uses_normalized_p75_and_stays_bounded(self):
        self.assertEqual(sampling.motion_floor([]) * 255, 80)
        self.assertAlmostEqual(sampling.motion_floor([(0, 2.5 / 255)]) * 255, 80)
        self.assertAlmostEqual(sampling.motion_floor([(0, 6.6 / 255)]) * 255, 140)
        self.assertAlmostEqual(sampling.motion_floor([(0, 10)]) * 255, 150)

    def scores(self, marks, duration=60, step=0.1):
        return [
            (i * step, 0.15 if any(abs(i * step - at) < 0.01 for at in marks) else 0)
            for i in range(round(duration / step))
        ]

    def test_sparse_slide_updates_survive_high_motion_cut_floor(self):
        scores = self.scores([10, 30, 50])
        self.assertEqual(sampling.scene_cuts(scores, 60, mode="mc"), [10, 30, 50])

    def test_still_fallback_rejects_dense_noise_and_near_duplicate_cuts(self):
        coarse = [10, 30]
        dense = self.scores(list(range(1, 60)))
        self.assertEqual(sampling.sparse_spikes(dense, coarse, 60), coarse)
        duplicate = self.scores([9, 10, 11, 29, 30, 31])
        self.assertEqual(sampling.sparse_spikes(duplicate, coarse, 60), coarse)
        novel = self.scores([10, 20, 30, 40, 50])
        self.assertEqual(sampling.sparse_spikes(novel, coarse, 60), [10, 20, 30, 40, 50])

    def test_high_change_content_never_relaxes_image_or_pixel_limits(self):
        for aspect in (16 / 9, 9 / 16, 2.4, 0.01, 100):
            plan = sampling.plan_images(32, aspect, 200000, "text", max_images=2)
            self.assertLessEqual(len(plan["layouts"]), 2)
            self.assertLessEqual(plan["actual_cells"], 32)
            for canvas in plan["layouts"]:
                self.assertLessEqual(canvas["width"] * canvas["height"], 200000)

    def test_real_pan_is_not_an_edit_but_following_color_cut_is_detected(self):
        frames = np.zeros((144, 36, 64, 3), np.uint8)
        for i in range(96):
            frames[i, :, (np.arange(64) + i) % 8 < 4, 2] = 220
        frames[96:, :, :, 0] = 220
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "pan.mkv"
            ffmpeg(
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                "64x36",
                "-r",
                24,
                "-i",
                "-",
                "-c:v",
                "ffv1",
                path,
                data=frames.tobytes(),
            )
            result = decode.scan_video(path, 24, 6, threading.Event())
        self.assertFalse([at for at in result["cuts"] if at < 3.9], result)
        self.assertTrue(any(abs(at - 4) < 0.1 for at in result["cuts"]), result)
        self.assertEqual(result["scoring"]["mode"], "mc")


class StorageTests(unittest.IsolatedAsyncioTestCase):
    async def test_space_check_uses_nearest_existing_destination_ancestor(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve()
            with patch.object(
                base.bili.shutil, "disk_usage", return_value=types.SimpleNamespace(free=0)
            ) as usage:
                with self.assertRaises(base.bili.DiskSpaceError):
                    base.bili.ensure_space(root / "not-created" / "media", 1)
                self.assertEqual(usage.call_args.args[0], root)

    async def test_disk_failure_is_not_retried_against_other_cdn_nodes(self):
        client = base.bili.BiliClient({})
        client._get = AsyncMock(side_effect=base.bili.DiskSpaceError("full"))
        with self.assertRaises(base.bili.DiskSpaceError):
            await client._stream(
                {
                    "baseUrl": "https://a.bilivideo.com/v",
                    "backupUrl": ["https://b.bilivideo.com/v"],
                },
                100,
                Path("video.mp4"),
            )
        client._get.assert_awaited_once()

    async def test_asr_low_space_prevents_decode_and_provider_calls(self):
        provider = types.SimpleNamespace(get_text=AsyncMock())
        with (
            tempfile.TemporaryDirectory() as folder,
            patch.object(base.engine, "ensure_space", side_effect=base.bili.DiskSpaceError("full")),
            patch.object(asyncio, "create_subprocess_exec", new_callable=AsyncMock) as spawn,
        ):
            with self.assertRaises(base.bili.DiskSpaceError):
                await base.engine.transcribe(provider, "unused.mp4", folder, 30)
            provider.get_text.assert_not_called()
            spawn.assert_not_called()
            self.assertFalse(list(Path(folder).iterdir()))

    async def test_download_space_failure_removes_partial_file_and_skips_writing(self):
        class Response:
            status = 200
            content_length = 100

            def raise_for_status(self):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

        class Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            def get(self, *args, **kwargs):
                return Response()

        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "clip.mp4"
            target.with_suffix(".part").write_bytes(b"old partial")
            with (
                patch.object(base.bili.aiohttp, "ClientSession", return_value=Client()),
                patch.object(
                    base.bili, "ensure_space", side_effect=base.bili.DiskSpaceError("full")
                ),
            ):
                with self.assertRaises(base.bili.DiskSpaceError):
                    await base.bili.BiliClient({})._get("https://example.com/v", 1000, target)
            self.assertFalse(target.exists())
            self.assertFalse(target.with_suffix(".part").exists())

    async def test_unknown_length_download_rechecks_space_and_preserves_original_target(self):
        class Response:
            status = 200
            content_length = None

            def raise_for_status(self):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def iter_chunked(self, size):
                for _ in range(20):
                    yield b"x" * 65536

            @property
            def content(self):
                return self

        class Client:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            def get(self, *args, **kwargs):
                return Response()

        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "clip.mp4"
            target.write_bytes(b"original")
            with (
                patch.object(base.bili.aiohttp, "ClientSession", return_value=Client()),
                patch.object(
                    base.bili, "ensure_space", side_effect=[None, base.bili.DiskSpaceError("full")]
                ) as space,
            ):
                with self.assertRaises(base.bili.DiskSpaceError):
                    await base.bili.BiliClient({})._get(
                        "https://example.com/v", 2 * 1048576, target
                    )
                self.assertEqual(space.call_count, 2)
            self.assertEqual(target.read_bytes(), b"original")
            self.assertFalse(target.with_suffix(".part").exists())
