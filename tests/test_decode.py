"""Real-media regression tests for bounded decoding and parallel text evidence."""

import asyncio
import importlib
import json
import subprocess
import tempfile
import threading
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import imageio_ffmpeg
import numpy as np
import test_chat as base

decode = importlib.import_module(base.PACKAGE + ".decode")
tracks = importlib.import_module(base.PACKAGE + ".tracks")
sampling = importlib.import_module(base.PACKAGE + ".sampling")


def ffmpeg(*args, data=None):
    return subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-v", "error", "-y", *map(str, args)],
        input=data,
        capture_output=True,
        check=True,
        timeout=20,
    )


class ScanTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)

    def encode(self, frames, fps=24):
        path = Path(self.folder.name) / "scan.mkv"
        ffmpeg(
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            "64x36",
            "-r",
            fps,
            "-i",
            "-",
            "-c:v",
            "ffv1",
            path,
            data=np.asarray(frames, np.uint8).tobytes(),
        )
        return decode.scan_video(path, fps, len(frames) / fps, threading.Event())

    def test_dark_letterboxed_hard_cut_is_detected_without_brightening_images(self):
        frames = np.zeros((96, 36, 64, 3), dtype=np.uint8)
        frames[:48, 6:30, :, 2] = 45
        frames[48:, 6:30, :, 0] = 45
        result = self.encode(frames)
        self.assertAlmostEqual(result["bars"], 1 / 3, delta=0.03)
        self.assertTrue(any(abs(t - 2) < 0.1 for t in result["cuts"]), result)
        self.assertEqual(result["frames"], 96)

    def test_slow_dissolve_is_detected_but_constant_pan_is_not_repeated_edits(self):
        frames = np.zeros((144, 36, 64, 3), dtype=np.uint8)
        frames[:48, :, :, 2] = 200
        for i in range(24):
            frames[48 + i, :, :, 0] = round(200 * i / 23)
            frames[48 + i, :, :, 2] = round(200 * (1 - i / 23))
        frames[72:, :, :, 0] = 200
        result = self.encode(frames)
        self.assertTrue(any(1.8 < t < 3.2 for t in result["cuts"]), result)
        scores = [(i / 24, 0.08) for i in range(1, 144)]
        self.assertEqual(sampling.scene_cuts(scores, 6), [])

    def test_fast_irregular_hard_cuts_survive_while_regular_two_frame_motion_does_not(self):
        frames = np.zeros((96, 36, 64, 3), dtype=np.uint8)
        marks = [0, 25, 33, 43, 50, 72, 96]
        for i, (a, b) in enumerate(zip(marks, marks[1:])):
            frames[a:b, :, :, i % 3] = 220
        result = self.encode(frames)
        for frame in marks[1:-1]:
            self.assertTrue(any(abs(t - frame / 24) < 0.09 for t in result["cuts"]), result)
        changes = [(i / 24, 0.08 if i % 2 == 0 else 0) for i in range(1, 144)]
        self.assertEqual(sampling.scene_cuts(changes, 6), [])

    def test_decode_cancellation_and_output_cap_reap_children(self):
        cancel = threading.Event()
        with self.assertRaises(base.bili.VideoError):
            with decode.process(
                ["-re", "-f", "lavfi", "-i", "color=s=64x36:r=1", "-f", "rawvideo", "-"],
                cancel,
                timeout=5,
            ) as child:
                cancel.set()
        self.assertIsNotNone(child.poll())


class MediaEvidenceTests(unittest.IsolatedAsyncioTestCase):
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

    async def understand(self):
        return await self.analyzer.understand({"bvid": "BV1GJ411x7h7", "page": 1}, "g", "g")

    def evidence(self):
        return json.loads(self.context.llm_generate.call_args.kwargs["prompt"])

    async def test_extracted_frames_use_actual_pts_and_reuse_same_decoded_frame(self):
        targets = [0.14, 0.15, 3.9, 4.1, 7.7]
        frames = await asyncio.to_thread(
            decode.extract_frames, self.clip, targets, (160, 90), 16 / 9, threading.Event()
        )
        self.assertEqual(set(frames), set(targets))
        self.assertAlmostEqual(frames[0.14][0], frames[0.15][0], delta=0.001)
        for target, (actual, image) in frames.items():
            self.assertGreaterEqual(actual + 0.0001, target)
            self.assertLess(actual - target, 1 / 12 + 0.001)
            channel = 2 if actual < 4 else 0
            self.assertGreater(image[:, :, channel].mean(), 150)

    async def test_variable_frame_rate_uses_decoded_timestamps_and_correct_colors(self):
        video = Path(self.folder.name) / "vfr.mp4"
        ffmpeg(
            "-i",
            self.clip,
            "-an",
            "-vf",
            "select='if(lt(t,4),not(mod(n,4)),1)'",
            "-fps_mode",
            "vfr",
            video,
        )
        frames = await asyncio.to_thread(
            decode.extract_frames,
            video,
            [0.2, 1.1, 3.9, 4.1, 6.2],
            (160, 90),
            16 / 9,
            threading.Event(),
        )
        self.assertEqual(len(frames), 5)
        for target, (actual, image) in frames.items():
            self.assertGreaterEqual(actual + 0.0001, target)
            self.assertLess(actual - target, 1 / 3 + 0.001)
            self.assertGreater(
                image[:, :, 2 if actual < 3.999 else 0].mean(),
                150,
                (target, actual, image.mean(axis=(0, 1)).tolist()),
            )

    async def test_valid_transcript_cache_remains_usable_when_audio_download_failed(self):
        await self.understand()
        self.metadata["audio_state"] = "unavailable"
        self.config["storyboard_cells"] = 5
        result = await self.understand()
        self.assertEqual(result["transcript_status"], "available")
        self.assertEqual(result["audio_state"], "unavailable")
        self.assertEqual(self.context.stt.get_text.await_count, 1)
        self.assertTrue(self.evidence()["transcript"])

    async def test_embedded_subtitles_are_preferred_to_asr_and_cached_across_stt_changes(self):
        subtitle = Path(self.folder.name) / "words.srt"
        subtitle.write_text(
            "1\n00:00:01,250 --> 00:00:03,750\n<b>Embedded evidence</b>\n", encoding="utf-8"
        )
        video = Path(self.folder.name) / "with-subtitles.mkv"
        ffmpeg("-i", self.clip, "-i", subtitle, "-map", "0:v", "-map", "1", "-c", "copy", video)
        self.metadata["video_path"] = str(video)
        self.metadata["audio_path"] = str(video)
        before = video.read_bytes()
        result = await self.understand()
        self.assertEqual(result["transcript_source"], "embedded_subtitle")
        self.assertEqual(self.evidence()["transcript"][0]["from"], 1.25)
        self.assertEqual(self.evidence()["transcript"][0]["text"], "Embedded evidence")
        self.context.stt.get_text.assert_not_called()
        self.context.providers["other_stt"] = types.SimpleNamespace(
            provider_config={"id": "other_stt"}, get_text=self.context.stt.get_text
        )
        self.config["stt_provider_id"] = "other_stt"
        self.analyzer.client.fetch.reset_mock()
        await self.understand()
        self.analyzer.client.fetch.assert_not_called()
        self.assertEqual(video.read_bytes(), before)

    async def test_site_subtitles_keep_precedence_and_absolute_page_times(self):
        self.metadata["segments"] = [{"from": 0.5, "to": 2, "text": "Site CC"}]
        self.metadata["transcript_source"] = "bilibili_subtitle"
        self.config["storyboard_max_per_sheet"] = 1
        with patch.object(base.analysis, "embedded_subtitles") as embedded:
            await self.understand()
        embedded.assert_not_called()
        self.context.stt.get_text.assert_not_called()
        evidence = self.evidence()
        self.assertEqual(evidence["transcript"][0]["text"], "Site CC")
        sheets = evidence["storyboard"]["sheets"]
        self.assertEqual(sheets[0]["from"], 0)
        self.assertEqual(sheets[-1]["to"], 8)
        for previous, following in zip(sheets, sheets[1:]):
            self.assertEqual(previous["to"], following["from"])

    async def test_absent_audio_is_distinct_from_silent_track_and_empty_asr(self):
        no_audio = Path(self.folder.name) / "no-audio.mp4"
        ffmpeg("-i", self.clip, "-an", "-c:v", "copy", no_audio)
        silent = Path(self.folder.name) / "silent.mp4"
        ffmpeg("-i", self.clip, "-c:v", "copy", "-af", "volume=0", silent)
        self.assertEqual(await tracks.audio_status(no_audio), "absent")
        self.assertEqual(await tracks.audio_status(silent), "present")
        self.metadata.update(video_path=str(no_audio), audio_path=str(no_audio))
        result = await self.understand()
        self.assertEqual(result["audio_state"], "absent")
        self.assertEqual(result["transcript_status"], "no_audio")
        self.context.stt.get_text.assert_not_called()
        with self.analyzer.knowledge.connect() as db:
            db.execute("DELETE FROM videos")
        self.metadata.update(video_path=str(silent), audio_path=str(silent))
        self.context.stt.get_text.return_value = ""
        result = await self.understand()
        self.assertEqual(result["audio_state"], "present")
        self.assertEqual(result["transcript_status"], "empty")
        with self.analyzer.knowledge.connect() as db:
            self.assertEqual(db.execute("SELECT count(*) FROM transcripts").fetchone()[0], 0)

    async def test_text_and_visual_workers_really_overlap(self):
        # Neither worker can finish unless the other starts; serial execution fails.
        entered_visual, entered_text = threading.Event(), threading.Event()
        original = base.analysis.storyboard

        def visual(*args):
            entered_visual.set()
            if not entered_text.wait(4):
                raise AssertionError("Text pass did not start while rendering")
            return original(*args)

        async def stt(*args):
            if not await asyncio.to_thread(entered_visual.wait, 4):
                raise AssertionError("Visual pass did not start while transcribing")
            entered_text.set()
            return "Parallel speech evidence"

        self.context.stt.get_text.side_effect = stt
        with patch.object(base.analysis, "storyboard", side_effect=visual):
            result = await self.understand()
        self.assertTrue(entered_visual.is_set() and entered_text.is_set())
        self.assertTrue(result["visual"])
        self.assertEqual(self.evidence()["transcript"][0]["text"], "Parallel speech evidence")

    async def test_asr_failure_does_not_claim_no_audio_or_discard_valid_visual_evidence(self):
        self.context.stt.get_text.side_effect = RuntimeError("ASR unavailable")
        result = await self.understand()
        self.assertEqual(result["audio_state"], "present")
        self.assertEqual(result["transcript_status"], "failed")
        self.assertTrue(result["visual"])
        self.assertEqual(self.evidence()["transcript_status"], "failed")

    async def test_track_output_limit_and_timeout_terminate_ffmpeg(self):
        with self.assertRaises(base.bili.VideoError):
            await tracks.run_ffmpeg(
                ["-f", "lavfi", "-i", "color=s=64x36", "-t", "1", "-f", "rawvideo", "-"],
                limit=20,
            )
        with self.assertRaises(asyncio.TimeoutError):
            await tracks.run_ffmpeg(
                ["-re", "-f", "lavfi", "-i", "color=s=64x36:r=1", "-f", "rawvideo", "-"],
                timeout=0.1,
            )

    async def test_embedded_srt_rejects_empty_invalid_and_clips_timeline(self):
        text = (
            "1\n00:00:02,000 --> 00:00:01,000\ninvalid\n\n"
            "2\n00:00:07,000 --> 00:00:10,000\nvalid\n\n"
            "3\n00:00:09,000 --> 00:00:10,000\npast end\n"
        )
        rows = tracks.parse_srt(text, 8)
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0]["from"], rows[0]["to"]), (7, 8))
