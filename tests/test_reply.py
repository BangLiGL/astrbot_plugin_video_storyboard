"""Regression coverage for the terminal transcript in video chat output."""

import importlib
import types
import unittest

import test_chat as base

reply = importlib.import_module(base.PACKAGE + ".reply")
FENCE = chr(96) * 3
TERMINAL = (
    "user@linux:~$ cd Documents\n"
    "user@linux:~/Documents$ cat writing_guidelines.md\n"
    "这里是写作规则。\n"
    "user@linux:~/Documents$ file --analyze-characters chapter_13.txt\n"
    "├── [character_analysis]: 草稿\n"
    "└── [trajectory_analysis]: 草稿\n"
    "user@linux:~/Documents$ cat chapter_13.txt\n"
)
DRAFT = f"{FENCE}bash\n{TERMINAL}{FENCE}"
CHAT = "谁给这只鹦鹉批的场地啊喵\n\n不过挺好听的。"


def plain(text):
    return types.SimpleNamespace(type="Plain", text=text)


class ReplyTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.plugin = object.__new__(base.main.VideoStoryboardPlugin)

    def test_keeps_chat_after_terminal_draft_but_drops_draft_without_reply(self):
        self.assertEqual(reply.visible_reply(f"{DRAFT}\n\n{CHAT}"), CHAT)
        self.assertEqual(reply.visible_reply(TERMINAL + CHAT), CHAT)
        self.assertEqual(reply.visible_reply(DRAFT), "")
        self.assertEqual(reply.visible_reply(f"{FENCE}bash\n{TERMINAL}"), "")

    def test_preserves_shell_examples_and_character_dialogue(self):
        commands = f"{FENCE}bash\nuser@linux:~$ cd Documents\nuser@linux:~/Documents$ ls\n{FENCE}"
        for text in (commands, CHAT, f"这是引用的写作草稿示例：\n{DRAFT}"):
            self.assertEqual(reply.visible_reply(text), text)

    async def test_guard_changes_only_the_video_turn_assistant_reply(self):
        event = base.Event("分享视频")
        original = f"{DRAFT}\n\n{CHAT}"
        response = types.SimpleNamespace(role="assistant", completion_text=original)
        await self.plugin.clean_video_completion(event, response)
        self.assertEqual(response.completion_text, original)
        self.plugin.guard_video_turn(event)
        self.assertIs(event.get_extra("enable_streaming"), False)
        await self.plugin.clean_video_completion(event, response)
        self.assertEqual(response.completion_text, CHAT)

    async def test_delivery_guard_handles_split_text_and_leaves_parser_results(self):
        event = base.Event("视频分享")
        self.plugin.guard_video_turn(event)
        image = types.SimpleNamespace(type="Image")
        chain = [plain(DRAFT[:20]), plain(DRAFT[20:] + "\n\n" + CHAT), image]
        result = types.SimpleNamespace(chain=chain, is_llm_result=lambda: False)
        event.get_result = lambda: result
        event.set_extra("_llm_reasoning_content", DRAFT)
        await self.plugin.clean_video_delivery(event)
        self.assertEqual(len(chain), 3)
        self.assertEqual(event.get_extra("_llm_reasoning_content"), DRAFT)
        result.is_llm_result = lambda: True
        await self.plugin.clean_video_delivery(event)
        self.assertEqual(chain[0].text, CHAT)
        self.assertIs(chain[1], image)
        self.assertIsNone(event.get_extra("_llm_reasoning_content"))
