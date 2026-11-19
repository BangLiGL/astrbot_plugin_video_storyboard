"""Keep terminal-style writing drafts out of video conversation replies."""

import re
from itertools import groupby

from .routing import kind

SHELL_PROMPT = re.compile(r"(?m)^[ \t]*[\w.-]+@[\w.-]+:[^\r\n$#]*[$#][ \t]+[^\r\n]+")
OPEN_FENCE = re.compile(r"^\x60{3,}[ \t]*(?:bash|sh|shell|zsh|console|text)?[ \t]*\r?\n", re.I)
CLOSE_FENCE = re.compile(r"(?m)^[ \t]*\x60{3,}[ \t]*\r?$")
READ_CHAPTER = re.compile(r"\b(?:cat|type)\s+(?:[^\s]*[/\\])?chapter[_-]\d+\.txt\b", re.I)


def visible_reply(text):
    """Identify the writing-draft format without stripping ordinary code samples."""
    body = text.lstrip()
    fence = OPEN_FENCE.match(body)
    shell = body[fence.end() :] if fence else body
    first = SHELL_PROMPT.match(shell)
    if first is None:
        return text
    commands = list(SHELL_PROMPT.finditer(shell))
    if len(commands) < 2:
        return text

    last = commands[-1]
    draft = shell[: last.end()].lower()
    chapter = READ_CHAPTER.search(last.group())
    has_metadata = "[character_analysis]" in draft and "[trajectory_analysis]" in draft
    if not has_metadata and not ("writing_guidelines.md" in draft and chapter):
        return text

    if fence:
        closing = CLOSE_FENCE.search(shell, last.end())
        return shell[closing.end() :].strip() if closing else ""
    return shell[last.end() :].strip() if chapter else ""


def clean_reply_chain(owner):
    chain = getattr(owner, "chain", None)
    if not chain:
        return
    cleaned = []
    for plain, group in groupby(chain, key=lambda part: kind(part) == "plain"):
        parts = list(group)
        if plain:
            original = "".join(part.text for part in parts)
            visible = visible_reply(original)
            if visible != original:
                if visible:
                    parts[0].text = visible
                    cleaned.append(parts[0])
                continue
        cleaned.extend(parts)
    chain[:] = cleaned
