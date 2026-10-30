"""Per-message ownership between media parsing and video conversation."""

import json
import re
import unicodedata
from urllib.parse import urlsplit

from .bilibili import BV, extract_refs

ASCII_URL = re.compile(r"https?://[A-Za-z0-9:/?#\[\]@!$&'()*+,;=._%~\-]+", re.I)


def kind(component):
    return str(getattr(component, "type", "")).lower().split(".")[-1]


def component_text(chain):
    parts = []
    for component in chain:
        if kind(component) == "plain":
            parts.append(str(getattr(component, "text", "")))
        elif kind(component) == "json":
            data = getattr(component, "data", "")
            if isinstance(data, str):
                try:
                    data = json.loads(data[:24000])
                except (ValueError, TypeError):
                    pass
            parts.append(json.dumps(data, ensure_ascii=False)[:12000])
    return "\n".join(parts)


def has_comment(text):
    """Ignore only actual Bili URL spans, identifiers, mentions and punctuation."""

    def replace_url(match):
        host = (urlsplit(match.group()).hostname or "").lower()
        if host == "b23.tv" or host == "bilibili.com" or host.endswith(".bilibili.com"):
            return " "
        return match.group()

    remaining = ASCII_URL.sub(replace_url, text)
    remaining = BV.sub(" ", remaining)
    remaining = re.sub(r"\[CQ:(?:at|reply),[^\]]*\]|<@!?\d+>", " ", remaining)
    return any(
        not char.isspace() and unicodedata.category(char)[0] not in {"P", "Z", "C"}
        for char in remaining
    )


def parser_handler(handler):
    return (
        "astrbot_plugin_parser" in str(getattr(handler, "handler_module_path", "")).split(".")
        and getattr(handler, "handler_name", "") == "on_message"
    )


def parser_available(context, event):
    handlers = event.get_extra("activated_handlers")
    if isinstance(handlers, list):
        return any(parser_handler(handler) for handler in handlers)
    getter = getattr(context, "get_registered_star", None)
    metadata = getter("astrbot_plugin_parser") if getter else None
    enabled = getattr(event, "plugins_name", None)
    return bool(
        metadata
        and getattr(metadata, "activated", True)
        and (not enabled or "*" in enabled or "astrbot_plugin_parser" in enabled)
    )


def decide(event, compatibility):
    chain = event.get_messages()
    plain = "\n".join(
        str(getattr(component, "text", "")) for component in chain if kind(component) == "plain"
    )
    # Some adapters expose only message_str. Never treat card JSON as a comment.
    if not plain and not any(kind(component) in {"json", "reply", "video"} for component in chain):
        plain = event.get_message_str()
    direct = extract_refs(plain)
    shared = extract_refs(component_text(chain))
    quoted = []
    for component in chain:
        if kind(component) != "reply":
            continue
        reply_chain = getattr(component, "chain", None) or []
        videos = [part for part in reply_chain if kind(part) == "video"]
        if videos:
            for video in videos[:2]:
                sources = list(
                    dict.fromkeys(
                        str(value)
                        for value in (
                            getattr(video, "path", ""),
                            getattr(video, "file", ""),
                            getattr(video, "url", ""),
                        )
                        if value
                    )
                )
                if sources:
                    quoted.append({"video_sources": sources})
        else:
            quoted.extend(extract_refs(component_text(reply_chain)))
            if not reply_chain:
                quoted.extend(extract_refs(getattr(component, "message_str", "") or ""))
    if quoted:
        return {"refs": (direct + quoted)[:2], "passive": False, "takeover": compatibility}
    if direct and (not compatibility or has_comment(plain)):
        return {"refs": direct, "passive": False, "takeover": compatibility}
    if compatibility and (direct or shared):
        return {"refs": [], "passive": True, "takeover": False}
    return {"refs": shared or direct, "passive": False, "takeover": False}


def skip_parser(event):
    """Mutate this event's queue, leaving global registration and other handlers intact."""
    handlers = event.get_extra("activated_handlers")
    if isinstance(handlers, list):
        handlers[:] = [handler for handler in handlers if not parser_handler(handler)]
