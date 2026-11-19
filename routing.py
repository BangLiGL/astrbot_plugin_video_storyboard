"""Per-message ownership between media parsing and video conversation."""

import json
import re
import unicodedata

from .bilibili import extract_refs, normalize_share_links

# Match share syntax independently of the URLs the analyzer can download.
BILI_SHARE_TARGET = (
    r"(?<![A-Za-z0-9._@/?=&%:#-])"
    r"(?:(?:https?://)?(?:b23\.tv|bili2233\.cn|(?:[a-z0-9-]+\.)*bilibili\.com)/|bilibili://)"
    r"[A-Za-z0-9:/?#\[\]@!$&'()*+,;=._%~\-]+"
)
BILI_SHARE_URL = re.compile(
    r"(?:【(?:[^【】\r\n]|【[^【】\r\n]*】){1,300}】\s*"
    r"|\[[^\]\r\n]{1,300}\]\(\s*<?)?" + BILI_SHARE_TARGET,
    re.I,
)
BILI_VIDEO_CODE = re.compile(
    r"(?<![A-Za-z0-9])(?:(?:bm)?BV[0-9A-Za-z]{10}|av[0-9]{6,})"
    r"(?:[ \t]+[0-9]{1,3})?(?![A-Za-z0-9])",
    re.I,
)


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
    """Ignore Bilibili-generated share text alongside the link it labels."""

    remaining = BILI_SHARE_URL.sub(" ", text)
    remaining = BILI_VIDEO_CODE.sub(" ", remaining)
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


def decide(event, parser_enabled):
    chain = event.get_messages()
    plain = "\n".join(
        str(getattr(component, "text", "")) for component in chain if kind(component) == "plain"
    )
    # Some adapters expose only message_str. Never treat card JSON as a comment.
    if not plain and not any(kind(component) in {"json", "reply", "video"} for component in chain):
        plain = event.get_message_str()
    plain = normalize_share_links(plain)
    direct = extract_refs(plain)
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
        return {"refs": (direct + quoted)[:2], "passive": False, "takeover": parser_enabled}
    if direct and has_comment(plain):
        return {"refs": direct, "passive": False, "takeover": parser_enabled}
    shared = normalize_share_links(component_text(chain) or plain)
    passive = bool(direct or BILI_SHARE_URL.search(shared) or BILI_VIDEO_CODE.search(shared))
    return {"refs": [], "passive": passive, "takeover": False}


def skip_parser(event):
    """Mutate this event's queue, leaving global registration and other handlers intact."""
    handlers = event.get_extra("activated_handlers")
    if isinstance(handlers, list):
        handlers[:] = [handler for handler in handlers if not parser_handler(handler)]
