"""Contribute evidence without taking ownership of another plugin's conversation."""

import json

COMPANION = "astrbot_plugin_private_companion"
COMPANION_CONTEXT_START = "<companion_v2_context>\n"
COMPANION_CONTEXT_END = "\n</companion_v2_context>"


def stopped(event):
    check = getattr(event, "is_stopped", None)
    return bool(check()) if callable(check) else False


def companion_active(context, event):
    enabled = getattr(event, "plugins_name", None)
    if enabled and "*" not in enabled and COMPANION not in enabled:
        return False
    handlers = event.get_extra("activated_handlers")
    if isinstance(handlers, list):
        return any(
            COMPANION in str(getattr(handler, "handler_module_path", "")).split(".")
            for handler in handlers
        )
    getter = getattr(context, "get_registered_star", None)
    metadata = getter(COMPANION) if callable(getter) else None
    return bool(metadata and getattr(metadata, "activated", True))


def wake_native_chat(context, config, event):
    """Only wake AstrBot's single default pipeline; never create a ProviderRequest."""
    if (
        config.get("chat_trigger_mode", "auto") != "auto"
        or companion_active(context, event)
        or stopped(event)
        or getattr(event, "_has_send_oper", False)
        or getattr(event, "call_llm", False)
    ):
        return
    event.is_at_or_wake_command = True
    event.is_wake = True


def note_companion_video(parts, source):
    """Report current external video evidence in the v7 request's perception block."""
    if source is None:
        return
    for part in parts:
        text = part.get("text") if isinstance(part, dict) else getattr(part, "text", None)
        if not isinstance(text, str) or not text.startswith(COMPANION_CONTEXT_START):
            continue
        if not text.endswith(COMPANION_CONTEXT_END):
            continue
        try:
            details = json.loads(text[len(COMPANION_CONTEXT_START) : -len(COMPANION_CONTEXT_END)])
            contents = details["environment"]["request_contents"]
        except (ValueError, KeyError, TypeError):
            return
        if not isinstance(contents, dict):
            return
        only_placeholder = contents.get("video") in ("absent", "placeholder_only")
        if contents.get("video_storyboard") == source and not only_placeholder:
            return
        if only_placeholder:
            contents["video"] = "external_" + source
        contents["video_storyboard"] = source
        updated = (
            COMPANION_CONTEXT_START
            + json.dumps(details, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            + COMPANION_CONTEXT_END
        )
        if isinstance(part, dict):
            part["text"] = updated
        else:
            part.text = updated
        return
