"""Contribute evidence without taking ownership of another plugin's conversation."""

COMPANION = "astrbot_plugin_private_companion"


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
