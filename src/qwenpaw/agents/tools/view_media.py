# -*- coding: utf-8 -*-
"""Load image or video files into the LLM context for analysis."""

import asyncio
import logging
import mimetypes
import os
import unicodedata
import urllib.parse
from urllib.parse import unquote
from pathlib import Path
from typing import Optional

from agentscope.message import DataBlock, TextBlock, URLSource
from agentscope.tool import ToolChunk
from agentscope.message import ToolResultState

from ...runtime.tool_registry import tool_descriptor
from .file_io import _path_to_file_url, _resolve_file_path

logger = logging.getLogger(__name__)


def _media_data_block(url: str, modality: str) -> DataBlock:
    """Build a DataBlock from a URL, inferring ``media_type`` from the path.

    Mirrors the behaviour of the deleted ``_compat.message.ImageBlock`` /
    ``VideoBlock`` shim: when ``mimetypes.guess_type`` can't decide we
    fall back to a wildcard like ``image/*`` so the formatter still
    routes the block as the right modality.
    """
    media_type, _ = mimetypes.guess_type(url)
    if not media_type:
        media_type = f"{modality}/*"
    return DataBlock(source=URLSource(url=url, media_type=media_type))


_IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".bmp",
    ".tiff",
    ".tif",
}

_VIDEO_EXTENSIONS = {
    ".mp4",
    ".webm",
    ".mpeg",
    ".mov",
    ".avi",
    ".mkv",
}


def _is_url(path: str) -> bool:
    """Return True if *path* looks like an HTTP(S) URL."""
    return path.startswith(("http://", "https://"))


def _validate_url_extension(
    url: str,
    allowed_extensions: set[str],
    mime_prefix: str,
) -> Optional[ToolChunk]:
    """Optionally validate that the URL path has an allowed extension.

    Returns an error ``ToolChunk`` when the extension is clearly
    unsupported, or ``None`` to let it through (including when the URL
    has no recognisable extension, e.g. dynamic endpoints).
    """
    url_path = urllib.parse.urlparse(url).path
    ext = Path(url_path).suffix.lower()
    if not ext:
        return None
    mime, _ = mimetypes.guess_type(url_path)
    if ext not in allowed_extensions and (
        not mime or not mime.startswith(f"{mime_prefix}/")
    ):
        return ToolChunk(
            is_last=True,
            state=ToolResultState.SUCCESS,
            content=[
                TextBlock(
                    type="text",
                    text=f"Error: URL does not point to a "
                    f"supported {mime_prefix} format: {url}",
                ),
            ],
        )
    return None


def _validate_media_path(
    file_path: str,
    allowed_extensions: set[str],
    mime_prefix: str,
) -> tuple[Path, Optional[ToolChunk]]:
    """Validate a local media file path.

    Returns ``(resolved_path, None)`` on success or
    ``(_, error_response)`` on failure.
    """
    file_path = unquote(file_path)
    file_path = unicodedata.normalize(
        "NFC",
        os.path.expanduser(file_path),
    )
    resolved = Path(_resolve_file_path(file_path))

    if not resolved.exists() or not resolved.is_file():
        return resolved, ToolChunk(
            is_last=True,
            state=ToolResultState.SUCCESS,
            content=[
                TextBlock(
                    type="text",
                    text=f"Error: {file_path} does not exist "
                    "or is not a file.",
                ),
            ],
        )

    ext = resolved.suffix.lower()
    mime, _ = mimetypes.guess_type(str(resolved))
    if ext not in allowed_extensions and (
        not mime or not mime.startswith(f"{mime_prefix}/")
    ):
        return resolved, ToolChunk(
            is_last=True,
            state=ToolResultState.SUCCESS,
            content=[
                TextBlock(
                    type="text",
                    text=f"Error: {resolved.name} is not a "
                    f"supported {mime_prefix} format.",
                ),
            ],
        )

    return resolved, None


async def _probe_multimodal_if_needed(
    media_type: str = "image",
) -> bool | None:
    """Trigger a multimodal probe if capability is unknown (None).

    For ``image``: runs an image-only probe (~3s) and fires the full
    probe (image + video) as a background task so video support is
    persisted without blocking the caller.

    For ``video``: runs the full probe and waits for the video result,
    since video support cannot be inferred from the image probe alone.

    Uses the same agent-specific model resolution as
    ``_get_active_model_info`` so that per-agent model overrides are
    respected.

    Returns the probe result (True/False) for the requested media type,
    or None if no probe was needed or the probe failed.
    """
    try:
        from ..prompt import _get_active_model_info
        from ...providers.provider_manager import ProviderManager

        model_info, _ = _get_active_model_info()
        if model_info is None or model_info.supports_multimodal is not None:
            return None

        # Resolve agent-specific active model (mirrors _get_active_model_info)
        manager = ProviderManager.get_instance()
        active = None
        try:
            from ...app.agent_context import get_current_agent_id
            from ...config.config import load_agent_config

            agent_id = get_current_agent_id()
            agent_config = load_agent_config(agent_id)
            if agent_config.active_model:
                active = agent_config.active_model
        except Exception:
            pass
        if not active:
            active = manager.get_active_model()
        if not active:
            return None

        if media_type == "image":
            logger.info(
                "Multimodal capability unknown for %s/%s — "
                "running image-only probe...",
                active.provider_id,
                active.model,
            )
            result = await manager.probe_model_multimodal(
                active.provider_id,
                active.model,
                image_only=True,
            )
            supports = result.get("supports_image", False)
            logger.info(
                "Image probe completed for %s/%s: supports_image=%s",
                active.provider_id,
                active.model,
                supports,
            )
            # Fire full probe in background to persist video support too
            asyncio.create_task(
                manager.probe_model_multimodal(
                    active.provider_id,
                    active.model,
                ),
            )
        else:
            # video: must run full probe to get video result
            logger.info(
                "Multimodal capability unknown for %s/%s — "
                "running full probe for video support...",
                active.provider_id,
                active.model,
            )
            result = await manager.probe_model_multimodal(
                active.provider_id,
                active.model,
            )
            supports = result.get("supports_video", False)
            logger.info(
                "Full probe completed for %s/%s: supports_video=%s",
                active.provider_id,
                active.model,
                supports,
            )
        return supports
    except Exception as e:
        logger.warning("Auto-probe in view_media failed: %s", e)
        return None


def _check_multimodal_support(media_type: str = "image") -> bool:
    """Check whether the active model supports the given media type (sync).

    For ``image``: returns True when supports_image or supports_multimodal
    is explicitly True.
    For ``video``: returns True only when supports_video is explicitly True.

    Returns False for unknown (None) or explicitly unsupported (False).
    The tool is still *registered*; the async probe path handles the
    probe-on-demand logic.
    """
    try:
        from ..prompt import _get_active_model_info

        model_info, _ = _get_active_model_info()
        if model_info is None:
            return True
        if media_type == "video":
            return model_info.supports_video is True
        # image: True if supports_image or the combined supports_multimodal
        return (
            model_info.supports_image is True
            or model_info.supports_multimodal is True
        )
    except Exception:
        return True


def _get_multimodal_fallback_hint(media_type: str, path: str) -> str:
    """Build a text hint for the model when multimodal is not available.

    The actual media block is still included in the response so the
    frontend/user can see it; the hint tells the agent it cannot perceive
    the media itself.
    """
    try:
        from ..prompt import get_active_model_multimodal_raw

        raw = get_active_model_multimodal_raw()
    except Exception:
        raw = None

    if raw is None:
        logger.warning(
            "view_%s was called but multimodal capability has not been "
            "confirmed for the active model. The %s at '%s' will be "
            "shown to the user but the model cannot see it. "
            "To fix, set supports_multimodal=true in provider settings.",
            media_type,
            media_type,
            path,
        )
        return (
            f"[Note: this model does not appear to support multimodal "
            f"input — no multimodal capability was detected. You cannot "
            f"see this {media_type}, but it has been shown to the user. "
            f"Inform the user that you cannot analyze the {media_type} "
            f"content. If they believe this model supports vision, they "
            f"can override this in provider settings by setting "
            f"`supports_multimodal: true`, then retry.]"
        )

    logger.warning(
        "view_%s was called but the active model explicitly does not "
        "support multimodal input. The %s at '%s' will be shown to "
        "the user but the model cannot see it.",
        media_type,
        media_type,
        path,
    )
    return (
        f"[Note: the current model does not support multimodal input — "
        f"you cannot see this {media_type}, but it has been shown to "
        f"the user. Inform the user that you cannot analyze the "
        f"{media_type} content. If they believe this model actually "
        f"supports vision, they can override `supports_multimodal: true` "
        f"in the provider settings, or switch to a vision-capable model.]"
    )


async def _analyze_with_vision_model(
    media_block: DataBlock,
    media_type: str,
) -> str | None:
    """Call the auxiliary vision model to analyze an image/video.

    Returns a text description of the media, or None if the call fails
    or no auxiliary vision model is configured.
    """
    try:
        from agentscope.message import Msg as ASCOPE_Msg

        from ...providers.provider_manager import ProviderManager

        manager = ProviderManager.get_instance()
        slot = manager.get_auxiliary_vision_model()
        if slot is None:
            return None

        provider = manager.get_provider(slot.provider_id)
        if provider is None:
            logger.warning(
                "Auxiliary vision provider '%s' not found", slot.provider_id
            )
            return None
        if not provider.has_model(slot.model):
            logger.warning(
                "Auxiliary vision model '%s' not found in provider '%s'",
                slot.model,
                slot.provider_id,
            )
            return None

        model = provider.get_chat_model_instance(slot.model)

        prompt = (
            f"Describe this {media_type} in detail so a text-only "
            f"assistant can understand its content. Include objects, "
            f"text, colors, spatial relationships, and any other "
            f"relevant details."
        )
        msg = ASCOPE_Msg(
            name="user",
            role="user",
            content=[
                TextBlock(type="text", text=prompt),
                media_block,
            ],
        )

        logger.info(
            "Calling auxiliary vision model %s/%s for %s analysis",
            slot.provider_id,
            slot.model,
            media_type,
        )

        response = await asyncio.wait_for(
            model([msg]),
            timeout=60.0,
        )

        # Handle both streaming (async generator) and non-streaming response
        if hasattr(response, "__aiter__"):
            accumulated = ""
            async for chunk in response:
                text = _extract_text_from_response(chunk)
                if text:
                    accumulated = text
            description = accumulated
        else:
            description = _extract_text_from_response(response)

        if description:
            logger.info(
                "Auxiliary vision model returned %d-char description",
                len(description),
            )
            return description
        logger.warning("Auxiliary vision model returned empty description")
        return None
    except Exception as e:
        logger.warning(
            "Auxiliary vision model call failed: %s: %s",
            type(e).__name__,
            e,
        )
        return None


def _safe_text_attr(obj, name):
    """Safely get text attribute from response objects."""
    if isinstance(obj, dict):
        return obj.get(name)
    try:
        return getattr(obj, name, None)
    except (AttributeError, KeyError, TypeError):
        return None


def _extract_text_from_response(response) -> str:
    """Pull text out of a ChatResponse-like object or stream chunk."""
    if response is None:
        return ""
    if isinstance(response, str):
        return response
    text = _safe_text_attr(response, "text")
    if isinstance(text, str) and text:
        return text
    content = _safe_text_attr(response, "content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                return item["text"]
            inner = _safe_text_attr(item, "text")
            if isinstance(inner, str):
                return inner
    return ""


@tool_descriptor(requires_sandbox=("file_read",), async_execution=True)
async def view_image(image_path: str) -> ToolChunk:
    """Load an image file into the LLM context so the model can see it.

    Use this after desktop_screenshot, browser_use, or any tool that
    produces an image file path.  Also accepts an HTTP(S) URL for
    online images — the URL is passed directly to the model without
    downloading.

    When the model does not support multimodal, the image is still
    returned (so the user/frontend can see it). If an auxiliary vision
    model is configured, the image is sent to it for analysis and the
    description is included. Otherwise a text hint tells the agent it
    cannot perceive the image.

    Args:
        image_path (`str`):
            Local path or HTTP(S) URL of the image to view.

    Returns:
        `ToolChunk`:
            An ImageBlock the model can inspect, or an error message.
    """
    # Determine whether we need a fallback hint
    fallback_hint: str | None = None
    vision_description: str | None = None
    if not _check_multimodal_support("image"):
        probe_result = await _probe_multimodal_if_needed("image")
        if probe_result is not True:
            # Try auxiliary vision model before using static fallback hint
            if _is_url(image_path):
                aux_media_block = _media_data_block(image_path, "image")
            else:
                resolved_tmp, _ = _validate_media_path(
                    image_path,
                    _IMAGE_EXTENSIONS,
                    "image",
                )
                if resolved_tmp and resolved_tmp.exists():
                    tmp_url = _path_to_file_url(str(resolved_tmp))
                    aux_media_block = _media_data_block(tmp_url, "image")
                else:
                    aux_media_block = None

            if aux_media_block is not None:
                vision_description = await _analyze_with_vision_model(
                    aux_media_block,
                    "image",
                )

            if vision_description is None:
                fallback_hint = _get_multimodal_fallback_hint(
                    "image", image_path
                )

    if _is_url(image_path):
        err = _validate_url_extension(
            image_path,
            _IMAGE_EXTENSIONS,
            "image",
        )
        if err is not None:
            return err

        content_blocks = [_media_data_block(image_path, "image")]
        if vision_description:
            content_blocks.append(
                TextBlock(
                    type="text",
                    text=(
                        f"[Vision model analysis of {image_path}]:\n"
                        f"{vision_description}"
                    ),
                )
            )
        else:
            content_blocks.append(
                TextBlock(
                    type="text",
                    text=(
                        fallback_hint
                        if fallback_hint
                        else f"Image loaded from URL: {image_path}"
                    ),
                )
            )
        return ToolChunk(
            is_last=True,
            state=ToolResultState.SUCCESS,
            content=content_blocks,
        )

    resolved, err = _validate_media_path(
        image_path,
        _IMAGE_EXTENSIONS,
        "image",
    )
    if err is not None:
        return err

    file_url = _path_to_file_url(str(resolved))

    content_blocks = [_media_data_block(file_url, "image")]
    if vision_description:
        content_blocks.append(
            TextBlock(
                type="text",
                text=(
                    f"[Vision model analysis of {resolved.name}]:\n"
                    f"{vision_description}"
                ),
            )
        )
    else:
        content_blocks.append(
            TextBlock(
                type="text",
                text=(
                    fallback_hint
                    if fallback_hint
                    else f"Image loaded: {resolved.name}"
                ),
            )
        )
    return ToolChunk(
        is_last=True,
        state=ToolResultState.SUCCESS,
        content=content_blocks,
    )


@tool_descriptor(requires_sandbox=("file_read",), async_execution=True)
async def view_video(video_path: str) -> ToolChunk:
    """Load a video file into the LLM context so the model can see it.

    Use this when the user asks about a video file or when another
    tool produces a video file path.  Also accepts an HTTP(S) URL —
    the URL is passed directly to the model without downloading.

    When the model does not support multimodal, the video is still
    returned (so the user/frontend can see it). If an auxiliary vision
    model is configured, the video is sent to it for analysis and the
    description is included. Otherwise a text hint tells the agent it
    cannot perceive the video.

    Args:
        video_path (`str`):
            Local path or HTTP(S) URL of the video to view.

    Returns:
        `ToolChunk`:
            A VideoBlock the model can inspect, or an error message.
    """
    fallback_hint: str | None = None
    vision_description: str | None = None
    if not _check_multimodal_support("video"):
        probe_result = await _probe_multimodal_if_needed("video")
        if probe_result is not True:
            if _is_url(video_path):
                aux_media_block = _media_data_block(video_path, "video")
            else:
                resolved_tmp, _ = _validate_media_path(
                    video_path,
                    _VIDEO_EXTENSIONS,
                    "video",
                )
                if resolved_tmp and resolved_tmp.exists():
                    tmp_url = _path_to_file_url(str(resolved_tmp))
                    aux_media_block = _media_data_block(tmp_url, "video")
                else:
                    aux_media_block = None

            if aux_media_block is not None:
                vision_description = await _analyze_with_vision_model(
                    aux_media_block,
                    "video",
                )

            if vision_description is None:
                fallback_hint = _get_multimodal_fallback_hint(
                    "video", video_path
                )

    if _is_url(video_path):
        err = _validate_url_extension(
            video_path,
            _VIDEO_EXTENSIONS,
            "video",
        )
        if err is not None:
            return err

        content_blocks = [_media_data_block(video_path, "video")]
        if vision_description:
            content_blocks.append(
                TextBlock(
                    type="text",
                    text=(
                        f"[Vision model analysis of {video_path}]:\n"
                        f"{vision_description}"
                    ),
                )
            )
        else:
            content_blocks.append(
                TextBlock(
                    type="text",
                    text=(
                        fallback_hint
                        if fallback_hint
                        else f"Video loaded from URL: {video_path}"
                    ),
                )
            )
        return ToolChunk(
            is_last=True,
            state=ToolResultState.SUCCESS,
            content=content_blocks,
        )

    resolved, err = _validate_media_path(
        video_path,
        _VIDEO_EXTENSIONS,
        "video",
    )
    if err is not None:
        return err

    file_url = _path_to_file_url(str(resolved))

    content_blocks = [_media_data_block(file_url, "video")]
    if vision_description:
        content_blocks.append(
            TextBlock(
                type="text",
                text=(
                    f"[Vision model analysis of {resolved.name}]:\n"
                    f"{vision_description}"
                ),
            )
        )
    else:
        content_blocks.append(
            TextBlock(
                type="text",
                text=(
                    fallback_hint
                    if fallback_hint
                    else f"Video loaded: {resolved.name}"
                ),
            )
        )
    return ToolChunk(
        is_last=True,
        state=ToolResultState.SUCCESS,
        content=content_blocks,
    )
