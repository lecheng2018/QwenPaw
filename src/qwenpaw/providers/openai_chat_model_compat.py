# -*- coding: utf-8 -*-
"""OpenAI chat model compatibility wrappers."""

from __future__ import annotations

import json
import logging
from datetime import datetime
from types import SimpleNamespace
from typing import Any, AsyncGenerator, Callable

from agentscope.model import OpenAIChatModel
from agentscope.model._model_response import ChatResponse

from qwenpaw.local_models.tag_parser import (
    parse_tool_calls_from_text,
    text_contains_tool_call_tag,
)

logger = logging.getLogger(__name__)


def _battr(block: Any, key: str, default: Any = None) -> Any:
    """Read an attribute from a dict *or* Pydantic block."""
    if isinstance(block, dict):
        return block.get(key, default)
    return getattr(block, key, default)


def _bset(block: Any, key: str, value: Any) -> None:
    """Set an attribute on a dict *or* Pydantic block."""
    if isinstance(block, dict):
        block[key] = value
    else:
        setattr(block, key, value)


def _clone_with_overrides(obj: Any, **overrides: Any) -> Any:
    """Clone a stream object into a mutable namespace with overrides."""
    data = dict(getattr(obj, "__dict__", {}))
    data.update(overrides)
    return SimpleNamespace(**data)


def _sanitize_tool_call(tool_call: Any) -> Any | None:
    """Normalize a tool call for parser safety, or drop it if unusable."""
    if not hasattr(tool_call, "index"):
        return None

    function = getattr(tool_call, "function", None)
    if function is None:
        return None

    has_name = hasattr(function, "name")
    has_arguments = hasattr(function, "arguments")

    raw_name = getattr(function, "name", "")
    if isinstance(raw_name, str):
        safe_name = raw_name
    elif raw_name is None:
        safe_name = ""
    else:
        safe_name = str(raw_name)

    raw_arguments = getattr(function, "arguments", "")
    if isinstance(raw_arguments, str):
        safe_arguments = raw_arguments
    elif raw_arguments is None:
        safe_arguments = ""
    else:
        try:
            safe_arguments = json.dumps(raw_arguments, ensure_ascii=False)
        except (TypeError, ValueError):
            safe_arguments = str(raw_arguments)

    if (
        has_name
        and has_arguments
        and isinstance(raw_name, str)
        and isinstance(
            raw_arguments,
            str,
        )
    ):
        return tool_call

    safe_function = SimpleNamespace(
        name=safe_name,
        arguments=safe_arguments,
    )
    return _clone_with_overrides(tool_call, function=safe_function)


def _sanitize_chunk(chunk: Any) -> Any:
    """Drop/normalize malformed tool-calls in a streaming chunk."""
    choices = getattr(chunk, "choices", None)
    if not choices:
        return chunk

    sanitized_choices: list[Any] = []
    changed = False

    for choice in choices:
        delta = getattr(choice, "delta", None)
        if delta is None:
            sanitized_choices.append(choice)
            continue

        raw_tool_calls = getattr(delta, "tool_calls", None)
        if not raw_tool_calls:
            sanitized_choices.append(choice)
            continue

        choice_changed = False
        sanitized_tool_calls: list[Any] = []
        for tool_call in raw_tool_calls:
            sanitized = _sanitize_tool_call(tool_call)
            if sanitized is not tool_call:
                choice_changed = True
            if sanitized is not None:
                sanitized_tool_calls.append(sanitized)

        if choice_changed:
            changed = True
            sanitized_delta = _clone_with_overrides(
                delta,
                tool_calls=sanitized_tool_calls,
            )
            sanitized_choice = _clone_with_overrides(
                choice,
                delta=sanitized_delta,
            )
            sanitized_choices.append(sanitized_choice)
            continue

        sanitized_choices.append(choice)

    if not changed:
        return chunk
    return _clone_with_overrides(chunk, choices=sanitized_choices)


def _sanitize_stream_item(item: Any) -> Any:
    """Sanitize either plain stream chunks or structured stream items."""
    if hasattr(item, "chunk"):
        chunk = item.chunk
        sanitized_chunk = _sanitize_chunk(chunk)
        if sanitized_chunk is chunk:
            return item
        return _clone_with_overrides(item, chunk=sanitized_chunk)

    return _sanitize_chunk(item)


class _SanitizedStream:
    """Proxy OpenAI async stream that sanitizes each emitted item and
    captures ``extra_content`` from tool-call chunks (used by Gemini
    thinking models to carry ``thought_signature``)."""

    def __init__(self, stream: Any):
        self._stream = stream
        self._ctx_stream: Any | None = None
        self.extra_contents: dict[str, Any] = {}

    async def __aenter__(self) -> "_SanitizedStream":
        self._ctx_stream = await self._stream.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: Any,
        exc: Any,
        tb: Any,
    ) -> bool | None:
        return await self._stream.__aexit__(exc_type, exc, tb)

    def __aiter__(self) -> "_SanitizedStream":
        return self

    async def __anext__(self) -> Any:
        if self._ctx_stream is None:
            raise StopAsyncIteration
        item = await self._ctx_stream.__anext__()
        self._capture_extra_content(item)
        return _sanitize_stream_item(item)

    def _capture_extra_content(self, item: Any) -> None:
        """Store ``extra_content`` keyed by tool-call id."""
        chunk = getattr(item, "chunk", item)
        choices = getattr(chunk, "choices", None) or []
        for choice in choices:
            delta = getattr(choice, "delta", None)
            if not delta:
                continue
            for tc in getattr(delta, "tool_calls", None) or []:
                tc_id = getattr(tc, "id", None)
                if not tc_id:
                    continue
                extra = getattr(tc, "extra_content", None)
                if extra is None:
                    model_extra = getattr(tc, "model_extra", None)
                    if isinstance(model_extra, dict):
                        extra = model_extra.get("extra_content")
                if extra:
                    self.extra_contents[tc_id] = extra


# JSON Schema keywords whose value is itself a schema.
_SINGLE_SCHEMA_KEYWORDS = frozenset(
    {
        "items",
        "additionalProperties",
        "additionalItems",
        "unevaluatedProperties",
        "unevaluatedItems",
        "contains",
        "propertyNames",
        "not",
        "if",
        "then",
        "else",
        "contentSchema",
    },
)
# Keywords whose value is an array of schemas.
_ARRAY_SCHEMA_KEYWORDS = frozenset(
    {"allOf", "anyOf", "oneOf", "prefixItems"},
)
# Keywords whose value is an object whose values are schemas.
_MAP_SCHEMA_KEYWORDS = frozenset(
    {
        "properties",
        "patternProperties",
        "$defs",
        "definitions",
        "dependentSchemas",
    },
)


def _walk_schema(
    schema: dict[str, Any],
    transform: Callable[[Any], Any],
    *,
    dependency_schema_types: tuple[type, ...] = (dict,),
) -> dict[str, Any]:
    """Walk known JSON Schema child positions using *transform*."""
    result: dict[str, Any] = {}
    for key, value in schema.items():
        if key in _SINGLE_SCHEMA_KEYWORDS:
            if key == "items" and isinstance(value, list):
                result[key] = [transform(item) for item in value]
            else:
                result[key] = transform(value)
        elif key in _ARRAY_SCHEMA_KEYWORDS:
            if isinstance(value, list):
                result[key] = [transform(item) for item in value]
            else:
                result[key] = value
        elif key in _MAP_SCHEMA_KEYWORDS:
            if isinstance(value, dict):
                result[key] = {
                    item_key: transform(item_value)
                    for item_key, item_value in value.items()
                }
            else:
                result[key] = value
        elif key == "dependencies" and isinstance(value, dict):
            # draft-07: value per key may be a schema or a string array.
            result[key] = {
                item_key: (
                    transform(item_value)
                    if isinstance(item_value, dependency_schema_types)
                    else item_value
                )
                for item_key, item_value in value.items()
            }
        else:
            result[key] = value
    return result


def _strip_boolean_schema_special_cases(schema: Any) -> Any:
    if not isinstance(schema, dict):
        return schema
    return {
        key: value
        for key, value in schema.items()
        if not (
            (key == "additionalProperties" and value is True)
            or (key == "required" and isinstance(value, bool))
        )
    }


# pylint: disable=too-many-branches
def _sanitize_boolean_schemas(schema: Any) -> Any:
    """Position-aware sanitizer for boolean JSON Schema values.

    JSON Schema uses booleans in two distinct ways:

    1. **Boolean schemas** — at a position where a schema is expected,
       ``true`` means "accept anything" and ``false`` means "reject
       everything".  Legal per spec but rejected by strict LLM providers
       (DeepSeek V4, OpenAI) that require an object schema.  We convert::

           true  → {}
           false → {"not": {}}

    2. **Boolean-valued keywords** — annotations like ``nullable``,
       ``deprecated``, ``readOnly``, ``writeOnly``, ``uniqueItems``,
       draft-04 ``exclusiveMinimum`` / ``exclusiveMaximum``.  These MUST
       remain booleans; providers validate them as ``type: boolean``.

    This walker recurses only into known schema-positions, so boolean
    annotations on ordinary keywords pass through unchanged.

    Special-cases retained:
    - ``additionalProperties: true``  → removed (JSON Schema default;
      explicit form rejected by some strict validators).
    - ``required: <bool>`` inside a property definition → removed
      (malformed; real JSON Schema uses ``required: ["field"]`` on the
      parent object).
    """
    if schema is True:
        return {}
    if schema is False:
        return {"not": {}}
    if not isinstance(schema, dict):
        return schema
    return _walk_schema(
        _strip_boolean_schema_special_cases(schema),
        _sanitize_boolean_schemas,
        dependency_schema_types=(dict, bool),
    )


def _is_null_schema(schema: Any) -> bool:
    if not isinstance(schema, dict):
        return False
    node_type = schema.get("type")
    if node_type == "null":
        return True
    return isinstance(node_type, list) and set(node_type) == {"null"}


def _ensure_object_type_for_untyped_schema(
    schema: dict[str, Any],
) -> dict[str, Any]:
    """Give a permissive nullable fallback an explicit provider type."""
    if "type" not in schema and not any(
        key in schema
        for key in (
            "$ref",
            "allOf",
            "anyOf",
            "oneOf",
            "not",
            "enum",
            "const",
        )
    ):
        return {**schema, "type": "object"}
    return schema


def _normalize_nullable_type(schema: dict[str, Any]) -> dict[str, Any]:
    node_type = schema.get("type")
    if node_type == "null":
        return {**schema, "type": "object"}
    if not isinstance(node_type, list) or "null" not in node_type:
        return schema

    non_null_types = [value for value in node_type if value != "null"]
    if not non_null_types:
        return {**schema, "type": "object"}
    if len(non_null_types) == 1:
        return {**schema, "type": non_null_types[0]}
    return {**schema, "type": non_null_types}


# pylint: disable=too-many-branches
def _sanitize_nullable_schemas(schema: Any) -> Any:
    """Remove JSON Schema ``null`` types from provider tool schemas.

    OpenAI-compatible relays that route to Gemini-style backends often reject
    functionDeclaration schemas containing nullable branches such as
    ``anyOf: [{"type": "string"}, {"type": "null"}]``.  Tool parameters are
    still optional via ``required`` and ``default: null``, so the provider
    schema can safely expose only the non-null branch.
    """
    if isinstance(schema, list):
        return [_sanitize_nullable_schemas(item) for item in schema]
    if not isinstance(schema, dict):
        return schema

    node = dict(schema)

    for key in ("anyOf", "oneOf"):
        variants = node.get(key)
        if not isinstance(variants, list):
            continue

        non_null_variants = [
            variant for variant in variants if not _is_null_schema(variant)
        ]
        if len(non_null_variants) == len(variants):
            continue

        if len(non_null_variants) == 1:
            branch = _sanitize_nullable_schemas(non_null_variants[0])
            if not isinstance(branch, dict):
                branch = {}
            siblings = {
                sibling_key: sibling_value
                for sibling_key, sibling_value in node.items()
                if sibling_key != key
            }
            sanitized_siblings = _sanitize_nullable_schemas(siblings)
            merged = dict(branch)
            if isinstance(sanitized_siblings, dict):
                for sibling_key, sibling_value in sanitized_siblings.items():
                    merged.setdefault(sibling_key, sibling_value)
            return _ensure_object_type_for_untyped_schema(
                _normalize_nullable_type(merged),
            )

        if non_null_variants:
            node[key] = non_null_variants
        else:
            node.pop(key, None)
            node = _ensure_object_type_for_untyped_schema(node)

    return _normalize_nullable_type(
        _walk_schema(node, _sanitize_nullable_schemas),
    )


def _collect_defs(schema: dict[str, Any]) -> dict[str, Any]:
    """Collect all named type definitions from a JSON Schema root.

    Supports both ``$defs`` (JSON Schema draft-2019+) and the legacy
    ``definitions`` keyword (draft-04/06/07).
    """
    defs: dict[str, Any] = {}
    if isinstance(schema.get("$defs"), dict):
        defs.update(schema["$defs"])
    if isinstance(schema.get("definitions"), dict):
        defs.update(schema["definitions"])
    return defs


def _resolve_local_ref(
    ref: str,
    defs: dict[str, Any],
) -> Any | None:
    """Resolve a local ``$ref`` of the form ``#/$defs/Name``.

    Returns the referenced schema dict, or ``None`` if *ref* is not a
    resolvable local reference.
    """
    if not isinstance(ref, str) or not ref.startswith("#/"):
        return None
    parts = ref[2:].split("/")
    if len(parts) == 2 and parts[0] in ("$defs", "definitions"):
        return defs.get(parts[1])
    return None


def _inline_schema_refs(
    node: Any,
    defs: dict[str, Any],
    _resolving: frozenset,
) -> Any:
    """Recursively inline ``$ref`` nodes using the provided *defs* mapping.

    Inner recursive worker for :func:`_expand_schema_refs`.
    """
    if isinstance(node, list):
        return [_inline_schema_refs(item, defs, _resolving) for item in node]
    if not isinstance(node, dict):
        return node

    ref = node.get("$ref")
    if isinstance(ref, str):
        if ref in _resolving:
            # Circular reference — break the cycle with an empty schema.
            return {}
        resolved = _resolve_local_ref(ref, defs)
        if resolved is not None:
            # Merge any sibling annotations (e.g. description) into the
            # resolved schema, then recurse to handle nested refs.
            siblings = {k: v for k, v in node.items() if k != "$ref"}
            merged = {**resolved, **siblings}
            return _inline_schema_refs(merged, defs, _resolving | {ref})
        # External or unresolvable $ref — fall through and keep as-is.

    # Recurse into all values; drop $defs / definitions from the output
    # because all references have been resolved inline.
    result: dict[str, Any] = {}
    for key, value in node.items():
        if key in ("$defs", "definitions"):
            continue
        result[key] = _inline_schema_refs(value, defs, _resolving)
    return result


def _expand_schema_refs(schema: Any) -> Any:
    """Inline all local ``$ref`` references in a JSON Schema.

    Some models (e.g. GLM-5.x via OpenCode Go) cannot process ``$ref`` /
    ``$defs`` patterns in tool parameter schemas.  When Pydantic generates
    schemas for complex nested types it emits a ``$defs`` block and refers to
    it with ``{"$ref": "#/$defs/TypeName"}``.  This function resolves every
    such reference by substituting the full definition inline, then drops the
    ``$defs`` / ``definitions`` sections so the output is a flat, self-
    contained schema that all providers can consume.

    Circular references are detected and replaced with an empty schema
    ``{}`` to avoid infinite recursion.

    Only local references of the form ``#/$defs/<name>`` or
    ``#/definitions/<name>`` are expanded; external ``$ref`` URLs are left
    unchanged.
    """
    if not isinstance(schema, dict):
        return schema
    defs = _collect_defs(schema)
    if not defs:
        # Fast-path: nothing to expand.
        return schema
    return _inline_schema_refs(schema, defs, frozenset())


def _sanitize_tool_schemas(
    tools: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Sanitize tool function schemas to be compatible with strict providers.

    Applies three passes over each tool's ``parameters`` schema:

    1. **$ref / $defs expansion** — inlines all local ``$ref`` references so
       that models which do not support ``$defs`` (e.g. GLM-5.x) receive a
       flat, self-contained schema.
    2. **Boolean schema sanitization** — replaces boolean JSON Schema values
       (``true`` / ``false``) that strict providers like DeepSeek V4 reject.
    3. **Nullable schema sanitization** — removes JSON Schema ``null`` type
       branches that OpenAI-compatible relays to Gemini-style providers reject
       in function declarations.
    """
    sanitized = []
    for tool in tools:
        if not isinstance(tool, dict):
            sanitized.append(tool)
            continue
        func = tool.get("function")
        if not isinstance(func, dict):
            sanitized.append(tool)
            continue
        params = func.get("parameters")
        if not isinstance(params, dict):
            sanitized.append(tool)
            continue
        sanitized_params = _sanitize_nullable_schemas(
            _sanitize_boolean_schemas(
                _expand_schema_refs(params),
            ),
        )
        sanitized.append(
            {**tool, "function": {**func, "parameters": sanitized_params}},
        )
    return sanitized


class OpenAIChatModelCompat(OpenAIChatModel):
    """OpenAIChatModel with robust parsing for malformed tool-call chunks
    and transparent ``extra_content`` (Gemini thought_signature) relay.

    Accepts extra constructor kwargs that ``OpenAIChatModel`` does not:

    * ``default_headers`` — injected as ``extra_headers`` on every API call
      (used for DashScope tracking headers, etc.).
    * ``extra_generate_kwargs`` — merged into every ``_call_api`` invocation
      (provider-level ``generate_kwargs`` that don't map to ``Parameters``).
    * ``endpoint_url`` — when set, calls this full URL directly instead of
      the provider's ``base_url + /chat/completions`` (for custom model
      providers with non-standard API paths).
    """

    def __init__(
        self,
        *,
        default_headers: dict[str, str] | None = None,
        extra_generate_kwargs: dict[str, Any] | None = None,
        endpoint_url: str = "",
        **kwargs: Any,
    ) -> None:
        self._default_headers = default_headers
        self._extra_generate_kwargs = extra_generate_kwargs or {}
        self._endpoint_url = endpoint_url
        super().__init__(**kwargs)

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        try:
            from ..observability.langfuse import (
                current_generation_kwargs,
            )
        except ImportError:
            pass
        else:
            try:
                langfuse_kwargs = current_generation_kwargs(self.model)
                if langfuse_kwargs:
                    kwargs = {**langfuse_kwargs, **kwargs}
            except Exception:
                logger.debug(
                    "langfuse generation kwargs failed",
                    exc_info=True,
                )
        return await super().__call__(*args, **kwargs)

    async def _call_api(
        self,
        model_name: str,
        messages: Any,
        tools: list[dict] | None = None,
        tool_choice: Any | None = None,
        **generate_kwargs: Any,
    ) -> Any:
        merged = {**self._extra_generate_kwargs, **generate_kwargs}
        if self._default_headers:
            existing = merged.get("extra_headers") or {}
            merged["extra_headers"] = {**self._default_headers, **existing}

        if not self._endpoint_url:
            return await super()._call_api(
                model_name,
                messages,
                tools,
                tool_choice,
                **merged,
            )

        return await self._call_api_with_url(
            model_name,
            messages,
            tools,
            tool_choice,
            **merged,
        )

    async def _call_api_with_url(
        self,
        model_name: str,
        messages: Any,
        tools: list[dict] | None = None,
        tool_choice: Any | None = None,
        **generate_kwargs: Any,
    ) -> Any:
        """Make API call to a custom full endpoint URL using raw httpx.

        This bypasses the OpenAI library's hardcoded ``/chat/completions``
        path so models with non-standard API paths (e.g. images/generations)
        work with the same request/response format.
        """
        import httpx
        from datetime import datetime
        from openai.types.chat import (
            ChatCompletion,
        )

        formatted_messages = await self.formatter.format(messages)

        body: dict[str, Any] = {
            "model": model_name,
            "messages": formatted_messages,
            "stream": self.stream,
        }

        if self.parameters.max_tokens is not None:
            body["max_tokens"] = self.parameters.max_tokens
        if self.parameters.temperature is not None:
            body["temperature"] = self.parameters.temperature
        if self.parameters.top_p is not None:
            body["top_p"] = self.parameters.top_p
        if (
            self.parameters.thinking_enable
            and self.parameters.reasoning_effort
        ):
            body["reasoning_effort"] = self.parameters.reasoning_effort
        if self.parameters.voice is not None:
            body["audio"] = {
                "voice": self.parameters.voice,
                "format": "pcm16",
            }
            body["modalities"] = ["text", "audio"]
        if self.extra_body is not None:
            body["extra_body"] = dict(self.extra_body)

        if self.stream:
            body["stream_options"] = {"include_usage": True}

        fmt_tools, fmt_tool_choice = self._format_tools(tools, tool_choice)
        if fmt_tools:
            body["tools"] = fmt_tools
            if not self.parameters.parallel_tool_calls:
                body["parallel_tool_calls"] = False
        if fmt_tool_choice is not None:
            body["tool_choice"] = fmt_tool_choice

        extra_headers = dict(generate_kwargs.get("extra_headers") or {})
        api_key = self.credential.api_key.get_secret_value()
        extra_headers.setdefault("Authorization", f"Bearer {api_key}")

        audio_fmt = "wav"
        audio_cfg = body.get("audio")
        if isinstance(audio_cfg, dict):
            audio_fmt = audio_cfg.get("format", "wav")

        start_dt = datetime.now()

        # Streaming path: the generator is lazy (consumed by the caller),
        # so the httpx client must be owned by the stream wrapper to stay
        # alive for the full iteration lifecycle.
        if self.stream:
            return await self._call_api_with_url_streaming(
                body, extra_headers, start_dt,
            )

        # Non-streaming: create a short-lived client, make a single request.
        async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
            resp = await client.post(
                self._endpoint_url,
                json=body,
                headers=extra_headers,
            )
            resp.raise_for_status()
            chat_completion = ChatCompletion.model_validate(resp.json())
            return self._parse_completion_response(
                start_dt,
                chat_completion,
                audio_fmt,
            )

    async def _call_api_with_url_streaming(
        self,
        body: dict[str, Any],
        headers: dict[str, str],
        start_dt: datetime,
    ) -> Any:
        """Streaming variant of ``_call_api_with_url``.

        Parses the SSE response and feeds ``ChatCompletionChunk`` objects
        into the existing stream-parsing pipeline.
        """
        import httpx
        from openai.types.chat import ChatCompletionChunk

        class _ChunkStream:
            """Minimal async context manager + iterator that wraps parsed
            SSE chunks so ``_parse_stream_response`` can consume them with
            ``async with``.

            Owns its own ``httpx.AsyncClient`` to keep it alive across the
            lazy generator lifecycle."""

            def __init__(
                self,
                endpoint_url: str,
                req_body: dict[str, Any],
                req_headers: dict[str, str],
            ):
                self._endpoint_url = endpoint_url
                self._body = req_body
                self._headers = req_headers
                self._client: httpx.AsyncClient | None = None
                self._gen: AsyncGenerator[ChatCompletionChunk, None] | None = (
                    None
                )

            async def __aenter__(self) -> "_ChunkStream":
                self._client = httpx.AsyncClient(
                    timeout=httpx.Timeout(300.0),
                )
                self._gen = self._iter_chunks()
                return self

            async def __aexit__(self, *args: Any) -> None:
                if self._client is not None:
                    await self._client.aclose()
                    self._client = None
                self._gen = None

            async def __aiter__(self) -> AsyncGenerator[ChatCompletionChunk, None]:  # type: ignore[return]
                if self._gen is None:
                    return
                async for chunk in self._gen:
                    yield chunk

            async def _iter_chunks(
                self,
            ) -> AsyncGenerator[ChatCompletionChunk, None]:
                if self._client is None:
                    return
                async with self._client.stream(
                    "POST",
                    self._endpoint_url,
                    json=self._body,
                    headers=self._headers,
                ) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        payload = line[6:].strip()
                        if payload == "[DONE]":
                            break
                        if not payload:
                            continue
                        try:
                            yield ChatCompletionChunk.model_validate_json(
                                payload,
                            )
                        except Exception as exc:
                            logger.debug(
                                "Skipping unparseable SSE chunk: %s",
                                exc,
                            )

        stream = _ChunkStream(self._endpoint_url, body, headers)
        return self._parse_stream_response(start_dt, stream)

    def _format_tools(
        self,
        tools: list[dict] | None,
        tool_choice: Any | None,
    ) -> tuple[list[dict] | None, Any]:
        """Sanitize boolean sub-schemas before forwarding to base.

        Some MCP servers declare parameters using JSON Schema boolean values
        (e.g. ``additionalProperties: true``, ``items: true``) which are valid
        per spec but rejected by strict providers such as DeepSeek V4.
        """
        if tools:
            tools = _sanitize_tool_schemas(tools)
        return super()._format_tools(tools, tool_choice)

    # pylint: disable=too-many-branches, too-many-statements
    async def _parse_stream_response(
        self,
        start_datetime: datetime,
        response: Any,
    ) -> AsyncGenerator[ChatResponse, None]:
        sanitized_response = _SanitizedStream(response)

        _think_tool_calls: dict[str, dict] = {}
        _text_tool_calls: dict[str, dict] = {}

        async for parsed in super()._parse_stream_response(
            start_datetime=start_datetime,
            response=sanitized_response,
        ):
            # Filter out malformed tool_use blocks (null id or empty name)
            # emitted by some OpenAI-compatible models, to prevent bad entries
            # from being persisted into session history (issue #4185).
            _tool_types = ("tool_use", "tool_call")

            parsed.content = [
                b
                for b in parsed.content
                if not (
                    (
                        b.get("type")
                        if isinstance(b, dict)
                        else getattr(b, "type", None)
                    )
                    in _tool_types
                    and (
                        not isinstance(
                            (
                                b.get("id")
                                if isinstance(b, dict)
                                else getattr(b, "id", None)
                            ),
                            str,
                        )
                        or not (
                            b.get("name")
                            if isinstance(b, dict)
                            else getattr(b, "name", None)
                        )
                    )
                )
            ]

            if sanitized_response.extra_contents:
                for block in parsed.content:
                    btype = (
                        block.get("type")
                        if isinstance(block, dict)
                        else getattr(block, "type", None)
                    )
                    if btype not in _tool_types:
                        continue
                    tool_id = (
                        block.get("id")
                        if isinstance(block, dict)
                        else getattr(block, "id", None)
                    )
                    if not isinstance(tool_id, str):
                        continue
                    ec = sanitized_response.extra_contents.get(tool_id)
                    if ec:
                        if isinstance(block, dict):
                            block["extra_content"] = ec
                        else:
                            block.extra_content = ec

            has_tool_use = any(
                (
                    b.get("type")
                    if isinstance(b, dict)
                    else getattr(b, "type", None)
                )
                in _tool_types
                for b in parsed.content
            )

            if has_tool_use:
                _think_tool_calls.clear()
                _text_tool_calls.clear()
            else:
                # --- 1. Scan thinking blocks ---
                for block in parsed.content:
                    btype = _battr(block, "type")
                    if btype != "thinking":
                        continue
                    thinking_text = _battr(block, "thinking") or ""
                    if not text_contains_tool_call_tag(thinking_text):
                        continue

                    think_parsed = parse_tool_calls_from_text(thinking_text)
                    if not think_parsed.tool_calls:
                        continue

                    _bset(block, "thinking", think_parsed.text_before.strip())

                    _think_tool_calls = {
                        f"thinking_{i}": {
                            "type": "tool_use",
                            "id": f"think_call_{i}",
                            "name": ptc.name,
                            "input": ptc.arguments,
                            "raw_input": ptc.raw_arguments,
                        }
                        for i, ptc in enumerate(think_parsed.tool_calls)
                    }

                # --- 2. Scan text/content blocks ---
                new_content: list | None = None
                for i, block in enumerate(parsed.content):
                    if _battr(block, "type") != "text":
                        continue
                    text = _battr(block, "text") or ""
                    if not text_contains_tool_call_tag(text):
                        continue

                    text_parsed = parse_tool_calls_from_text(text)
                    clean_text = text_parsed.text_before.strip()
                    _bset(block, "text", clean_text)

                    if text_parsed.tool_calls:
                        _text_tool_calls = {
                            f"text_{j}": {
                                "type": "tool_use",
                                "id": f"text_call_{j}",
                                "name": ptc.name,
                                "input": ptc.arguments,
                                "raw_input": ptc.raw_arguments,
                            }
                            for j, ptc in enumerate(text_parsed.tool_calls)
                        }

                    # If the text block is now empty, mark it for removal.
                    if not clean_text:
                        if new_content is None:
                            new_content = list(parsed.content)
                        new_content[i] = None  # type: ignore[index]

                if new_content is not None:
                    parsed.content = [b for b in new_content if b is not None]

                extra = list(_think_tool_calls.values()) + list(
                    _text_tool_calls.values(),
                )
                if extra:
                    parsed.content = list(parsed.content) + extra

            yield parsed
