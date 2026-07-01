# -*- coding: utf-8 -*-
"""ReMe-backed memory manager for agents.

The public class and registry key keep the historical ``ReMeLight`` naming so
existing agent configs continue to work, but the implementation delegates to
ReMe's application/job framework.
"""

import json
import logging
import uuid
from contextlib import suppress
from typing import Any, TYPE_CHECKING

import httpx
from agentscope.message import Msg, TextBlock
from agentscope.message import ToolCallBlock, ToolCallState
from agentscope.message import ToolResultBlock, ToolResultState
from agentscope.tool import ToolChunk

from .base_memory_manager import BaseMemoryManager, memory_registry
from .prompts import build_memory_guidance_prompt
from .reme_config import get_reme_app_config
from ..model_factory import create_model_and_formatter
from ...app.inbox_store import append_event as append_inbox_event
from ...config import load_config
from ...constant import (
    AUTO_MEMORY_SEARCH_MESSAGE_TAG,
    AUTO_MEMORY_SEARCH_TEXT,
    QWENPAW_MESSAGE_TAG_KEY,
)
from ...config.config import (
    load_agent_config,
    AgentProfileConfig,
    RerankerConfig,
)

if TYPE_CHECKING:
    from reme import ReMe
    from reme.application import Response

logger = logging.getLogger(__name__)

MAX_QUERY_CHARS = 50
NO_MEMORY_RESULTS = "(no memory results)"
INBOX_RESULT_JOB_NAMES = {"auto_memory", "auto_dream", "auto_resource"}
INBOX_RESULT_HOOK_KEY = "qwenpaw_memory_result_hook"
INBOX_EMITTED_METADATA_KEY = "_qwenpaw_inbox_emitted"
MAX_INBOX_BODY_CHARS = 4000


def _tool_chunk(text: str, *, ok: bool = True) -> ToolChunk:
    return ToolChunk(
        is_last=True,
        state=ToolResultState.SUCCESS if ok else ToolResultState.ERROR,
        content=[TextBlock(type="text", text=text)],
    )


@memory_registry.register("remelight")
class ReMeLightMemoryManager(BaseMemoryManager):
    """Memory manager backed by ReMe.

    ReMe uses the QwenPaw workspace root as its vault.  Daily memory,
    digest memory, search, auto-memory, and auto-dream are executed through
    ReMe jobs.
    """

    def __init__(self, working_dir: str, agent_id: str):
        super().__init__(working_dir=working_dir, agent_id=agent_id)
        self._reme: "ReMe | None" = None
        self._reranker_config_cache: RerankerConfig | None = None
        logger.info(
            "ReMeLightMemoryManager init: agent_id=%s working_dir=%s",
            agent_id,
            working_dir,
        )

        try:
            from reme import ReMe as ReMeApp  # type: ignore

            agent_config: AgentProfileConfig = load_agent_config(self.agent_id)
            global_config = load_config()
            self._reme = ReMeApp(
                **get_reme_app_config(
                    working_dir=self.working_dir,
                    agent_config=agent_config,
                    user_timezone=getattr(
                        global_config,
                        "user_timezone",
                        None,
                    ),
                ),
            )
            self._install_reme_result_hook()
        except Exception as exc:
            logger.warning("ReMe import failed; memory disabled: %s", exc)

    async def start(self) -> None:
        """Start the embedded ReMe application."""
        if self._reme is None:
            return

        await self._update_qwenpaw_model()
        try:
            await self._reme.start()
            logger.info(
                "ReMe memory manager started for agent '%s'",
                self.agent_id,
            )
        except Exception:
            logger.exception("ReMe start failed")
            return

        agent_config = load_agent_config(self.agent_id)
        cfg = agent_config.running.reme_light_memory_config
        if cfg.rebuild_memory_index_on_start:
            await self._run_reme_job("reindex")
            logger.info(
                "Memory index rebuilt on start for agent '%s'",
                self.agent_id,
            )

    async def close(self) -> bool:
        """Close ReMe and cleanup background summary worker state."""
        logger.info(
            "ReMeLightMemoryManager closing: agent_id=%s",
            self.agent_id,
        )

        worker = self._worker_task
        if worker is not None and not worker.done():
            worker.cancel()
            with suppress(BaseException):
                await worker

        if self._reme is not None:
            try:
                await self._reme.close()
            except Exception:
                logger.exception("ReMe close failed")
                return False

        self._reme = None
        return True

    def get_memory_prompt(self) -> str:
        """Return memory guidance for system prompt injection."""
        agent_config = load_agent_config(self.agent_id)
        cfg = agent_config.running.reme_light_memory_config
        return build_memory_guidance_prompt(
            agent_config.language,
            daily_dir=cfg.daily_dir,
        )

    def list_memory_tools(self):
        """Return memory tool functions to register with the agent toolkit."""
        return [self.memory_search]

    def get_auto_memory_interval(self) -> int:
        """Return ReMe light auto-memory cadence from agent config."""
        agent_config = load_agent_config(self.agent_id)
        interval = (
            agent_config.running.reme_light_memory_config.auto_memory_interval
        )
        if interval is None:
            return 0
        return int(interval)

    async def _update_qwenpaw_model(self) -> None:
        """Reuse QwenPaw's active model in ReMe's default LLM component."""
        if self._reme is None:
            return

        model, _formatter = create_model_and_formatter(self.agent_id)
        await self._reme.update_component(
            "as_llm",
            "default",
            model=model,
        )

    async def _run_reme_job(
        self,
        name: str,
        *,
        needs_llm: bool = False,
        **kwargs: Any,
    ) -> "Response | None":
        if self._reme is None or not getattr(self._reme, "is_started", False):
            logger.debug("ReMe job skipped; app not started: %s", name)
            return None
        try:
            if needs_llm:
                await self._update_qwenpaw_model()
            response = await self._reme.run_job(name, **kwargs)
            await self._append_reme_job_result_to_inbox(
                name,
                response=response,
                kwargs=kwargs,
            )
            return response
        except Exception:
            logger.exception("ReMe job failed: %s", name)
            return None

    def _install_reme_result_hook(self) -> None:
        """Expose QwenPaw inbox delivery to ReMe background steps."""
        if self._reme is None:
            return
        context = getattr(self._reme, "context", None)
        metadata = getattr(context, "metadata", None)
        if not isinstance(metadata, dict):
            logger.debug("ReMe result hook skipped; metadata unavailable")
            return
        metadata[INBOX_RESULT_HOOK_KEY] = self._handle_reme_result_hook

    async def _handle_reme_result_hook(
        self,
        *,
        job_name: str,
        response: "Response",
        kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Handle result notifications emitted from ReMe background steps."""
        del metadata
        await self._append_reme_job_result_to_inbox(
            job_name,
            response=response,
            kwargs=kwargs or {},
        )

    async def _append_reme_job_result_to_inbox(
        self,
        name: str,
        *,
        response: "Response",
        kwargs: dict[str, Any],
    ) -> bool:
        if name not in INBOX_RESULT_JOB_NAMES:
            return False
        response_metadata = getattr(response, "metadata", None)
        if isinstance(response_metadata, dict) and response_metadata.get(
            INBOX_EMITTED_METADATA_KEY,
        ):
            return False
        if (
            name in {"auto_memory", "auto_resource"}
            and isinstance(response_metadata, dict)
            and response_metadata.get("modified") is False
        ):
            logger.info(
                "ReMe job result inbox push skipped; no memory change: "
                "agent_id=%s job_name=%s modified=False",
                self.agent_id,
                name,
            )
            return False

        answer = str(getattr(response, "answer", "") or "").strip()
        if len(answer) > MAX_INBOX_BODY_CHARS:
            answer = f"{answer[:MAX_INBOX_BODY_CHARS].rstrip()}\n..."
        success = bool(getattr(response, "success", False))
        title = self._inbox_result_title(name)
        body = answer or self._empty_inbox_result_body(name)
        payload: dict[str, Any] = {
            "job_name": name,
            "session_id": str(kwargs.get("session_id") or ""),
            "date": str(kwargs.get("date") or ""),
            "hint": str(
                kwargs.get("memory_hint") or kwargs.get("hint") or "",
            ),
        }
        if name == "auto_resource":
            changes = kwargs.get("changes") or []
            if isinstance(changes, list):
                payload["change_count"] = len(changes)
            if isinstance(response_metadata, dict):
                payload["processed"] = response_metadata.get("processed")

        try:
            event = await append_inbox_event(
                agent_id=self.agent_id,
                source_type="memory",
                source_id=name,
                event_type=f"{name}_result",
                status="success" if success else "error",
                severity="info" if success else "error",
                title=title,
                body=body,
                payload=payload,
            )
            if isinstance(response_metadata, dict):
                response_metadata[INBOX_EMITTED_METADATA_KEY] = True
            logger.info(
                "ReMe job result pushed to inbox: "
                "agent_id=%s job_name=%s event_id=%s status=%s modified=%s",
                self.agent_id,
                name,
                event.get("id"),
                event.get("status"),
                (
                    response_metadata.get("modified")
                    if isinstance(response_metadata, dict)
                    else None
                ),
            )
            return True
        except Exception:  # pylint: disable=broad-except
            logger.exception(
                "failed to push ReMe job result to inbox: "
                "agent_id=%s job_name=%s success=%s",
                self.agent_id,
                name,
                success,
            )
            return False

    @staticmethod
    def _inbox_result_title(name: str) -> str:
        return {
            "auto_memory": "Auto-memory result",
            "auto_dream": "Auto-dream result",
            "auto_resource": "Auto-resource result",
        }.get(name, "Memory job result")

    @staticmethod
    def _empty_inbox_result_body(name: str) -> str:
        return {
            "auto_memory": "Auto-memory completed with no returned content.",
            "auto_dream": "Auto-dream completed with no returned content.",
            "auto_resource": (
                "Auto-resource completed with no returned content."
            ),
        }.get(name, "Memory job completed with no returned content.")

    async def memory_search(
        self,
        query: str,
        max_results: int = 5,
        min_score: float = 0,
    ) -> ToolChunk:
        """Search memory files semantically.

        Use this tool before answering questions about prior work,
        decisions, dates, people, preferences, or todos. Returns top
        relevant snippets with file paths and line numbers.

        Args:
            query (`str`):
                The semantic search query to find relevant memory snippets.
            max_results (`int`, optional):
                Maximum number of search results to return. Defaults to 5.
            min_score (`float`, optional):
                Minimum relevance score for results. Defaults to 0; keep this
                at 0 in normal use because ReMe search may mix BM25 and fused
                scores with different scales, and raising it can hide valid
                keyword matches.

        Returns:
            `ToolResponse`:
                Search results formatted with paths, line numbers, and
                content.
        """
        query = query.strip()
        if not query:
            return _tool_chunk("Error: query cannot be empty", ok=False)

        response = await self._run_reme_job(
            "search",
            query=query,
            limit=max(1, max_results),
            min_score=max(0.0, min_score),
        )
        if response is None:
            return _tool_chunk("ReMe is not started.", ok=False)

        # Apply reranker if configured
        if response.success and response.metadata.get("results"):
            reranker_config = self._get_reranker_config()
            if reranker_config:
                try:
                    await self._rerank_search_results(
                        query,
                        response,
                        reranker_config,
                    )
                except Exception:
                    logger.warning(
                        "[rerank] failed, using original order",
                        exc_info=True,
                    )

        answer = str(response.answer or "").strip()
        if not answer:
            answer = NO_MEMORY_RESULTS
        return _tool_chunk(answer, ok=response.success)

    async def _rerank_search_results(
        self,
        query: str,
        response: "Response",
        config: RerankerConfig,
    ) -> None:
        """Re-order search results using a dedicated reranker API."""
        results = response.metadata.get("results")
        if not results or len(results) <= 1:
            return

        # Truncate long texts to 500 chars each for the reranker call
        texts: list[str] = [r.get("text", "")[:500] for r in results]

        new_order = await self._call_reranker_api(query, texts, config)
        if not new_order or len(new_order) != len(results):
            return

        reordered: list[dict] = []
        for idx in new_order:
            if 0 <= idx < len(results):
                reordered.append(results[idx])

        if len(reordered) != len(results):
            return

        response.metadata["results"] = reordered

        # Rebuild answer text (same format as search_step)
        answer_lines: list[str] = []
        for r in reordered:
            path = r.get("path", "")
            start_line = r.get("start_line", 0)
            end_line = r.get("end_line", 0)
            score = r.get("score", 0.0)
            text = r.get("text", "")
            header = (
                f"========== {path}:{start_line}-{end_line} "
                f"[score={score:.4f}] =========="
            )
            answer_lines.append(f"{header}\n{text}")
        response.answer = "\n".join(answer_lines)

        logger.info(
            "[rerank] reordered %d results with model=%s",
            len(results),
            config.model_name,
        )

    def _get_reranker_config(self) -> RerankerConfig | None:
        """Return cached reranker config, or None if not enabled.

        Config is cached on first access and refreshed when the agent
        config is reloaded (via `_refresh_config_cache`).
        """
        if self._reranker_config_cache is not None:
            return self._reranker_config_cache

        try:
            agent_cfg = load_agent_config(self.agent_id)
            cfg = agent_cfg.running.reme_light_memory_config.reranker_config
            if cfg and cfg.enabled and cfg.model_name:
                self._reranker_config_cache = cfg
                return cfg
        except Exception:
            logger.warning("[rerank] failed to load config", exc_info=True)

        self._reranker_config_cache = None
        return None

    def _refresh_config_cache(self) -> None:
        """Invalidate cached reranker config (call after config reload)."""
        self._reranker_config_cache = None

    async def _call_reranker_api(
        self,
        query: str,
        documents: list[str],
        config: RerankerConfig,
    ) -> list[int] | None:
        """Call a reranker API to score and reorder documents by relevance.

        Uses the standard OpenAI-compatible reranker endpoint::

            POST {base_url}/rerank
            {
                "model": "...",
                "query": "...",
                "documents": ["...", ...],
                "top_n": N
            }

        Returns a list of indices sorted by relevance (most relevant first),
        or ``None`` on failure.
        """
        if not config.base_url:
            logger.warning("[rerank] base_url not configured")
            return None
        if not query or not documents:
            return None

        base_url = config.base_url.rstrip("/")
        url = f"{base_url}/rerank"

        payload: dict[str, Any] = {
            "model": config.model_name,
            "query": query,
            "documents": documents,
        }

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    url,
                    headers={
                        "Authorization": f"Bearer {config.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
            resp.raise_for_status()
            data = resp.json()

            # Reranker response format: { "results": [{"index": 0, "relevance_score": 0.99}, ...] }  # noqa: E501
            results = data.get("results", [])
            if not results:
                return None

            # Sort by relevance score descending
            # Some providers use "relevance_score", others use "score"
            scored = []
            for i, r in enumerate(results):
                score_val = r.get("relevance_score") or r.get("score") or 0.0
                scored.append((r.get("index", i), score_val))
            scored.sort(key=lambda x: x[1], reverse=True)

            return [idx for idx, _ in scored]
        except httpx.TimeoutException:
            logger.warning("[rerank] API timeout")
            return None
        except httpx.RequestError as e:
            logger.warning("[rerank] HTTP error: %s", e)
            return None
        except Exception as e:
            logger.warning("[rerank] unexpected error: %s", e)
            return None

    async def summarize(
        self,
        messages: list[Msg],
        **kwargs: Any,
    ) -> str:
        """Persist conversation messages through ReMe auto-memory."""
        if not messages:
            return ""

        response = await self._run_reme_job(
            "auto_memory",
            needs_llm=True,
            messages=[msg.model_dump(mode="json") for msg in messages],
            session_id=str(kwargs.get("session_id") or ""),
            memory_hint=str(kwargs.get("memory_hint") or ""),
        )
        if response is None:
            return ""
        return str(response.answer or "")

    async def auto_memory_search(
        self,
        messages: list[Msg] | Msg,
        agent_name: str = "",
        **kwargs: Any,
    ) -> dict | None:
        """Auto-search memory and expose it as a completed tool interaction."""
        del kwargs
        agent_config = load_agent_config(self.agent_id)
        memory_cfg = agent_config.running.reme_light_memory_config
        if not memory_cfg.auto_memory_search_config.enabled:
            return None

        msgs = [messages] if isinstance(messages, Msg) else list(messages)
        query = self._build_query(msgs)
        if not query:
            return None

        search_cfg = memory_cfg.auto_memory_search_config

        response = await self._run_reme_job(
            "search",
            query=query,
            limit=max(1, search_cfg.max_results),
            min_score=0,
        )
        if response is None or not response.success:
            return None

        text = str(response.answer or "").strip()
        if not text:
            return None

        tool_call_id = uuid.uuid4().hex
        tool_input = {
            "query": query,
            "max_results": search_cfg.max_results,
        }
        assistant_msg = Msg(
            name=agent_name or self.agent_id,
            role="assistant",
            metadata={
                QWENPAW_MESSAGE_TAG_KEY: AUTO_MEMORY_SEARCH_MESSAGE_TAG,
            },
            content=[
                TextBlock(text=AUTO_MEMORY_SEARCH_TEXT),
                ToolCallBlock(
                    id=tool_call_id,
                    name="memory_search",
                    input=json.dumps(tool_input, ensure_ascii=False),
                    state=ToolCallState.FINISHED,
                ),
            ],
        )
        tool_result_msg = Msg(
            name=agent_name or self.agent_id,
            role="assistant",
            metadata={
                QWENPAW_MESSAGE_TAG_KEY: AUTO_MEMORY_SEARCH_MESSAGE_TAG,
            },
            content=[
                ToolResultBlock(
                    id=tool_call_id,
                    name="memory_search",
                    output=[TextBlock(text=text)],
                    state=ToolResultState.SUCCESS,
                ),
            ],
        )
        return {
            "query": query,
            "text": text,
            "msg": msgs + [assistant_msg, tool_result_msg],
        }

    async def auto_memory(
        self,
        all_messages: list[Msg],
        **kwargs: Any,
    ) -> None:
        """Auto-extract memory for a prepared reply batch."""
        if not all_messages:
            return
        session_id = str(kwargs.get("session_id") or "")
        if not session_id:
            logger.warning(
                "ReMe auto_memory skipped; session_id is empty: "
                "agent_id=%s messages=%s",
                self.agent_id,
                len(all_messages),
            )
            return

        self.add_summarize_task(
            messages=all_messages,
            session_id=session_id,
        )

    async def dream(self, **kwargs: Any) -> None:
        """Run one ReMe auto-dream pass."""
        response = await self._run_reme_job(
            "auto_dream",
            needs_llm=True,
            date=str(kwargs.get("date") or ""),
            hint=str(kwargs.get("hint") or ""),
        )
        if response is not None and not response.success:
            raise RuntimeError(str(response.answer))

    @staticmethod
    def _build_query(messages: list[Msg]) -> str:
        parts = []
        total = 0
        for msg in reversed(messages):
            if msg.role not in {"user", "assistant"}:
                continue
            text = (msg.get_text_content() or "").strip()
            if not text:
                continue
            remaining = MAX_QUERY_CHARS - total - (1 if parts else 0)
            if remaining <= 0:
                break
            parts.insert(0, text[-remaining:])
            total += min(len(text), remaining) + (1 if len(parts) > 1 else 0)
        return " ".join(parts).strip()
