"""MCP server wiring: tool definitions Claude Code reads, delegating to tools/*."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any, Literal

from mcp.server.mcpserver import Context, MCPServer
from mcp.types import ToolAnnotations

from .app import App
from .backend import Backend, OllamaBackend
from .config import Settings
from .outputs import OutputStore
from .routing import profile_models
from .tools import delegate, review, search, status, summarize
from .warmup import warmup

READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)
WRITES_INDEX = ToolAnnotations(read_only_hint=False, destructive_hint=False, idempotent_hint=True, open_world_hint=False)


def build_app(settings: Settings, backend: Backend | None = None) -> App:
    return App(
        settings=settings,
        backend=backend or OllamaBackend(settings.ollama_host),
        outputs=OutputStore(settings.data_dir),
    )


def build_server(settings: Settings, backend: Backend | None = None) -> MCPServer:
    app = build_app(settings, backend)
    models = profile_models(settings)

    def _tool(**kw: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """server.tool with the docstring dedented (the SDK passes it through verbatim)."""

        def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
            kw.setdefault("description", inspect.cleandoc(fn.__doc__ or ""))
            return server.tool(**kw)(fn)

        return deco

    @asynccontextmanager
    async def lifespan(_server: MCPServer) -> AsyncIterator[dict[str, Any]]:
        # Preload in the background: the MCP handshake must not wait 20 s for a model.
        task = asyncio.create_task(warmup(app)) if settings.warmup else None
        try:
            yield {}
        finally:
            if task is not None and not task.done():
                task.cancel()

    server = MCPServer(
        name="ollama-agent",
        lifespan=lifespan,
        title="Local Ollama models",
        instructions=(
            f"Local GPU models via Ollama (profile '{settings.profile}': strong={models['strong']}, "
            f"fast={models['fast']}, embeddings={models['embed']}). Use them for bulk or cheap work — "
            "summarising large files/logs, first-pass code review, semantic code search, boilerplate and "
            "test generation — so cloud context is spent on judgement. Local output is a suggestion: "
            "verify anything that matters. Do not send tasks needing more than ~32K tokens of context."
        ),
        version="0.1.0",
    )

    @_tool(annotations=READ_ONLY, structured_output=False)
    async def delegate_task(
        task: str,
        ctx: Context,
        context_files: list[str] | None = None,
        model_tier: Literal["strong", "fast"] = "strong",
        think: bool = False,
        max_tokens: int = 4096,
        timeout_s: int = 120,
    ) -> str:
        """Run a self-contained subtask on a LOCAL model and return its text.

        Good for: boilerplate, unit tests for a given file, docstrings, log triage, rewrites,
        first drafts, mechanical transformations. Pass the files it needs in `context_files`
        (absolute paths, ~90k chars total). `model_tier="fast"` is ~1.4x quicker for simple
        jobs; `think=true` adds reasoning for tricky ones (slower). Output is saved to a file
        whose path is returned, so ask for long outputs freely.

        NOT for: final correctness decisions, reasoning about the whole repo, tasks needing
        more than ~32K tokens of context, or anything you must get right without checking.
        """
        return await delegate.delegate_task(
            app,
            task=task,
            context_files=context_files,
            model_tier=model_tier,
            think=think,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
            ctx=ctx,
        )

    @_tool(annotations=READ_ONLY, structured_output=True)
    async def review_diff(
        ctx: Context,
        diff: str = "",
        git_range: str = "",
        cwd: str = "",
        focus: str = "",
        think: bool = True,
        timeout_s: int = 180,
    ) -> review.ReviewResult:
        """Second-opinion code review of a diff by a LOCAL model; returns structured findings.

        Give either `diff` text, or `git_range` and let the server run `git diff` in `cwd`
        (defaults: cwd = current dir, range = HEAD i.e. all uncommitted changes). Examples:
        `git_range="--staged"`, `git_range="main...HEAD"`. `focus` narrows the review
        ("concurrency", "error handling"). Findings carry file, line, severity
        (critical/major/minor/nit), summary and suggestion.

        Use it as a first pass before or alongside your own review, then verify each finding
        against the code — it is a cheap second opinion, not a verdict.
        """
        return await review.review_diff(
            app,
            diff=diff,
            git_range=git_range,
            cwd=cwd,
            focus=focus,
            think=think,
            timeout_s=timeout_s,
            ctx=ctx,
        )

    @_tool(annotations=READ_ONLY, structured_output=False)
    async def summarize(
        ctx: Context,
        paths: list[str] | None = None,
        text: str = "",
        question: str = "",
        timeout_s: int = 300,
    ) -> str:
        """Summarise large files, logs or text with LOCAL models (map-reduce; megabytes are fine).

        Use instead of Read when a file is long (logs, dumps, generated code, long docs) and
        you need the gist or an answer to a specific `question` ("what errors occurred after
        14:00?"). Accepts absolute `paths` and/or inline `text`. Concrete details
        (identifiers, numbers, errors) are preserved. Full summary is also saved to a file.

        NOT for short files (<200 lines): just Read them.
        """
        return await summarize.summarize(
            app, paths=paths, text=text, question=question, timeout_s=timeout_s, ctx=ctx
        )

    @_tool(annotations=WRITES_INDEX, structured_output=False)
    async def index_codebase(
        root: str,
        ctx: Context,
        include_globs: list[str] | None = None,
        force: bool = False,
    ) -> str:
        """Build or refresh the LOCAL semantic search index for a directory tree.

        Incremental: only files whose mtime/hash changed are re-embedded. Stored at
        `<root>/.ollama-agent/index.sqlite`. Run once per repo; `search_code` refreshes it
        automatically afterwards. `include_globs` defaults to common source/doc extensions.
        """
        return await search.index_codebase(
            app, root=root, include_globs=include_globs, force=force, ctx=ctx
        )

    @_tool(annotations=READ_ONLY, structured_output=False)
    async def search_code(
        query: str,
        root: str,
        ctx: Context,
        top_k: int = 8,
        refresh: bool = True,
    ) -> str:
        """Semantic search over a codebase by MEANING using local embeddings.

        Ask in natural language: "where do we retry failed uploads?", "code that parses the
        config file". Returns the best-matching chunks as path:start-end plus a snippet.
        Refreshes the index first unless `refresh=false`.

        NOT for exact identifiers or strings — use Grep for those.
        """
        return await search.search_code(
            app, query=query, root=root, top_k=top_k, refresh=refresh, ctx=ctx
        )

    @_tool(annotations=READ_ONLY, structured_output=False)
    async def local_models_status() -> str:
        """Show the active local profile, which model serves each tier, and what Ollama has loaded.

        Call this when a local tool is slow or fails, or before a batch of local work, to see
        VRAM placement and whether an unrelated model is loaded and competing for the GPU.
        """
        return await status.local_models_status(app)

    return server
