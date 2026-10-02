"""实现通过实例复用的 AI 种子搜索客户端。"""

import asyncio
import logging
import math
import re
import unicodedata
from typing import Any, Mapping, Protocol

import httpx

from ai_torrent_search.analyzer import ChatModel, StructuredIntentAnalyzer
from ai_torrent_search.models import (
    AITorrentSearchError,
    AITorrentSearchResult,
    QueryParameters,
    SearchIntent,
)
from ai_torrent_search.provider import TorrentApiPyProvider, TorrentProvider, TorrentProviderError


logger = logging.getLogger("ai_torrent_search.client")


class IntentAnalyzer(Protocol):
    """定义可以替换默认结构化模型分析器的最小接口。"""

    async def analyze(self, query: str) -> SearchIntent:
        """把自然语言请求转换为结构化影视意图。"""
        ...


class AITorrentSearch:
    """组合模型意图识别、多查询种子检索、去重和相关性重排。"""

    def __init__(
        self,
        model: ChatModel,
        query_parameters: QueryParameters | Mapping[str, Any] | None = None,
        *,
        http_client: httpx.AsyncClient | None = None,
        analyzer: IntentAnalyzer | None = None,
        provider: TorrentProvider | None = None,
    ):
        """校验初始化参数，并保存可选的共享客户端和可替换依赖。"""
        # 模型由调用方创建；搜索器只依赖统一的 LangChain 风格 ainvoke 接口。
        self.model = model
        self.query_parameters = QueryParameters.model_validate(query_parameters or {})
        # 网络客户端延迟创建，因此实例可以在异步上下文外构造。
        self._client = http_client
        self._analyzer = analyzer or StructuredIntentAnalyzer(model)
        self._provider = provider
        # 只关闭实例自行创建的客户端，避免影响应用共享连接池。
        self._owns_client = http_client is None
        self._initialization_lock = asyncio.Lock()
        self._closed = False

    async def search(self, request: str) -> AITorrentSearchResult:
        """理解自然语言需求，执行扩展查询并返回去重重排结果。"""
        raw_request = request.strip()
        if not 1 <= len(raw_request) <= 200:
            raise AITorrentSearchError("请输入 1–200 字的搜索需求", 400)
        await self._ensure_ready()
        assert self._analyzer is not None
        assert self._provider is not None
        # 模型只生成搜索计划，不生成种子内容或链接。
        intent = await self._analyzer.analyze(raw_request)
        candidates = [intent.primary_query, *intent.queries, *intent.aliases]
        expanded_queries: list[str] = []
        for candidate in candidates:
            cleaned = candidate.strip()
            if cleaned and cleaned not in expanded_queries:
                expanded_queries.append(cleaned)
            if len(expanded_queries) >= self.query_parameters.max_expanded_queries:
                break
        # 多个扩展词并发查询，同一分支失败时保留其他来源结果。
        branches = await asyncio.gather(
            *(self._search_one(query, intent) for query in expanded_queries),
            return_exceptions=True,
        )
        raw_items: list[dict[str, Any]] = []
        warnings: list[str] = []
        failures: list[BaseException] = []
        for query, branch in zip(expanded_queries, branches):
            if isinstance(branch, BaseException):
                failures.append(branch)
                warnings.append(f"查询“{query}”失败：{branch}")
            else:
                raw_items.extend(branch)
        if failures and len(failures) == len(expanded_queries):
            first = failures[0]
            status = first.status if isinstance(first, TorrentProviderError) else 502
            raise AITorrentSearchError(f"种子搜索来源全部失败：{first}", status) from first
        ranked = self._merge_and_rank(raw_items, intent)
        limited = ranked[:self.query_parameters.result_limit]
        result = AITorrentSearchResult(
            intent=intent.model_dump(),
            expanded_queries=expanded_queries,
            items=limited,
            total=len(limited),
            warnings=warnings,
        )
        logger.info(
            "search_completed queries=%d raw_items=%d returned=%d warnings=%d",
            len(expanded_queries), len(raw_items), result.total, len(warnings),
        )
        return result

    async def _ensure_ready(self) -> None:
        """按需创建默认种子来源使用的 HTTP 客户端。"""
        if self._closed:
            raise AITorrentSearchError("AI 种子搜索实例已经关闭", 400)
        if self._analyzer is not None and self._provider is not None:
            return
        async with self._initialization_lock:
            if self._client is None:
                self._client = httpx.AsyncClient(follow_redirects=True)
            if self._provider is None:
                self._provider = TorrentApiPyProvider(self._client, self.query_parameters)

    async def _search_one(self, query: str, intent: SearchIntent) -> list[dict[str, Any]]:
        """执行一个扩展查询，并补充命中查询和资源类型。"""
        assert self._provider is not None
        items = await self._provider.search(
            query,
            self._category(intent),
            self.query_parameters.page_size,
        )
        return [
            {**item, "matched_query": query, "resource_type": "torrent"}
            for item in items
            if isinstance(item, dict)
        ]

    def _category(self, intent: SearchIntent) -> str:
        """把模型类别和画质映射为种子来源的分类提示。"""
        configured = self.query_parameters.category
        if configured != "auto":
            return configured
        if intent.category not in {"movie", "tv"}:
            return "video"
        return f"hd_{intent.category}" if intent.quality else intent.category

    def _merge_and_rank(self, raw_items: list[dict[str, Any]], intent: SearchIntent) -> list[dict[str, Any]]:
        """执行实体过滤、info hash 去重和确定性排序。"""
        required = [_normalize(value) for value in intent.required_terms if value.strip()]
        excluded = [_normalize(value) for value in intent.excluded_terms if value.strip()]
        deduplicated: dict[str, dict[str, Any]] = {}
        for source_item in raw_items:
            item = dict(source_item)
            title = _normalize(str(item.get("title") or ""))
            if self.query_parameters.required_terms_filter and required:
                if not all(term in title for term in required):
                    continue
            if excluded and any(term in title for term in excluded):
                continue
            key = str(item.get("info_hash") or item.get("id") or title).lower()
            if not key:
                continue
            item["match_score"] = _score(item, intent)
            previous = deduplicated.get(key)
            if previous is None or item["match_score"] > previous["match_score"]:
                deduplicated[key] = item
        items = list(deduplicated.values())
        configured_sort = self.query_parameters.sort
        sort = intent.sort if configured_sort == "auto" else configured_sort
        if sort == "size":
            key = lambda item: (int(item.get("size_bytes") or 0), item["match_score"])
        elif sort in {"latest", "recent"}:
            key = lambda item: (str(item.get("added_at") or ""), item["match_score"])
        elif sort == "seeders":
            key = lambda item: (int(item.get("seeders") or 0), item["match_score"])
        else:
            key = lambda item: (item["match_score"], int(item.get("seeders") or 0))
        return sorted(items, key=key, reverse=True)

    async def close(self) -> None:
        """关闭实例自行创建的 HTTP 客户端。"""
        if self._closed:
            return
        self._closed = True
        if self._owns_client and self._client is not None:
            await self._client.aclose()

    async def __aenter__(self) -> "AITorrentSearch":
        """初始化资源并返回异步上下文中的实例。"""
        await self._ensure_ready()
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        """退出异步上下文时释放内部连接。"""
        await self.close()


def _normalize(value: str) -> str:
    """统一大小写、全半角和空白。"""
    return "".join(unicodedata.normalize("NFKC", value).casefold().split())


def _score(item: Mapping[str, Any], intent: SearchIntent) -> float:
    """根据实体、别名、画质、年份和做种快照计算相关度。"""
    title = _normalize(str(item.get("title") or ""))
    primary = _normalize(intent.primary_query)
    score = 50.0 if primary and primary in title else 0.0
    score += sum(8.0 for alias in intent.aliases if _normalize(alias) in title)
    if intent.quality and _normalize(intent.quality) in title:
        score += 10.0
    if intent.year and str(intent.year) in title:
        score += 6.0
    if intent.season and re.search(rf"(?:s|season)0?{intent.season}(?:\D|$)", title, re.I):
        score += 6.0
    score += min(20.0, math.log1p(max(0, int(item.get("seeders") or 0))) * 5)
    return round(score, 3)
