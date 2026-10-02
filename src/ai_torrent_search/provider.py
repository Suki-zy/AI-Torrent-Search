"""定义可替换种子来源以及默认 Torrent-Api-py/BitSearch 实现。"""

import asyncio
import base64
import re
from datetime import datetime
from typing import Any, Protocol
from urllib.parse import quote

import httpx
from bs4 import BeautifulSoup

from ai_torrent_search.models import QueryParameters


HASH_PATTERN = re.compile(r"[0-9a-fA-F]{40}")
MAGNET_HASH_PATTERN = re.compile(r"urn:btih:([0-9a-fA-F]{40}|[A-Z2-7]{32})", re.I)
SIZE_PATTERN = re.compile(r"([0-9]+(?:\.[0-9]+)?)\s*([KMGT]?I?B)?", re.I)


class TorrentProviderError(Exception):
    """表示远程种子来源请求失败或响应不可识别。"""

    def __init__(self, message: str, status: int = 502):
        """保存可由上层继续传递的消息和状态码。"""
        super().__init__(message)
        self.status = status


class TorrentProvider(Protocol):
    """定义调用方可以替换的种子检索来源接口。"""

    async def search(self, query: str, category: str, limit: int) -> list[dict[str, Any]]:
        """返回统一字段的种子候选列表。"""
        ...


class TorrentApiPyProvider:
    """并发调用 Torrent-Api-py 分站，并兼容当前 BitSearch 页面。"""

    def __init__(self, client: httpx.AsyncClient, parameters: QueryParameters):
        """保存共享 HTTP 客户端和经过校验的查询配置。"""
        self.client = client
        self.parameters = parameters

    async def search(self, query: str, category: str, limit: int) -> list[dict[str, Any]]:
        """查询全部配置站点，并规范化、过滤和去重结果。"""
        headers = {"Accept": "application/json", "User-Agent": "ai-torrent-search/0.1"}
        if self.parameters.provider_api_key:
            headers["X-API-Key"] = self.parameters.provider_api_key.get_secret_value()
        results = await asyncio.gather(
            *(self._search_site(site, query, headers) for site in self.parameters.provider_sites),
            return_exceptions=True,
        )
        items: list[dict[str, Any]] = []
        errors: list[BaseException] = []
        for site, result in zip(self.parameters.provider_sites, results):
            if isinstance(result, BaseException):
                errors.append(result)
                continue
            for raw in result:
                item = _normalize_item(raw, category, f"torrent-api-py:{site}")
                if item is not None and _matches_query(item["title"], query):
                    items.append(item)
        deduplicated: dict[str, dict[str, Any]] = {}
        for item in items:
            previous = deduplicated.get(item["info_hash"])
            if previous is None or item["seeders"] > previous["seeders"]:
                deduplicated[item["info_hash"]] = item
        if deduplicated:
            return list(deduplicated.values())[:limit]
        if len(errors) == len(self.parameters.provider_sites):
            first = errors[0]
            if isinstance(first, TorrentProviderError):
                raise first
            raise TorrentProviderError(f"全部种子来源失败：{first}") from first
        return []

    async def _search_site(
        self, site: str, query: str, headers: dict[str, str]
    ) -> list[dict[str, Any]]:
        """调用一个分站，并对瞬时网络故障进行有限重试。"""
        if site == "bitsearch":
            return await self._search_bitsearch(query, headers)
        last_error = TorrentProviderError(f"{site} 来源暂时不可用")
        for attempt in range(self.parameters.provider_retries + 1):
            try:
                response = await self.client.get(
                    f"{self.parameters.provider_url}/api/v1/search",
                    params={"site": site, "query": query, "limit": self.parameters.provider_limit},
                    headers=headers,
                    timeout=self.parameters.provider_timeout_seconds,
                )
            except httpx.TimeoutException as exc:
                last_error = TorrentProviderError(f"{site} 来源响应超时", 504)
                last_error.__cause__ = exc
            except httpx.RequestError as exc:
                last_error = TorrentProviderError(f"无法连接 {site} 来源")
                last_error.__cause__ = exc
            else:
                if response.status_code == 404:
                    return []
                if response.status_code == 429:
                    raise TorrentProviderError(f"{site} 来源请求频繁", 429)
                if response.status_code == 403:
                    raise TorrentProviderError(f"{site} 来源拒绝请求或 API Key 无效", 503)
                if response.is_success:
                    try:
                        payload = response.json()
                    except ValueError as exc:
                        raise TorrentProviderError(f"{site} 来源返回无效 JSON") from exc
                    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
                        raise TorrentProviderError(f"{site} 来源返回无法识别的数据")
                    return [item for item in payload["data"] if isinstance(item, dict)]
                last_error = TorrentProviderError(f"{site} 来源返回 HTTP {response.status_code}")
            if attempt < self.parameters.provider_retries:
                await asyncio.sleep(0.3 * (attempt + 1))
        raise last_error

    async def _search_bitsearch(self, query: str, headers: dict[str, str]) -> list[dict[str, Any]]:
        """并发抓取 BitSearch 多页，保留成功分页的结果。"""
        html_headers = {**headers, "Accept": "text/html,application/xhtml+xml"}
        pages = await asyncio.gather(
            *(
                self._fetch_bitsearch_page(query, page, html_headers)
                for page in range(1, self.parameters.bitsearch_pages + 1)
            ),
            return_exceptions=True,
        )
        items: list[dict[str, Any]] = []
        errors: list[BaseException] = []
        for page in pages:
            if isinstance(page, BaseException):
                errors.append(page)
            else:
                items.extend(page)
        if items or len(errors) < self.parameters.bitsearch_pages:
            return items
        first = errors[0]
        if isinstance(first, TorrentProviderError):
            raise first
        raise TorrentProviderError(f"BitSearch 全部分页失败：{first}") from first

    async def _fetch_bitsearch_page(
        self, query: str, page: int, headers: dict[str, str]
    ) -> list[dict[str, Any]]:
        """获取并解析一页 BitSearch HTML。"""
        last_error = TorrentProviderError("BitSearch 暂时不可用")
        for attempt in range(self.parameters.provider_retries + 1):
            try:
                response = await self.client.get(
                    f"{self.parameters.bitsearch_url}/search",
                    params={"q": query, "page": page},
                    headers=headers,
                    timeout=self.parameters.provider_timeout_seconds,
                )
            except httpx.TimeoutException as exc:
                last_error = TorrentProviderError("BitSearch 响应超时", 504)
                last_error.__cause__ = exc
            except httpx.RequestError as exc:
                last_error = TorrentProviderError("无法连接 BitSearch")
                last_error.__cause__ = exc
            else:
                if response.status_code == 429:
                    raise TorrentProviderError("BitSearch 请求频繁", 429)
                if response.status_code == 403:
                    raise TorrentProviderError("BitSearch 拒绝请求", 503)
                if response.is_success:
                    return parse_bitsearch_html(response.text, self.parameters.provider_limit)
                last_error = TorrentProviderError(f"BitSearch 返回 HTTP {response.status_code}")
            if attempt < self.parameters.provider_retries:
                await asyncio.sleep(0.3 * (attempt + 1))
        raise last_error


def parse_bitsearch_html(html: str, limit: int = 20) -> list[dict[str, Any]]:
    """解析 BitSearch 卡片页面为 Torrent-Api-py 兼容字典。"""
    soup = BeautifulSoup(html, "html.parser")
    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for title_link in soup.select('a[href^="/torrent/"]'):
        title = title_link.get_text(" ", strip=True)
        card = title_link.parent
        magnet_link = None
        while card is not None and getattr(card, "name", None) != "body":
            magnet_link = card.select_one('a[href^="magnet:"]') if hasattr(card, "select_one") else None
            if magnet_link is not None:
                break
            card = card.parent
        magnet = str(magnet_link.get("href") or "") if magnet_link is not None else ""
        info_hash = _magnet_hash(magnet)
        if not title or info_hash is None or info_hash in seen or card is None:
            continue
        seen.add(info_hash)
        card_text = " ".join(card.stripped_strings)
        size = re.search(r"\b([0-9]+(?:\.[0-9]+)?\s*[KMGT]i?B)\b", card_text, re.I)
        seeders = re.search(r"\b([0-9,]+)\s+seeders\b", card_text, re.I)
        leechers = re.search(r"\b([0-9,]+)\s+leechers\b", card_text, re.I)
        date = re.search(r"\b\d{1,2}/\d{1,2}/\d{4}\b", card_text)
        items.append({
            "name": title,
            "hash": info_hash,
            "magnet": magnet,
            "size": size.group(1) if size else "0 B",
            "seeders": seeders.group(1).replace(",", "") if seeders else "0",
            "leechers": leechers.group(1).replace(",", "") if leechers else "0",
            "date": date.group(0) if date else None,
            "source": "bitsearch.eu",
        })
        if len(items) >= max(1, limit):
            break
    return items


def _normalize_item(raw: dict[str, Any], category_hint: str, source: str) -> dict[str, Any] | None:
    """把分站原始字段转换成包对外返回的统一种子字典。"""
    title = str(raw.get("name") or raw.get("title") or "").strip()
    magnet = str(raw.get("magnet") or raw.get("magnet_url") or "")
    info_hash = _direct_hash(str(raw.get("hash") or raw.get("info_hash") or "")) or _magnet_hash(magnet)
    if not title or len(title) > 500 or info_hash is None:
        return None
    category, quality = _classify(title)
    if category_hint in {"movie", "hd_movie"}:
        category = "movie"
    elif category_hint in {"tv", "hd_tv"}:
        category = "tv"
    if category_hint in {"hd_movie", "hd_tv"} and quality is None:
        quality = "1080p"
    return {
        "id": info_hash,
        "info_hash": info_hash,
        "title": title,
        "magnet_url": magnet if _magnet_hash(magnet) == info_hash else (
            f"magnet:?xt=urn:btih:{info_hash}&dn={quote(title, safe='')}"
        ),
        "size_bytes": _size_bytes(raw.get("size", raw.get("size_bytes"))),
        "seeders": _integer(raw.get("seeders")),
        "leechers": _integer(raw.get("leechers")),
        "file_count": _integer(raw.get("file_count", raw.get("num_files"))),
        "added_at": _date_text(raw.get("date", raw.get("added_at"))),
        "category": category,
        "quality": quality,
        "source": str(raw.get("source") or source),
    }


def _classify(title: str) -> tuple[str, str | None]:
    """根据标题推断电影/剧集和常见画质。"""
    lowered = title.casefold()
    quality = "4k" if "2160p" in lowered or re.search(r"\b4k\b", lowered) else None
    quality = "1080p" if quality is None and "1080p" in lowered else quality
    quality = "720p" if quality is None and "720p" in lowered else quality
    if re.search(r"(?:\bS\d{1,2}E\d{1,3}\b|\bSeason[ ._-]*\d+\b|第\s*\d+\s*[集季])", title, re.I):
        return "tv", quality
    if re.search(r"(?:19|20)\d{2}", title):
        return "movie", quality
    return "video", quality


def _matches_query(title: str, query: str) -> bool:
    """中文查询要求连续匹配，英文查询信任远程站点分词。"""
    if re.search(r"[\u3400-\u9fff]", query) is None:
        return True
    normalized_query = query.casefold().strip()
    normalized_title = title.casefold()
    return bool(normalized_query and normalized_query in normalized_title)


def _direct_hash(value: str) -> str | None:
    """校验并规范化直接提供的十六进制 info hash。"""
    return value.lower() if HASH_PATTERN.fullmatch(value) else None


def _magnet_hash(value: str) -> str | None:
    """从 magnet 中提取十六进制或 Base32 BTIH。"""
    match = MAGNET_HASH_PATTERN.search(value)
    if match is None:
        return None
    encoded = match.group(1)
    if len(encoded) == 40:
        return encoded.lower()
    try:
        return base64.b32decode(encoded.upper()).hex()
    except (ValueError, TypeError):
        return None


def _integer(value: Any) -> int:
    """把远程数值转换成非负整数。"""
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _size_bytes(value: Any) -> int:
    """把字节数或可读大小转换成整数字节。"""
    if isinstance(value, (int, float)):
        return max(0, int(value))
    match = SIZE_PATTERN.fullmatch(str(value or "").strip())
    if match is None:
        return 0
    number = float(match.group(1))
    unit = (match.group(2) or "B").upper().replace("IB", "B")
    power = {"B": 0, "KB": 1, "MB": 2, "GB": 3, "TB": 4}.get(unit, 0)
    return int(number * 1024**power)


def _date_text(value: Any) -> str | None:
    """把来源日期规范化为可排序文本。"""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    text = str(value).strip()
    return text or None
