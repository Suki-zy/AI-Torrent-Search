"""定义 AI 种子搜索包的公开配置、结果和异常模型。"""

from typing import Any, Literal

from pydantic import BaseModel, Field, SecretStr, field_validator


class QueryParameters(BaseModel):
    """保存实例默认使用的检索、排序和远程来源参数。"""

    # auto 表示根据模型识别出的影视类型和画质选择 SDK 分类。
    category: Literal["auto", "video", "movie", "hd_movie", "tv", "hd_tv", "3d"] = "auto"
    # auto 表示使用模型意图；relevance 只影响合并后的本地重排。
    sort: Literal["auto", "relevance", "seeders", "recent", "size"] = "auto"
    # 每个扩展查询最多从底层 SDK 取出的候选数量。
    page_size: int = Field(default=50, ge=1, le=100)
    # 多查询合并、过滤和去重后最终返回的最大数量。
    result_limit: int = Field(default=30, ge=1, le=200)
    # 限制模型扩展词数量，避免一次请求扇出过多远程调用。
    max_expanded_queries: int = Field(default=4, ge=1, le=8)
    # required_terms_filter 控制是否严格过滤缺少模型必选实体的标题。
    required_terms_filter: bool = True
    # 默认远程来源沿用项目当前使用的 Torrent-Api-py 服务。
    provider_url: str = "https://torrent-api-py-nx0x.onrender.com"
    provider_timeout_seconds: float = Field(default=30, ge=5, le=120)
    provider_limit: int = Field(default=10, ge=1, le=50)
    provider_retries: int = Field(default=1, ge=0, le=5)
    provider_sites: tuple[str, ...] = Field(default=("nyaasi", "1337x", "bitsearch"), min_length=1)
    provider_api_key: SecretStr | None = None
    bitsearch_url: str = "https://bitsearch.eu"
    bitsearch_pages: int = Field(default=5, ge=1, le=10)

    @field_validator("provider_url", "bitsearch_url")
    @classmethod
    def normalize_provider_url(cls, value: str) -> str:
        """清理种子来源地址并拒绝空值。"""
        normalized = value.strip().rstrip("/")
        if not normalized:
            raise ValueError("种子来源地址不能为空")
        return normalized

    @field_validator("provider_sites")
    @classmethod
    def normalize_sites(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        """规范化站点名称并保持原顺序去重。"""
        normalized = tuple(dict.fromkeys(site.strip().lower() for site in value if site.strip()))
        if not normalized:
            raise ValueError("至少需要一个种子搜索站点")
        return normalized


class SearchIntent(BaseModel):
    """描述模型从自然语言中提取的影视搜索意图。"""

    # 独立包只检索种子，因此不需要 cloud/all 来源字段。
    category: Literal["movie", "tv", "other"] = "other"
    primary_query: str = Field(min_length=1, max_length=80)
    aliases: list[str] = Field(default_factory=list, max_length=8)
    required_terms: list[str] = Field(default_factory=list, max_length=8)
    excluded_terms: list[str] = Field(default_factory=list, max_length=8)
    year: int | None = Field(default=None, ge=1800, le=2100)
    season: int | None = Field(default=None, ge=1, le=200)
    quality: Literal["4k", "1080p", "720p"] | None = None
    sort: Literal["relevance", "seeders", "latest", "size"] = "relevance"
    queries: list[str] = Field(default_factory=list, max_length=8)

class AITorrentSearchResult(BaseModel):
    """描述一次 AI 种子搜索的结构化结果。"""

    # intent 是模型经过 Pydantic 校验后的完整意图。
    intent: dict[str, Any]
    # expanded_queries 按实际执行顺序列出规范化后的检索词。
    expanded_queries: list[str]
    # items 保存去重、过滤和重排后的种子字典。
    items: list[dict[str, Any]]
    total: int
    warnings: list[str] = Field(default_factory=list)


class AITorrentSearchError(Exception):
    """表示模型分析或全部种子查询无法完成。"""

    def __init__(self, message: str, status: int = 502):
        """保存可供 API 适配层复用的消息和状态码。"""
        super().__init__(message)
        self.status = status
