"""验证模型注入、结构化意图和结果合并。"""

import asyncio

import pytest
from pydantic import ValidationError

from ai_torrent_search import AITorrentSearch, AITorrentSearchError, QueryParameters, SearchIntent, TorrentProviderError


def intent() -> SearchIntent:
    """生成包含别名、画质和过滤条件的稳定测试意图。"""
    return SearchIntent(category="movie", primary_query="Sintel", aliases=["辛特尔"], required_terms=["Sintel"],
                        excluded_terms=["预告"], year=2025, quality="1080p", sort="seeders",
                        queries=["Sintel 2025 1080p"])


class FakeRunnable:
    """模拟 LangChain 结构化输出 runnable。"""

    async def ainvoke(self, messages):
        """记录消息并返回已经校验的意图。"""
        self.messages = messages
        return intent()


class FakeLangChainModel:
    """模拟带 with_structured_output 的 LangChain 模型。"""

    def __init__(self):
        """创建供断言使用的 runnable。"""
        self.runnable = FakeRunnable()
        self.schema = None

    def with_structured_output(self, schema):
        """记录结构化 Schema 并返回 runnable。"""
        self.schema = schema
        return self.runnable

    async def ainvoke(self, messages):
        """该分支不应在结构化输出可用时调用。"""
        raise AssertionError("unexpected plain ainvoke")


class PlainMessage:
    """模拟 LangChain AIMessage 的 content 属性。"""

    def __init__(self, content):
        """保存文本或内容块。"""
        self.content = content


class PlainModel:
    """模拟只实现 ainvoke 的自定义聊天模型。"""

    async def ainvoke(self, messages, **kwargs):
        """在 Markdown JSON 代码块中返回搜索意图。"""
        return PlainMessage(f"```json\n{intent().model_dump_json()}\n```")


class FakeProvider:
    """记录扩展查询并返回重复、相关和排除项。"""

    def __init__(self, fail_all=False):
        """初始化调用记录和可选的全失败开关。"""
        self.calls = []
        self.fail_all = fail_all

    async def search(self, query, category, limit):
        """返回稳定候选，或模拟底层 Provider 故障。"""
        self.calls.append((query, category, limit))
        if self.fail_all:
            raise TorrentProviderError("provider unavailable", 504)
        return [
            {"title": "Sintel 2025 1080p 测试电影", "info_hash": "a" * 40, "seeders": 8},
            {"title": "Sintel 2025 1080p 预告", "info_hash": "b" * 40, "seeders": 999},
            {"title": "完全无关电影", "info_hash": "c" * 40, "seeders": 999},
        ]


def test_langchain_style_model_expands_deduplicates_and_ranks():
    """模型实例应通过结构化接口驱动扩展、过滤、去重和排序。"""

    async def scenario():
        model = FakeLangChainModel()
        provider = FakeProvider()
        searcher = AITorrentSearch(model, {"max_expanded_queries": 3, "result_limit": 5, "page_size": 40}, provider=provider)
        result = await searcher.search("找 Sintel 2025 年 1080p 电影，按做种数排序")
        assert model.schema is SearchIntent
        assert model.runnable.messages[-1][0] == "human"
        assert result.expanded_queries == ["Sintel", "Sintel 2025 1080p", "辛特尔"]
        assert result.total == 1
        assert result.items[0]["info_hash"] == "a" * 40
        assert len(provider.calls) == 3
        assert all(call[1] == "hd_movie" and call[2] == 40 for call in provider.calls)
        await searcher.close()

    asyncio.run(scenario())


def test_plain_ainvoke_model_is_supported():
    """不依赖 LangChain 的模型也可通过 ainvoke 和 JSON 文本接入。"""

    async def scenario():
        result = await AITorrentSearch(PlainModel(), provider=FakeProvider()).search("找 Sintel")
        assert result.intent["primary_query"] == "Sintel"

    asyncio.run(scenario())


def test_invalid_model_and_query_configuration_are_rejected():
    """初始化应拒绝缺少 ainvoke 的模型和越界查询参数。"""
    with pytest.raises(TypeError):
        AITorrentSearch(object(), provider=FakeProvider())
    with pytest.raises(ValidationError):
        QueryParameters(result_limit=0)
    with pytest.raises(ValidationError):
        QueryParameters(provider_sites=())


def test_all_query_failures_raise_package_error():
    """全部扩展查询失败时不应返回容易误解的空结果。"""

    async def scenario():
        searcher = AITorrentSearch(PlainModel(), provider=FakeProvider(fail_all=True))
        with pytest.raises(AITorrentSearchError) as error:
            await searcher.search("找 Sintel")
        assert error.value.status == 504
        assert "全部失败" in str(error.value)

    asyncio.run(scenario())
