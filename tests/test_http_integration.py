"""验证模型适配器和远程 Provider 的 HTTP 调用链。"""

import asyncio
import json
from types import SimpleNamespace

import httpx

from ai_torrent_search import AITorrentSearch, DashScopeChatModel, OpenAICompatibleChatModel, SearchIntent
from ai_torrent_search.provider import parse_bitsearch_html


HASH = "a" * 40


def intent_json():
    """返回适配器测试使用的最小意图 JSON。"""
    return SearchIntent(primary_query="Sintel", queries=["Sintel"]).model_dump_json()


def test_openai_compatible_adapter_and_provider():
    """OpenAI 兼容适配器应与默认 Provider 组合完成搜索。"""
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        """分别模拟聊天模型接口和 Torrent-Api-py。"""
        if request.url.path.endswith("/chat/completions"):
            captured["authorization"] = request.headers.get("authorization")
            captured["model_body"] = json.loads(request.content)
            plan = SearchIntent(category="movie", primary_query="Sintel", required_terms=["Sintel"],
                                quality="1080p", sort="seeders", queries=["Sintel"])
            return httpx.Response(200, json={"choices": [{"message": {"content": plan.model_dump_json()}}]})
        if request.url.path.endswith("/api/v1/search"):
            captured["provider_params"] = dict(request.url.params)
            return httpx.Response(200, json={"data": [{"name": "Sintel 2010 1080p", "hash": HASH,
                                                        "size": "1.5 GiB", "seeders": "12", "leechers": "3"}]})
        return httpx.Response(404)

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            model = OpenAICompatibleChatModel(api_key="test-secret-key", base_url="https://model.example/v1",
                                               model="qwen-plus", http_client=client)
            searcher = AITorrentSearch(model, {"provider_url": "https://torrent.example",
                                               "provider_sites": ["nyaasi"], "max_expanded_queries": 1,
                                               "page_size": 15}, http_client=client)
            result = await searcher.search("找 Sintel 1080p 电影")
        assert result.total == 1
        assert captured["authorization"] == "Bearer test-secret-key"
        assert captured["model_body"]["messages"][-1]["role"] == "user"
        assert captured["provider_params"]["site"] == "nyaasi"
        assert "test-secret-key" not in repr(model)

    asyncio.run(scenario())


def test_dashscope_native_sdk_adapter():
    """DashScope 适配器应调用原生 SDK 并隐藏 API Key。"""
    captured = {}

    class Generation:
        """模拟 dashscope.Generation。"""

        @classmethod
        def call(cls, **kwargs):
            """记录调用并返回原生 SDK 风格响应。"""
            captured.update(kwargs)
            return {"status_code": 200, "output": {"choices": [{"message": {"content": intent_json()}}]}}

    async def scenario():
        model = DashScopeChatModel(api_key="native-secret", model="qwen-plus",
                                   sdk=SimpleNamespace(Generation=Generation))
        result = await model.ainvoke([("system", "规则"), ("human", "Sintel")])
        assert json.loads(result)["primary_query"] == "Sintel"
        assert captured["messages"][-1]["role"] == "user"
        assert "native-secret" not in repr(model)

    asyncio.run(scenario())


def test_bitsearch_parser_extracts_card_metadata():
    """HTML 兼容解析器应提取 magnet、大小和活跃数。"""
    html = f"""<html><body><article><a href="/torrent/example">Sintel 2010 1080p</a>
      <a href="magnet:?xt=urn:btih:{HASH}&amp;dn=Sintel">Magnet</a>
      <span>1.5 GiB</span><span>12 seeders</span><span>3 leechers</span><time>01/02/2025</time>
    </article></body></html>"""
    items = parse_bitsearch_html(html)
    assert len(items) == 1
    assert items[0]["hash"] == HASH
    assert items[0]["size"] == "1.5 GiB"
    assert items[0]["seeders"] == "12"
