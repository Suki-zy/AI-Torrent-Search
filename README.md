# AI Torrent Search

一个可嵌入 Python 应用和 Agent 工作流的异步 AI 种子元数据搜索包。调用方创建模型实例，`AITorrentSearch` 通过统一的 LangChain 风格 `ainvoke(messages)` 接口使用它，因此不绑定模型厂商。

模型只负责识别影视实体、扩展关键词和确定排序意图。真实结果由可替换的种子索引 Provider 返回，再由本地代码完成实体过滤、info hash 去重和确定性重排。

## 安装

```bash
python -m pip install -e .
python -m pip install -e '.[qwen]'       # DashScope 原生 SDK
python -m pip install -e '.[langchain]'  # LangChain 模型
python -m pip install -e '.[dev]'        # 测试
```

## 统一模型接口

搜索器接受任何实现以下接口的对象：

```python
async def ainvoke(messages, **kwargs): ...
```

如果模型还实现了 LangChain 的 `with_structured_output(SearchIntent)`，包会优先使用结构化输出；否则解析 `ainvoke` 返回的 JSON 字符串、字典、`AIMessage.content` 或文本内容块。模型实例及其连接由调用方创建和关闭。

### LangChain + OpenAI

```python
from langchain_openai import ChatOpenAI
from ai_torrent_search import AITorrentSearch

model = ChatOpenAI(model="gpt-4.1-mini", api_key="...")
searcher = AITorrentSearch(model, {"result_limit": 30})
result = await searcher.search("找 2024 年 1080p 科幻电影，做种优先")
```

### LangChain + 通义千问

```python
from langchain_community.chat_models.tongyi import ChatTongyi
from ai_torrent_search import AITorrentSearch

model = ChatTongyi(model="qwen-plus", dashscope_api_key="...")
searcher = AITorrentSearch(model)
result = await searcher.search("搜索 Sintel，按相关度排序")
```

### OpenAI 兼容接口

内置轻量适配器可用于 OpenAI、百炼兼容模式以及本地兼容服务，无需安装 LangChain：

```python
import os
from ai_torrent_search import AITorrentSearch, OpenAICompatibleChatModel

model = OpenAICompatibleChatModel(
    api_key=os.environ["DASHSCOPE_API_KEY"],
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    model="qwen-plus",
    extra_body={"enable_thinking": False},
)

async with AITorrentSearch(model, {"provider_sites": ["nyaasi", "1337x", "bitsearch"]}) as searcher:
    result = await searcher.search("找 2024 年的 1080p 科幻电影，做种优先")

await model.aclose()
```

### 通义千问原生 SDK

```python
import os
from ai_torrent_search import AITorrentSearch, DashScopeChatModel

model = DashScopeChatModel(api_key=os.environ["DASHSCOPE_API_KEY"], model="qwen-plus")
searcher = AITorrentSearch(model)
result = await searcher.search("找 Sintel 相关影视资源")
```

## 查询配置

`QueryParameters` 可以作为第二个参数传入配置对象或字典，支持：

- `category`：`auto`、`video`、`movie`、`hd_movie`、`tv`、`hd_tv`、`3d`。
- `sort`：`auto`、`relevance`、`seeders`、`recent`、`size`。
- `result_limit`、`page_size`、`max_expanded_queries`。
- `required_terms_filter`：是否严格要求标题包含模型识别出的核心实体。
- Torrent-Api-py 地址、站点、超时、重试、每站条数和 BitSearch 页数。

`search()` 返回 `AITorrentSearchResult`，包含经过校验的 `intent`、实际执行的 `expanded_queries`、合并后的 `items`、`total` 和分支失败 `warnings`。

## 自定义模型和来源

自定义 Provider 只需实现：

```python
async def search(query: str, category: str, limit: int) -> list[dict]: ...
```

所有注入对象的生命周期均由调用方管理。

## 安全与范围

- API Key 只保存在调用方模型或内置适配器中；适配器的调试表示会隐藏密钥。
- 包只检索公开索引元数据，不下载或托管文件。
- 搜索结果可能失效、不准确或涉及第三方权利。请只访问你有权使用的内容。

## 测试与构建

```bash
pytest -q
python -m build
```
