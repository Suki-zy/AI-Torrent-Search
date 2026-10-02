"""把 LangChain 风格聊天模型的输出转换为搜索意图。"""

import json
import logging
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from pydantic import ValidationError

from ai_torrent_search.models import AITorrentSearchError, SearchIntent


logger = logging.getLogger("ai_torrent_search.analyzer")


@runtime_checkable
class ChatModel(Protocol):
    """定义搜索器接受的 LangChain 风格异步模型接口。"""

    async def ainvoke(self, messages: Any, **kwargs: Any) -> Any:
        """接收聊天消息并异步返回模型结果。"""
        ...


class StructuredIntentAnalyzer:
    """用调用方提供的聊天模型生成并校验影视搜索意图。"""

    def __init__(self, model: ChatModel):
        """保存模型实例；模型的创建与关闭均由调用方负责。"""
        if not callable(getattr(model, "ainvoke", None)):
            raise TypeError("model 必须实现异步 ainvoke(messages, **kwargs)")
        self.model = model

    async def analyze(self, query: str) -> SearchIntent:
        """优先调用结构化输出能力，否则解析普通消息中的 JSON。"""
        messages = _build_messages(query)
        try:
            structured_factory = getattr(self.model, "with_structured_output", None)
            if callable(structured_factory):
                try:
                    runnable = structured_factory(SearchIntent)
                    result = await runnable.ainvoke(messages)
                    return _validate_result(result)
                except (NotImplementedError, AttributeError):
                    logger.debug("structured_output_unsupported model=%s", type(self.model).__name__)
            result = await self.model.ainvoke(messages)
            return _validate_result(result)
        except AITorrentSearchError:
            raise
        except Exception as exc:
            logger.warning("model_intent_failed model=%s error_type=%s", type(self.model).__name__, type(exc).__name__)
            raise AITorrentSearchError(f"模型意图识别失败：{exc}") from exc


def _build_messages(query: str) -> list[tuple[str, str]]:
    """构造可被 LangChain 聊天模型直接接收的角色消息。"""
    schema = json.dumps(SearchIntent.model_json_schema(), ensure_ascii=False)
    system_prompt = (
        "你是影视种子搜索意图解析器。只分析用户想搜索什么，不回答问题，也不生成链接。"
        "提取作品名、简繁体或外文别名、电影/剧集、年份、季数、画质、排除词和排序。"
        "queries 必须包含 primary_query，并只生成少量高相关查询。"
        f"严格返回符合以下 JSON Schema 的 JSON：{schema}"
    )
    return [("system", system_prompt), ("human", query)]


def _validate_result(result: Any) -> SearchIntent:
    """兼容 Pydantic 对象、字典、AIMessage 和文本内容块。"""
    if isinstance(result, SearchIntent):
        return result
    if isinstance(result, Mapping):
        try:
            return SearchIntent.model_validate(result)
        except ValidationError:
            content = result.get("content")
    else:
        content = getattr(result, "content", result)
    text = _content_to_text(content)
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        return SearchIntent.model_validate_json(text)
    except (ValidationError, ValueError, TypeError) as exc:
        raise AITorrentSearchError("模型返回了无法识别的搜索计划") from exc


def _content_to_text(content: Any) -> str:
    """把常见 LangChain/OpenAI 内容格式规范化为文本。"""
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes, bytearray)):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, Mapping) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "".join(parts)
    raise AITorrentSearchError("模型返回了无法识别的搜索计划")
