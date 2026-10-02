"""提供无需 LangChain 依赖的聊天模型适配器。"""

import asyncio
from collections.abc import Mapping, Sequence
from typing import Any

import httpx
from pydantic import SecretStr

from ai_torrent_search.models import AITorrentSearchError


class OpenAICompatibleChatModel:
    """把 OpenAI 兼容 Chat Completions 接口适配为 ``ainvoke``。"""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        temperature: float = 0.1,
        max_tokens: int = 1000,
        timeout_seconds: float = 30,
        extra_body: Mapping[str, Any] | None = None,
        http_client: httpx.AsyncClient | None = None,
    ):
        """保存连接参数；外部传入的 HTTP 客户端不会被自动关闭。"""
        if not api_key.strip() or not model.strip() or not base_url.strip():
            raise ValueError("api_key、model 和 base_url 不能为空")
        self.api_key = SecretStr(api_key)
        self.model = model.strip()
        self.base_url = base_url.strip().rstrip("/")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.extra_body = dict(extra_body or {})
        self._client = http_client
        self._owns_client = http_client is None

    async def ainvoke(self, messages: Any, **kwargs: Any) -> str:
        """调用兼容接口，并返回第一条 assistant 文本。"""
        if self._client is None:
            self._client = httpx.AsyncClient()
        payload = {
            "model": self.model,
            "messages": [_message_to_dict(message) for message in messages],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
            **self.extra_body,
            **kwargs,
        }
        try:
            response = await self._client.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key.get_secret_value()}"},
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except httpx.TimeoutException as exc:
            raise AITorrentSearchError("模型意图识别超时", 504) from exc
        except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
            raise AITorrentSearchError(f"OpenAI 兼容模型调用失败：{exc}") from exc

    async def aclose(self) -> None:
        """关闭适配器自行创建的 HTTP 客户端。"""
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

    def __repr__(self) -> str:
        """返回不包含明文 API Key 的调试表示。"""
        return f"OpenAICompatibleChatModel(model={self.model!r}, base_url={self.base_url!r}, api_key=SecretStr('**********'))"


class DashScopeChatModel:
    """把阿里云 DashScope 原生 Python SDK 适配为 ``ainvoke``。"""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "qwen-plus",
        temperature: float = 0.1,
        max_tokens: int = 1000,
        enable_thinking: bool = False,
        sdk: Any | None = None,
    ):
        """保存原生 SDK 参数；``sdk`` 允许测试或高级用户注入兼容实现。"""
        if not api_key.strip() or not model.strip():
            raise ValueError("api_key 和 model 不能为空")
        self.api_key = SecretStr(api_key)
        self.model = model.strip()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        self._sdk = sdk

    async def ainvoke(self, messages: Any, **kwargs: Any) -> str:
        """在线程中执行同步 DashScope SDK，并返回 assistant 文本。"""
        sdk = self._sdk
        if sdk is None:
            try:
                import dashscope as sdk
            except ImportError as exc:
                raise AITorrentSearchError("请安装 ai-torrent-search[qwen] 以使用 DashScopeChatModel") from exc
        payload = {
            "api_key": self.api_key.get_secret_value(),
            "model": self.model,
            "messages": [_message_to_dict(message) for message in messages],
            "result_format": "message",
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "enable_thinking": self.enable_thinking,
            **kwargs,
        }
        try:
            response = await asyncio.to_thread(sdk.Generation.call, **payload)
            status_code = _value(response, "status_code")
            if status_code is not None and int(status_code) >= 400:
                raise AITorrentSearchError(f"DashScope 模型调用失败，状态码 {status_code}")
            output = _value(response, "output")
            choices = _value(output, "choices")
            message = _value(choices[0], "message")
            return _value(message, "content")
        except AITorrentSearchError:
            raise
        except (AttributeError, KeyError, IndexError, TypeError, ValueError) as exc:
            raise AITorrentSearchError(f"DashScope 模型返回格式异常：{exc}") from exc

    def __repr__(self) -> str:
        """返回不包含明文 API Key 的调试表示。"""
        return f"DashScopeChatModel(model={self.model!r}, api_key=SecretStr('**********'))"


def _message_to_dict(message: Any) -> dict[str, Any]:
    """把元组、字典或 LangChain BaseMessage 转换为角色消息。"""
    if isinstance(message, Mapping):
        return {"role": str(message["role"]), "content": message["content"]}
    if isinstance(message, Sequence) and not isinstance(message, (str, bytes)) and len(message) == 2:
        role, content = message
        return {"role": "user" if role == "human" else str(role), "content": content}
    role = getattr(message, "type", getattr(message, "role", None))
    content = getattr(message, "content", None)
    if role is None or content is None:
        raise TypeError("无法识别的聊天消息格式")
    return {"role": "user" if role == "human" else str(role), "content": content}


def _value(value: Any, key: str) -> Any:
    """同时读取字典键和 SDK 响应对象属性。"""
    return value.get(key) if isinstance(value, Mapping) else getattr(value, key)
