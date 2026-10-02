"""公开 AI 种子搜索实例、配置和结果模型。"""

from ai_torrent_search.analyzer import ChatModel, StructuredIntentAnalyzer
from ai_torrent_search.chat_models import DashScopeChatModel, OpenAICompatibleChatModel
from ai_torrent_search.client import AITorrentSearch, IntentAnalyzer
from ai_torrent_search.models import (
    AITorrentSearchError,
    AITorrentSearchResult,
    QueryParameters,
    SearchIntent,
)
from ai_torrent_search.provider import TorrentApiPyProvider, TorrentProvider, TorrentProviderError

__all__ = [
    "AITorrentSearch",
    "AITorrentSearchError",
    "AITorrentSearchResult",
    "ChatModel",
    "DashScopeChatModel",
    "IntentAnalyzer",
    "OpenAICompatibleChatModel",
    "QueryParameters",
    "SearchIntent",
    "StructuredIntentAnalyzer",
    "TorrentApiPyProvider",
    "TorrentProvider",
    "TorrentProviderError",
]
