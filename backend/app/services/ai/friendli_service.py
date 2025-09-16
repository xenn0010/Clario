"""
FriendliAI service wrapper for Clario
Provides chat/text completions used by agents and services
"""

from typing import Any, Dict, List, Optional, Union, AsyncIterator
import asyncio
import logging
import os

try:
    from friendli import Friendli, AsyncFriendli
except Exception:  # pragma: no cover - allow running without package in dev
    Friendli = None
    AsyncFriendli = None

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger("friendli")


class FriendliService:
    """FriendliAI client singleton service"""

    _client: Optional[Friendli] = None
    _async_client: Optional[Any] = None
    _openai_client: Optional[Any] = None
    _initialized: bool = False

    @classmethod
    async def initialize(cls) -> None:
        """Initialize Friendli clients"""
        if cls._initialized:
            return
        # Prefer OpenAI-compatible client with Friendli Serverless API
        try:
            from openai import OpenAI as OpenAIClient  # type: ignore
            token = getattr(settings, 'FRIENDLIAI_TOKEN', None) or os.getenv('FRIENDLI_TOKEN')
            base_url = getattr(settings, 'FRIENDLIAI_BASE_URL', None) or os.getenv('FRIENDLI_BASE_URL') or 'https://api.friendli.ai/serverless/v1'
            if token:
                cls._openai_client = OpenAIClient(api_key=token, base_url=base_url)
                logger.info("Friendli OpenAI-compatible client initialized")
        except Exception as e:
            logger.warning(f"OpenAI-compatible Friendli client not available: {e}")

        # Fallback to Friendli official SDK if available
        if cls._openai_client is None:
            if Friendli is None or AsyncFriendli is None:
                logger.warning("Friendli SDK not available; running in mock mode")
                cls._initialized = True
                return
            try:
                token = getattr(settings, 'FRIENDLIAI_TOKEN', None) or os.getenv('FRIENDLI_TOKEN')
                cls._client = Friendli(
                    token=token,
                    base_url=getattr(settings, "FRIENDLIAI_BASE_URL", "https://api.friendli.ai/v1"),
                    timeout=getattr(settings, "FRIENDLIAI_TIMEOUT", 120),
                )
                cls._async_client = AsyncFriendli(
                    token=token,
                    base_url=getattr(settings, "FRIENDLIAI_BASE_URL", "https://api.friendli.ai/v1"),
                    timeout=getattr(settings, "FRIENDLIAI_TIMEOUT", 120),
                )
                logger.info("Friendli SDK client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Friendli SDK: {e}")
        cls._initialized = True

    @classmethod
    async def cleanup(cls) -> None:
        """Cleanup resources (noop for Friendli)"""
        cls._client = None
        cls._async_client = None
        cls._initialized = False

    @classmethod
    async def chat_completion(
        cls,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        stream: bool = False,
    ) -> Union[str, AsyncIterator[str]]:
        """Generate chat completion. Returns text or async iterator if stream=True"""
        await cls.initialize()

        model = model or getattr(settings, "FRIENDLIAI_MODEL", "meta-llama-3.1-70b-instruct")
        temperature = temperature if temperature is not None else getattr(settings, "FRIENDLIAI_TEMPERATURE", 0.7)
        max_tokens = max_tokens if max_tokens is not None else getattr(settings, "FRIENDLIAI_MAX_TOKENS", 2000)
        top_p = top_p if top_p is not None else getattr(settings, "FRIENDLIAI_TOP_P", 0.9)

        # OpenAI-compatible route via Friendli Serverless API
        if cls._openai_client is not None:
            try:
                if stream:
                    # The OpenAI Python SDK supports streaming; simplify by concatenating
                    resp = await asyncio.to_thread(
                        cls._openai_client.chat.completions.create,
                        model=model,
                        messages=messages,
                        stream=False,
                    )
                else:
                    resp = await asyncio.to_thread(
                        cls._openai_client.chat.completions.create,
                        model=model,
                        messages=messages,
                    )
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:
                logger.error(f"Friendli (OpenAI-compatible) call failed: {e}")

        # Friendli SDK route or mock
        if cls._client is None:
            text = cls._mock_response_from_messages(messages)
            if stream:
                async def _gen():
                    yield text
                return _gen()
            return text

        try:
            if stream:
                # Friendli streaming iterator (synchronous client yields chunks)
                def _iter():
                    stream_obj = cls._client.completions.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        stream=True,
                    )
                    for chunk in stream_obj:
                        if chunk.choices and chunk.choices[0].delta.content:
                            yield chunk.choices[0].delta.content

                async def _agen():
                    for part in await asyncio.to_thread(lambda: list(_iter())):
                        yield part

                return _agen()

            # Non-streaming
            resp = await asyncio.to_thread(
                cls._client.completions.create,
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stream=False,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            logger.error(f"Friendli chat completion failed: {e}")
            return cls._mock_response_from_messages(messages)

    @staticmethod
    def _mock_response_from_messages(messages: List[Dict[str, str]]) -> str:
        last_user = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                last_user = m.get("content", "")
                break
        return f"[MOCK FRIENDLI] Response to: {last_user[:200]}"


