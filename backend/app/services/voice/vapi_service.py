"""
VAPI voice service integration for Clario
Provides assistant creation and webhook verification helpers
"""

from typing import Any, Dict, Optional
import hmac
import hashlib
import json
import logging
import httpx

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger("vapi")


class VapiService:
    """VAPI service wrapper"""

    def __init__(self):
        self.base_url = getattr(settings, "VAPI_BASE_URL", "https://api.vapi.ai/v1")
        self.api_key = getattr(settings, "VAPI_API_KEY", None)
        self.webhook_secret = getattr(settings, "VAPI_WEBHOOK_SECRET", None)
        self.client = httpx.AsyncClient(timeout=30)

    async def create_assistant(self, name: str, system_message: str) -> Optional[str]:
        if not self.api_key:
            logger.warning("VAPI_API_KEY not set; skipping assistant creation")
            return None
        try:
            payload = {
                "name": name,
                "systemMessage": system_message,
                "voice": getattr(settings, "VAPI_DEFAULT_VOICE", "nova"),
                "firstMessage": getattr(settings, "VAPI_FIRST_MESSAGE", "Hello!"),
                "silenceTimeoutSeconds": int(getattr(settings, "VAPI_SILENCE_TIMEOUT", 30)),
                "maxDurationSeconds": int(getattr(settings, "VAPI_MAX_DURATION", 900)),
                "webhookUrl": getattr(settings, "VAPI_WEBHOOK_URL", None),
            }
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            resp = await self.client.post(f"{self.base_url}/assistants", json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            return data.get("id")
        except Exception as e:
            logger.error(f"Failed to create VAPI assistant: {e}")
            return None

    def verify_webhook(self, body: bytes, signature: str) -> bool:
        if not self.webhook_secret:
            return False
        try:
            computed = hmac.new(self.webhook_secret.encode(), body, hashlib.sha256).hexdigest()
            return hmac.compare_digest(computed, signature)
        except Exception:
            return False

    async def close(self):
        try:
            await self.client.aclose()
        except Exception:
            pass


