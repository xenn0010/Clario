"""
VAPI voice endpoints: webhook and assistant bootstrap
"""

from fastapi import APIRouter, Request, Header, HTTPException
from typing import Any, Dict, Optional

from pydantic import BaseModel

from app.services.voice.vapi_service import VapiService
from app.services.voice.decision_voice_orchestrator import get_decision_voice_orchestrator
from app.core.logging import get_logger

logger = get_logger("vapi_api")

router = APIRouter()


class DecisionVoiceRequest(BaseModel):
    organization_id: str
    agenda_hint: Optional[str] = None
    include_similar: bool = True


@router.post("/webhook")
async def vapi_webhook(request: Request, x_vapi_signature: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    body = await request.body()
    service = VapiService()
    try:
        if not service.verify_webhook(body, x_vapi_signature or ""):
            raise HTTPException(status_code=401, detail="Invalid webhook signature")

        payload = await request.json()
        event_type = payload.get("type", "event")

        # Handle basic events (call.started, call.ended, message, etc.)
        logger.info("VAPI webhook event", event_type=event_type)

        # For decision Q&A, expect payload to include decision_id or context
        # Here we simply acknowledge; extension: fetch decision by id and reply
        return {"received": True}
    finally:
        await service.close()


@router.post("/assistant/bootstrap")
async def create_assistant() -> Dict[str, Any]:
    service = VapiService()
    try:
        from app.core.config import settings
        assistant_id = await service.create_assistant(
            name="Clario Decision Assistant",
            system_message=getattr(
                settings,
                "VAPI_SYSTEM_MESSAGE",
                "You are Clario's AI assistant helping users discuss meeting decisions.",
            ),
        )
        return {"assistant_id": assistant_id}
    finally:
        await service.close()


@router.post("/decisions/{decision_id}/assistant")
async def create_decision_voice_assistant(
    decision_id: str,
    request: DecisionVoiceRequest
) -> Dict[str, Any]:
    orchestrator = await get_decision_voice_orchestrator()
    payload = await orchestrator.create_voice_session(
        decision_id=decision_id,
        organization_id=request.organization_id,
        agenda_hint=request.agenda_hint,
        include_similar=request.include_similar
    )

    if not payload.get("assistant_id"):
        raise HTTPException(status_code=503, detail="Unable to provision voice assistant")

    return payload
