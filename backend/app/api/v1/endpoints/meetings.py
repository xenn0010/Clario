"""
Meeting endpoints for vector operations
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
from pydantic import BaseModel

from app.services.vector.vector_manager import get_vector_manager
from app.core.logging import get_logger

logger = get_logger("meetings_api")

router = APIRouter()


class MeetingIndexRequest(BaseModel):
    """Request to index a meeting"""
    meeting_data: Dict[str, Any]
    force_reindex: bool = False


@router.post("/{meeting_id}/index")
async def index_meeting(
    meeting_id: str,
    request: MeetingIndexRequest
) -> Dict[str, Any]:
    """Index a meeting in the vector database"""
    try:
        vector_manager = await get_vector_manager()
        
        # Ensure meeting ID matches
        request.meeting_data["id"] = meeting_id
        
        success = await vector_manager.index_meeting(
            request.meeting_data,
            request.force_reindex
        )
        
        return {
            "success": success,
            "meeting_id": meeting_id,
            "message": "Meeting indexed successfully" if success else "Failed to index meeting"
        }
        
    except Exception as e:
        logger.error(f"Meeting indexing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")


@router.get("/{meeting_id}/context")
async def get_meeting_context(
    meeting_id: str,
    agenda: str,
    organization_id: str
) -> Dict[str, Any]:
    """Get relevant context for a meeting based on its agenda"""
    try:
        from app.services.vector.weaviate_service import get_weaviate_service
        
        weaviate_service = await get_weaviate_service()
        
        context = await weaviate_service.get_meeting_context(
            current_agenda=agenda,
            organization_id=organization_id
        )
        
        return {
            "meeting_id": meeting_id,
            "context": context
        }
        
    except Exception as e:
        logger.error(f"Get meeting context failed: {e}")
        raise HTTPException(status_code=500, detail=f"Context retrieval failed: {str(e)}")
