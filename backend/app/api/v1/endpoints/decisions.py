"""
Decision endpoints for vector operations
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
from pydantic import BaseModel

from app.services.vector.vector_manager import get_vector_manager
from app.core.logging import get_logger

logger = get_logger("decisions_api")

router = APIRouter()


class DecisionIndexRequest(BaseModel):
    """Request to index a decision"""
    decision_data: Dict[str, Any]
    force_reindex: bool = False


@router.post("/{decision_id}/index")
async def index_decision(
    decision_id: str,
    request: DecisionIndexRequest
) -> Dict[str, Any]:
    """Index a decision in the vector database"""
    try:
        vector_manager = await get_vector_manager()
        
        # Ensure decision ID matches
        request.decision_data["id"] = decision_id
        
        success = await vector_manager.index_decision(
            request.decision_data,
            request.force_reindex
        )
        
        return {
            "success": success,
            "decision_id": decision_id,
            "message": "Decision indexed successfully" if success else "Failed to index decision"
        }
        
    except Exception as e:
        logger.error(f"Decision indexing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")
