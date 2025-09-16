"""
Organization endpoints for vector operations
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, List
from pydantic import BaseModel

from app.services.vector.vector_manager import get_vector_manager
from app.core.logging import get_logger

logger = get_logger("organizations_api")

router = APIRouter()


class OrganizationIndexRequest(BaseModel):
    """Request to index organization context"""
    organization_data: Dict[str, Any]


class UserProfileIndexRequest(BaseModel):
    """Request to index user profile"""
    user_data: Dict[str, Any]
    profile_data: Dict[str, Any]


@router.post("/{org_id}/index")
async def index_organization(
    org_id: str,
    request: OrganizationIndexRequest
) -> Dict[str, Any]:
    """Index organization context in the vector database"""
    try:
        vector_manager = await get_vector_manager()
        
        # Ensure org ID matches
        request.organization_data["id"] = org_id
        
        success = await vector_manager.index_organization_context(
            request.organization_data
        )
        
        return {
            "success": success,
            "organization_id": org_id,
            "message": "Organization indexed successfully" if success else "Failed to index organization"
        }
        
    except Exception as e:
        logger.error(f"Organization indexing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")


@router.post("/{org_id}/users/{user_id}/profile/index")
async def index_user_profile(
    org_id: str,
    user_id: str,
    request: UserProfileIndexRequest
) -> Dict[str, Any]:
    """Index user decision profile for agent training"""
    try:
        vector_manager = await get_vector_manager()
        
        # Ensure IDs match
        request.user_data["id"] = user_id
        request.user_data["organization_id"] = org_id
        
        success = await vector_manager.index_user_profile(
            request.user_data,
            request.profile_data
        )
        
        return {
            "success": success,
            "user_id": user_id,
            "organization_id": org_id,
            "message": "User profile indexed successfully" if success else "Failed to index user profile"
        }
        
    except Exception as e:
        logger.error(f"User profile indexing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")


@router.post("/{org_id}/reindex")
async def reindex_organization(
    org_id: str,
    object_types: List[str] = Query(default=["meeting", "decision", "user_profile"])
) -> Dict[str, Any]:
    """Reindex all data for an organization"""
    try:
        vector_manager = await get_vector_manager()
        
        results = await vector_manager.reindex_organization(
            organization_id=org_id,
            object_types=object_types
        )
        
        return {
            "organization_id": org_id,
            "reindex_results": results,
            "message": "Organization reindexing completed"
        }
        
    except Exception as e:
        logger.error(f"Organization reindexing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Reindexing failed: {str(e)}")


@router.get("/{org_id}/analytics/patterns")
async def analyze_decision_patterns(
    org_id: str,
    time_period_days: int = Query(default=90, ge=1, le=365)
) -> Dict[str, Any]:
    """Analyze decision patterns for an organization"""
    try:
        from app.services.vector.weaviate_service import get_weaviate_service
        
        weaviate_service = await get_weaviate_service()
        
        patterns = await weaviate_service.analyze_decision_patterns(
            organization_id=org_id,
            time_period_days=time_period_days
        )
        
        return {
            "organization_id": org_id,
            "analysis_period_days": time_period_days,
            "patterns": patterns
        }
        
    except Exception as e:
        logger.error(f"Decision pattern analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Pattern analysis failed: {str(e)}")


@router.get("/indexing/status")
async def get_indexing_status() -> Dict[str, Any]:
    """Get current indexing status"""
    try:
        vector_manager = await get_vector_manager()
        
        status = await vector_manager.get_indexing_status()
        
        return status
        
    except Exception as e:
        logger.error(f"Get indexing status failed: {e}")
        raise HTTPException(status_code=500, detail=f"Status retrieval failed: {str(e)}")
