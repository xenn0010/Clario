"""
Search endpoints for Clario
Vector search and semantic discovery
"""

from fastapi import APIRouter, HTTPException, Query, Depends
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

from app.services.vector.search_engine import get_search_engine, SearchType, ResultType, SearchFilter
from app.core.logging import get_logger

logger = get_logger("search_api")

router = APIRouter()


class SearchRequest(BaseModel):
    """Search request model"""
    query: str = Field(..., description="Search query")
    search_type: SearchType = Field(default=SearchType.HYBRID, description="Type of search to perform")
    result_types: List[ResultType] = Field(default=[ResultType.MEETING, ResultType.DECISION], description="Types of results to return")
    organization_id: Optional[str] = Field(default=None, description="Organization to search within")
    date_from: Optional[datetime] = Field(default=None, description="Start date for filtering")
    date_to: Optional[datetime] = Field(default=None, description="End date for filtering")
    meeting_type: Optional[str] = Field(default=None, description="Filter by meeting type")
    decision_type: Optional[str] = Field(default=None, description="Filter by decision type")
    status: Optional[str] = Field(default=None, description="Filter by status")
    participants: Optional[List[str]] = Field(default=None, description="Filter by participants")
    confidence_min: Optional[float] = Field(default=None, ge=0, le=1, description="Minimum confidence score")
    priority: Optional[str] = Field(default=None, description="Filter by priority")
    limit: int = Field(default=20, ge=1, le=100, description="Maximum number of results")
    min_certainty: float = Field(default=0.7, ge=0, le=1, description="Minimum certainty threshold")


class SearchResultResponse(BaseModel):
    """Search result response model"""
    id: str
    type: str
    title: str
    content: str
    metadata: Dict[str, Any]
    score: float
    certainty: float
    matched_fields: List[str]
    context: Optional[str] = None


class SearchResponse(BaseModel):
    """Search response model"""
    query: str
    total_results: int
    search_type: str
    processing_time_ms: float
    results: List[SearchResultResponse]


@router.post("/", response_model=SearchResponse)
async def search(
    request: SearchRequest
) -> SearchResponse:
    """
    Perform semantic search across meetings and decisions
    
    - **query**: Search query text
    - **search_type**: semantic, hybrid, or keyword search
    - **result_types**: Types of content to search (meetings, decisions, etc.)
    - **filters**: Various filters to narrow down results
    """
    try:
        start_time = datetime.utcnow()
        
        # Get search engine
        search_engine = await get_search_engine()
        
        # Build search filter
        filters = SearchFilter(
            organization_id=request.organization_id,
            date_from=request.date_from,
            date_to=request.date_to,
            meeting_type=request.meeting_type,
            decision_type=request.decision_type,
            status=request.status,
            participants=request.participants,
            confidence_min=request.confidence_min,
            priority=request.priority
        )
        
        # Perform search
        results = await search_engine.search(
            query=request.query,
            search_type=request.search_type,
            result_types=request.result_types,
            filters=filters,
            limit=request.limit,
            min_certainty=request.min_certainty
        )
        
        # Calculate processing time
        end_time = datetime.utcnow()
        processing_time = (end_time - start_time).total_seconds() * 1000
        
        # Convert results to response model
        result_responses = [
            SearchResultResponse(
                id=result.id,
                type=result.type,
                title=result.title,
                content=result.content,
                metadata=result.metadata,
                score=result.score,
                certainty=result.certainty,
                matched_fields=result.matched_fields,
                context=result.context
            )
            for result in results
        ]
        
        logger.info(f"Search completed: {len(results)} results for query '{request.query}'")
        
        return SearchResponse(
            query=request.query,
            total_results=len(results),
            search_type=request.search_type.value,
            processing_time_ms=processing_time,
            results=result_responses
        )
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/similar-meetings/{meeting_id}")
async def get_similar_meetings(
    meeting_id: str,
    organization_id: str = Query(..., description="Organization ID"),
    limit: int = Query(default=5, ge=1, le=20, description="Number of similar meetings to return")
) -> List[SearchResultResponse]:
    """
    Find meetings similar to the specified meeting
    """
    try:
        search_engine = await get_search_engine()
        
        results = await search_engine.find_similar_meetings(
            meeting_id=meeting_id,
            organization_id=organization_id,
            limit=limit
        )
        
        return [
            SearchResultResponse(
                id=result.id,
                type=result.type,
                title=result.title,
                content=result.content,
                metadata=result.metadata,
                score=result.score,
                certainty=result.certainty,
                matched_fields=result.matched_fields,
                context=result.context
            )
            for result in results
        ]
        
    except Exception as e:
        logger.error(f"Similar meetings search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Similar meetings search failed: {str(e)}")


@router.get("/decision-precedents")
async def get_decision_precedents(
    context: str = Query(..., description="Decision context to find precedents for"),
    organization_id: str = Query(..., description="Organization ID"),
    limit: int = Query(default=10, ge=1, le=50, description="Number of precedents to return")
) -> List[SearchResultResponse]:
    """
    Find previous decisions that could inform current decision-making
    """
    try:
        search_engine = await get_search_engine()
        
        results = await search_engine.get_decision_precedents(
            decision_context=context,
            organization_id=organization_id,
            limit=limit
        )
        
        return [
            SearchResultResponse(
                id=result.id,
                type=result.type,
                title=result.title,
                content=result.content,
                metadata=result.metadata,
                score=result.score,
                certainty=result.certainty,
                matched_fields=result.matched_fields,
                context=result.context
            )
            for result in results
        ]
        
    except Exception as e:
        logger.error(f"Decision precedents search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Decision precedents search failed: {str(e)}")


@router.get("/by-participant/{participant_name}")
async def search_by_participant(
    participant_name: str,
    organization_id: str = Query(..., description="Organization ID"),
    limit: int = Query(default=20, ge=1, le=100, description="Number of results to return")
) -> List[SearchResultResponse]:
    """
    Search for meetings and decisions involving a specific participant
    """
    try:
        search_engine = await get_search_engine()
        
        results = await search_engine.search_by_participant(
            participant_name=participant_name,
            organization_id=organization_id,
            limit=limit
        )
        
        return [
            SearchResultResponse(
                id=result.id,
                type=result.type,
                title=result.title,
                content=result.content,
                metadata=result.metadata,
                score=result.score,
                certainty=result.certainty,
                matched_fields=result.matched_fields,
                context=result.context
            )
            for result in results
        ]
        
    except Exception as e:
        logger.error(f"Participant search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Participant search failed: {str(e)}")


@router.get("/trending-topics")
async def get_trending_topics(
    organization_id: str = Query(..., description="Organization ID"),
    days: int = Query(default=30, ge=1, le=365, description="Number of days to analyze"),
    limit: int = Query(default=10, ge=1, le=50, description="Number of topics to return")
) -> Dict[str, Any]:
    """
    Get trending topics from recent meetings and decisions
    """
    try:
        search_engine = await get_search_engine()
        
        topics = await search_engine.get_trending_topics(
            organization_id=organization_id,
            days=days,
            limit=limit
        )
        
        return {
            "organization_id": organization_id,
            "analysis_period_days": days,
            "trending_topics": topics
        }
        
    except Exception as e:
        logger.error(f"Trending topics analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Trending topics analysis failed: {str(e)}")


@router.get("/autocomplete")
async def search_autocomplete(
    query: str = Query(..., description="Partial search query"),
    organization_id: str = Query(..., description="Organization ID"),
    limit: int = Query(default=5, ge=1, le=20, description="Number of suggestions to return")
) -> List[Dict[str, str]]:
    """
    Get search suggestions for autocomplete
    """
    try:
        # This would implement autocomplete suggestions
        # For now, return empty list
        return []
        
    except Exception as e:
        logger.error(f"Autocomplete failed: {e}")
        raise HTTPException(status_code=500, detail=f"Autocomplete failed: {str(e)}")
