"""
Advanced search engine for Clario using Weaviate
Combines semantic search with traditional filtering
"""

from typing import List, Dict, Any, Optional, Union, Tuple
import asyncio
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

from app.services.vector.weaviate_service import get_weaviate_service
from app.services.vector.embeddings import get_embedding_service
from app.core.logging import get_logger

logger = get_logger("search_engine")


class SearchType(str, Enum):
    """Search type options"""
    SEMANTIC = "semantic"
    HYBRID = "hybrid"
    KEYWORD = "keyword"


class ResultType(str, Enum):
    """Result type options"""
    MEETING = "meeting"
    DECISION = "decision"
    AGENDA_ITEM = "agenda_item"
    DISCUSSION = "discussion"
    ALL = "all"


@dataclass
class SearchFilter:
    """Search filter configuration"""
    organization_id: Optional[str] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    meeting_type: Optional[str] = None
    decision_type: Optional[str] = None
    status: Optional[str] = None
    participants: Optional[List[str]] = None
    confidence_min: Optional[float] = None
    priority: Optional[str] = None


@dataclass
class SearchResult:
    """Search result with metadata"""
    id: str
    type: str
    title: str
    content: str
    metadata: Dict[str, Any]
    score: float
    certainty: float
    matched_fields: List[str]
    context: Optional[str] = None


class ClarioSearchEngine:
    """Advanced search engine for Clario meetings and decisions"""
    
    def __init__(self):
        self.weaviate_service = None
        self.embedding_service = None
        
    async def initialize(self) -> None:
        """Initialize search engine"""
        self.weaviate_service = await get_weaviate_service()
        self.embedding_service = await get_embedding_service()
        logger.info("Search engine initialized")
    
    async def search(
        self,
        query: str,
        search_type: SearchType = SearchType.HYBRID,
        result_types: List[ResultType] = None,
        filters: Optional[SearchFilter] = None,
        limit: int = 20,
        min_certainty: float = 0.7
    ) -> List[SearchResult]:
        """Perform comprehensive search across all content types"""
        try:
            if not result_types:
                result_types = [ResultType.MEETING, ResultType.DECISION]
            
            all_results = []
            
            # Search each content type
            for result_type in result_types:
                if result_type == ResultType.MEETING:
                    results = await self._search_meetings(
                        query, search_type, filters, limit, min_certainty
                    )
                elif result_type == ResultType.DECISION:
                    results = await self._search_decisions(
                        query, search_type, filters, limit, min_certainty
                    )
                elif result_type == ResultType.AGENDA_ITEM:
                    results = await self._search_agenda_items(
                        query, search_type, filters, limit, min_certainty
                    )
                elif result_type == ResultType.DISCUSSION:
                    results = await self._search_discussions(
                        query, search_type, filters, limit, min_certainty
                    )
                else:
                    continue
                
                all_results.extend(results)
            
            # Sort by relevance score and limit
            all_results.sort(key=lambda x: x.score, reverse=True)
            return all_results[:limit]
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    async def _search_meetings(
        self,
        query: str,
        search_type: SearchType,
        filters: Optional[SearchFilter],
        limit: int,
        min_certainty: float
    ) -> List[SearchResult]:
        """Search meetings with enhanced metadata"""
        try:
            # Build where filter
            where_filter = self._build_where_filter(filters, "Meeting")
            
            if search_type == SearchType.SEMANTIC:
                # Pure semantic search
                results = await self.weaviate_service.search_meetings(
                    query=query,
                    organization_id=filters.organization_id if filters else None,
                    limit=limit,
                    min_certainty=min_certainty
                )
            elif search_type == SearchType.HYBRID:
                # Hybrid search (semantic + keyword)
                results = await self._hybrid_search_meetings(
                    query, where_filter, limit, min_certainty
                )
            else:
                # Keyword search
                results = await self._keyword_search_meetings(
                    query, where_filter, limit
                )
            
            # Convert to SearchResult objects
            search_results = []
            for result in results:
                search_result = SearchResult(
                    id=result.get("meetingId", ""),
                    type="meeting",
                    title=result.get("title", ""),
                    content=self._extract_meeting_content(result),
                    metadata=self._extract_meeting_metadata(result),
                    score=result.get("_additional", {}).get("certainty", 0),
                    certainty=result.get("_additional", {}).get("certainty", 0),
                    matched_fields=self._identify_matched_fields(result, query),
                    context=self._generate_context(result, query)
                )
                search_results.append(search_result)
            
            return search_results
            
        except Exception as e:
            logger.error(f"Meeting search failed: {e}")
            return []
    
    async def _search_decisions(
        self,
        query: str,
        search_type: SearchType,
        filters: Optional[SearchFilter],
        limit: int,
        min_certainty: float
    ) -> List[SearchResult]:
        """Search decisions with enhanced metadata"""
        try:
            where_filter = self._build_where_filter(filters, "Decision")
            
            results = await self.weaviate_service.search_decisions(
                query=query,
                organization_id=filters.organization_id if filters else None,
                limit=limit,
                min_certainty=min_certainty
            )
            
            search_results = []
            for result in results:
                search_result = SearchResult(
                    id=result.get("decisionId", ""),
                    type="decision",
                    title=result.get("title", ""),
                    content=self._extract_decision_content(result),
                    metadata=self._extract_decision_metadata(result),
                    score=result.get("_additional", {}).get("certainty", 0),
                    certainty=result.get("_additional", {}).get("certainty", 0),
                    matched_fields=self._identify_matched_fields(result, query),
                    context=self._generate_context(result, query)
                )
                search_results.append(search_result)
            
            return search_results
            
        except Exception as e:
            logger.error(f"Decision search failed: {e}")
            return []
    
    async def _search_agenda_items(
        self,
        query: str,
        search_type: SearchType,
        filters: Optional[SearchFilter],
        limit: int,
        min_certainty: float
    ) -> List[SearchResult]:
        """Search agenda items"""
        try:
            # Implementation for agenda item search
            # This would be similar to meetings/decisions
            return []
        except Exception as e:
            logger.error(f"Agenda item search failed: {e}")
            return []
    
    async def _search_discussions(
        self,
        query: str,
        search_type: SearchType,
        filters: Optional[SearchFilter],
        limit: int,
        min_certainty: float
    ) -> List[SearchResult]:
        """Search discussion logs"""
        try:
            # Implementation for discussion search
            return []
        except Exception as e:
            logger.error(f"Discussion search failed: {e}")
            return []
    
    async def _hybrid_search_meetings(
        self,
        query: str,
        where_filter: Dict[str, Any],
        limit: int,
        min_certainty: float
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search combining semantic and keyword"""
        try:
            # This would implement hybrid search using both semantic similarity
            # and keyword matching with BM25 or similar
            
            # For now, fallback to semantic search
            return await self.weaviate_service.search_meetings(
                query=query,
                limit=limit,
                min_certainty=min_certainty
            )
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return []
    
    async def _keyword_search_meetings(
        self,
        query: str,
        where_filter: Dict[str, Any],
        limit: int
    ) -> List[Dict[str, Any]]:
        """Perform keyword-based search"""
        try:
            # Implementation would use Weaviate's keyword search capabilities
            # For now, return empty results
            return []
        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            return []
    
    def _build_where_filter(
        self,
        filters: Optional[SearchFilter],
        class_name: str
    ) -> Dict[str, Any]:
        """Build Weaviate where filter from SearchFilter"""
        if not filters:
            return {}
        
        where_conditions = []
        
        # Organization filter
        if filters.organization_id:
            where_conditions.append({
                "path": ["organizationId"],
                "operator": "Equal",
                "valueString": filters.organization_id
            })
        
        # Date range filter
        if filters.date_from or filters.date_to:
            if class_name == "Meeting" and filters.date_from:
                where_conditions.append({
                    "path": ["scheduledAt"],
                    "operator": "GreaterThanEqual",
                    "valueDate": filters.date_from.isoformat()
                })
            if class_name == "Meeting" and filters.date_to:
                where_conditions.append({
                    "path": ["scheduledAt"],
                    "operator": "LessThanEqual",
                    "valueDate": filters.date_to.isoformat()
                })
        
        # Type filters
        if filters.meeting_type and class_name == "Meeting":
            where_conditions.append({
                "path": ["meetingType"],
                "operator": "Equal",
                "valueString": filters.meeting_type
            })
        
        if filters.decision_type and class_name == "Decision":
            where_conditions.append({
                "path": ["decisionType"],
                "operator": "Equal",
                "valueString": filters.decision_type
            })
        
        # Status filter
        if filters.status:
            where_conditions.append({
                "path": ["status"],
                "operator": "Equal",
                "valueString": filters.status
            })
        
        # Confidence filter
        if filters.confidence_min:
            where_conditions.append({
                "path": ["confidenceScore"],
                "operator": "GreaterThanEqual",
                "valueNumber": filters.confidence_min
            })
        
        # Combine conditions with AND
        if len(where_conditions) == 1:
            return where_conditions[0]
        elif len(where_conditions) > 1:
            return {
                "operator": "And",
                "operands": where_conditions
            }
        
        return {}
    
    def _extract_meeting_content(self, result: Dict[str, Any]) -> str:
        """Extract searchable content from meeting result"""
        content_parts = []
        
        if result.get("title"):
            content_parts.append(result["title"])
        
        if result.get("description"):
            content_parts.append(result["description"])
        
        if result.get("agendaText"):
            content_parts.append(result["agendaText"])
        
        if result.get("summary"):
            content_parts.append(result["summary"])
        
        return " | ".join(content_parts)
    
    def _extract_decision_content(self, result: Dict[str, Any]) -> str:
        """Extract searchable content from decision result"""
        content_parts = []
        
        if result.get("title"):
            content_parts.append(result["title"])
        
        if result.get("description"):
            content_parts.append(result["description"])
        
        if result.get("reasoning"):
            content_parts.append(result["reasoning"])
        
        return " | ".join(content_parts)
    
    def _extract_meeting_metadata(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from meeting result"""
        return {
            "meeting_id": result.get("meetingId"),
            "organization_id": result.get("organizationId"),
            "meeting_type": result.get("meetingType"),
            "scheduled_at": result.get("scheduledAt"),
            "duration": result.get("duration"),
            "status": result.get("status"),
            "participants": result.get("participants", []),
            "key_topics": result.get("keyTopics", []),
            "confidence_score": result.get("confidenceScore")
        }
    
    def _extract_decision_metadata(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metadata from decision result"""
        return {
            "decision_id": result.get("decisionId"),
            "meeting_id": result.get("meetingId"),
            "organization_id": result.get("organizationId"),
            "decision_type": result.get("decisionType"),
            "urgency": result.get("urgency"),
            "status": result.get("status"),
            "impact_areas": result.get("impactAreas", []),
            "estimated_cost": result.get("estimatedCost"),
            "timeline": result.get("timeline"),
            "confidence_score": result.get("confidenceScore"),
            "decided_at": result.get("decidedAt")
        }
    
    def _identify_matched_fields(
        self,
        result: Dict[str, Any],
        query: str
    ) -> List[str]:
        """Identify which fields matched the search query"""
        matched_fields = []
        query_lower = query.lower()
        
        # Check each field for matches
        for field, value in result.items():
            if isinstance(value, str) and query_lower in value.lower():
                matched_fields.append(field)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str) and query_lower in item.lower():
                        matched_fields.append(field)
                        break
        
        return matched_fields
    
    def _generate_context(
        self,
        result: Dict[str, Any],
        query: str
    ) -> str:
        """Generate context snippet around the matched query"""
        # Find the best matching field and extract context
        query_lower = query.lower()
        
        for field in ["title", "description", "agendaText", "summary", "reasoning"]:
            value = result.get(field, "")
            if isinstance(value, str) and query_lower in value.lower():
                # Extract context around the match
                index = value.lower().find(query_lower)
                start = max(0, index - 50)
                end = min(len(value), index + len(query) + 50)
                context = value[start:end]
                
                if start > 0:
                    context = "..." + context
                if end < len(value):
                    context = context + "..."
                
                return context
        
        return ""
    
    # Advanced search methods
    async def find_similar_meetings(
        self,
        meeting_id: str,
        organization_id: str,
        limit: int = 5
    ) -> List[SearchResult]:
        """Find meetings similar to the given meeting"""
        try:
            # Get the meeting data to use as reference
            # This would require getting meeting from database first
            # For now, return empty
            return []
        except Exception as e:
            logger.error(f"Failed to find similar meetings: {e}")
            return []
    
    async def get_decision_precedents(
        self,
        decision_context: str,
        organization_id: str,
        limit: int = 10
    ) -> List[SearchResult]:
        """Find previous decisions that could inform current decision"""
        try:
            results = await self.search(
                query=decision_context,
                search_type=SearchType.SEMANTIC,
                result_types=[ResultType.DECISION],
                filters=SearchFilter(organization_id=organization_id),
                limit=limit,
                min_certainty=0.6
            )
            return results
        except Exception as e:
            logger.error(f"Failed to get decision precedents: {e}")
            return []
    
    async def search_by_participant(
        self,
        participant_name: str,
        organization_id: str,
        limit: int = 20
    ) -> List[SearchResult]:
        """Search for meetings/decisions involving specific participant"""
        try:
            # Search meetings where participant was involved
            filters = SearchFilter(
                organization_id=organization_id,
                participants=[participant_name]
            )
            
            return await self.search(
                query=participant_name,
                result_types=[ResultType.MEETING, ResultType.DECISION],
                filters=filters,
                limit=limit
            )
        except Exception as e:
            logger.error(f"Failed to search by participant: {e}")
            return []
    
    async def search_by_date_range(
        self,
        date_from: datetime,
        date_to: datetime,
        organization_id: str,
        limit: int = 50
    ) -> List[SearchResult]:
        """Search for content within date range"""
        try:
            filters = SearchFilter(
                organization_id=organization_id,
                date_from=date_from,
                date_to=date_to
            )
            
            return await self.search(
                query="*",  # Wildcard to get all results in range
                search_type=SearchType.KEYWORD,
                filters=filters,
                limit=limit,
                min_certainty=0.1  # Lower certainty for date-based search
            )
        except Exception as e:
            logger.error(f"Failed to search by date range: {e}")
            return []
    
    async def get_trending_topics(
        self,
        organization_id: str,
        days: int = 30,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get trending topics from recent meetings"""
        try:
            # This would analyze recent meetings to identify trending topics
            # Implementation would involve:
            # 1. Get recent meetings
            # 2. Extract key topics
            # 3. Analyze frequency and growth
            # 4. Return trending topics
            
            return []
        except Exception as e:
            logger.error(f"Failed to get trending topics: {e}")
            return []


# Global search engine instance
search_engine = None


async def get_search_engine() -> ClarioSearchEngine:
    """Get search engine instance"""
    global search_engine
    if not search_engine:
        search_engine = ClarioSearchEngine()
        await search_engine.initialize()
    return search_engine
