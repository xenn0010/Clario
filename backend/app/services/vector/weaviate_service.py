"""
Weaviate vector database service for Clario
Handles meeting embeddings, semantic search, and decision context
"""

import weaviate
from weaviate.exceptions import UnexpectedStatusCodeException
from typing import List, Dict, Any, Optional, Union
import asyncio
import json
import logging
from datetime import datetime
import hashlib

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger("weaviate")


class WeaviateService:
    """Weaviate vector database service"""
    
    def __init__(self):
        self.client = None
        self.is_connected = False
        
    @classmethod
    async def initialize(cls) -> 'WeaviateService':
        """Initialize Weaviate service"""
        service = cls()
        await service.connect()
        await service.setup_schemas()
        return service
    
    async def connect(self) -> None:
        """Connect to Weaviate instance"""
        try:
            auth = weaviate.AuthApiKey(api_key=settings.WEAVIATE_API_KEY) if settings.WEAVIATE_API_KEY else None
            timeout = (settings.WEAVIATE_TIMEOUT, settings.WEAVIATE_TIMEOUT * 2)

            self.client = weaviate.Client(
                url=settings.WEAVIATE_URL,
                auth_client_secret=auth,
                timeout_config=timeout
            )

            await asyncio.to_thread(self.client.schema.get)
            self.is_connected = True
            logger.info("Connected to Weaviate successfully")

        except Exception as e:
            logger.error(f"Failed to connect to Weaviate: {e}")
            raise
    
    async def setup_schemas(self) -> None:
        """Setup Weaviate schemas for Clario objects"""
        schemas = [
            self._get_meeting_schema(),
            self._get_decision_schema(),
            self._get_agenda_item_schema(),
            self._get_discussion_schema(),
            self._get_user_profile_schema(),
            self._get_organization_schema()
        ]
        
        for schema in schemas:
            await self._create_class_if_not_exists(schema)
    
    def _get_meeting_schema(self) -> Dict[str, Any]:
        """Meeting schema for vector storage"""
        return {
            "class": "Meeting",
            "description": "Meeting information with semantic embeddings",
            "vectorizer": "text2vec-openai",  # Can be changed to other vectorizers
            "properties": [
                {
                    "name": "meetingId",
                    "dataType": ["string"],
                    "description": "Unique meeting identifier"
                },
                {
                    "name": "organizationId", 
                    "dataType": ["string"],
                    "description": "Organization identifier"
                },
                {
                    "name": "title",
                    "dataType": ["text"],
                    "description": "Meeting title"
                },
                {
                    "name": "description",
                    "dataType": ["text"],
                    "description": "Meeting description"
                },
                {
                    "name": "agendaText",
                    "dataType": ["text"],
                    "description": "Full agenda content for semantic search"
                },
                {
                    "name": "summary",
                    "dataType": ["text"],
                    "description": "AI-generated meeting summary"
                },
                {
                    "name": "keyTopics",
                    "dataType": ["string[]"],
                    "description": "Key topics discussed"
                },
                {
                    "name": "meetingType",
                    "dataType": ["string"],
                    "description": "Type of meeting"
                },
                {
                    "name": "participants",
                    "dataType": ["string[]"],
                    "description": "Meeting participants"
                },
                {
                    "name": "scheduledAt",
                    "dataType": ["date"],
                    "description": "Meeting scheduled date"
                },
                {
                    "name": "duration",
                    "dataType": ["int"],
                    "description": "Meeting duration in minutes"
                },
                {
                    "name": "status",
                    "dataType": ["string"],
                    "description": "Meeting status"
                },
                {
                    "name": "confidenceScore",
                    "dataType": ["number"],
                    "description": "AI confidence score for decisions"
                }
            ]
        }
    
    def _get_decision_schema(self) -> Dict[str, Any]:
        """Decision schema for vector storage"""
        return {
            "class": "Decision",
            "description": "Meeting decisions with context and reasoning",
            "vectorizer": "text2vec-openai",
            "properties": [
                {
                    "name": "decisionId",
                    "dataType": ["string"],
                    "description": "Unique decision identifier"
                },
                {
                    "name": "meetingId",
                    "dataType": ["string"],
                    "description": "Parent meeting identifier"
                },
                {
                    "name": "organizationId",
                    "dataType": ["string"],
                    "description": "Organization identifier"
                },
                {
                    "name": "title",
                    "dataType": ["text"],
                    "description": "Decision title"
                },
                {
                    "name": "description",
                    "dataType": ["text"],
                    "description": "Detailed decision description"
                },
                {
                    "name": "reasoning",
                    "dataType": ["text"],
                    "description": "Decision reasoning and rationale"
                },
                {
                    "name": "optionsConsidered",
                    "dataType": ["text"],
                    "description": "Alternative options that were considered"
                },
                {
                    "name": "decisionType",
                    "dataType": ["string"],
                    "description": "Type/category of decision"
                },
                {
                    "name": "urgency",
                    "dataType": ["string"],
                    "description": "Decision urgency level"
                },
                {
                    "name": "impactAreas",
                    "dataType": ["string[]"],
                    "description": "Areas affected by this decision"
                },
                {
                    "name": "stakeholders",
                    "dataType": ["string[]"],
                    "description": "Decision stakeholders"
                },
                {
                    "name": "estimatedCost",
                    "dataType": ["number"],
                    "description": "Estimated financial impact"
                },
                {
                    "name": "timeline",
                    "dataType": ["string"],
                    "description": "Implementation timeline"
                },
                {
                    "name": "status",
                    "dataType": ["string"],
                    "description": "Decision status"
                },
                {
                    "name": "confidenceScore",
                    "dataType": ["number"],
                    "description": "AI confidence in this decision"
                },
                {
                    "name": "decidedAt",
                    "dataType": ["date"],
                    "description": "When decision was made"
                }
            ]
        }
    
    def _get_agenda_item_schema(self) -> Dict[str, Any]:
        """Agenda item schema for granular search"""
        return {
            "class": "AgendaItem",
            "description": "Individual agenda items for detailed analysis",
            "vectorizer": "text2vec-openai",
            "properties": [
                {
                    "name": "itemId",
                    "dataType": ["string"],
                    "description": "Unique agenda item identifier"
                },
                {
                    "name": "meetingId",
                    "dataType": ["string"],
                    "description": "Parent meeting identifier"
                },
                {
                    "name": "organizationId",
                    "dataType": ["string"],
                    "description": "Organization identifier"
                },
                {
                    "name": "title",
                    "dataType": ["text"],
                    "description": "Agenda item title"
                },
                {
                    "name": "description",
                    "dataType": ["text"],
                    "description": "Detailed description"
                },
                {
                    "name": "itemType",
                    "dataType": ["string"],
                    "description": "Type of agenda item"
                },
                {
                    "name": "priority",
                    "dataType": ["string"],
                    "description": "Item priority level"
                },
                {
                    "name": "estimatedDuration",
                    "dataType": ["int"],
                    "description": "Estimated duration in minutes"
                },
                {
                    "name": "requiresDecision",
                    "dataType": ["boolean"],
                    "description": "Whether item requires a decision"
                },
                {
                    "name": "orderIndex",
                    "dataType": ["int"],
                    "description": "Order in the agenda"
                },
                {
                    "name": "status",
                    "dataType": ["string"],
                    "description": "Item completion status"
                }
            ]
        }
    
    def _get_discussion_schema(self) -> Dict[str, Any]:
        """Discussion log schema for conversation analysis"""
        return {
            "class": "Discussion",
            "description": "AI agent discussion logs for context learning",
            "vectorizer": "text2vec-openai",
            "properties": [
                {
                    "name": "discussionId",
                    "dataType": ["string"],
                    "description": "Unique discussion identifier"
                },
                {
                    "name": "meetingId",
                    "dataType": ["string"],
                    "description": "Parent meeting identifier"
                },
                {
                    "name": "agentId",
                    "dataType": ["string"],
                    "description": "Speaking agent identifier"
                },
                {
                    "name": "content",
                    "dataType": ["text"],
                    "description": "Discussion content"
                },
                {
                    "name": "messageType",
                    "dataType": ["string"],
                    "description": "Type of message"
                },
                {
                    "name": "reasoning",
                    "dataType": ["text"],
                    "description": "Agent's reasoning"
                },
                {
                    "name": "confidenceLevel",
                    "dataType": ["number"],
                    "description": "Confidence in the statement"
                },
                {
                    "name": "timestamp",
                    "dataType": ["date"],
                    "description": "When message was sent"
                },
                {
                    "name": "organizationId",
                    "dataType": ["string"],
                    "description": "Organization identifier"
                }
            ]
        }
    
    def _get_user_profile_schema(self) -> Dict[str, Any]:
        """User decision profile schema"""
        return {
            "class": "UserProfile",
            "description": "User decision-making profiles for agent training",
            "vectorizer": "text2vec-openai",
            "properties": [
                {
                    "name": "userId",
                    "dataType": ["string"],
                    "description": "User identifier"
                },
                {
                    "name": "organizationId",
                    "dataType": ["string"],
                    "description": "Organization identifier"
                },
                {
                    "name": "decisionStyle",
                    "dataType": ["text"],
                    "description": "Decision-making style description"
                },
                {
                    "name": "communicationStyle",
                    "dataType": ["text"],
                    "description": "Communication preferences"
                },
                {
                    "name": "riskTolerance",
                    "dataType": ["string"],
                    "description": "Risk tolerance level"
                },
                {
                    "name": "decisionSpeed",
                    "dataType": ["string"],
                    "description": "Decision-making speed preference"
                },
                {
                    "name": "primaryValues",
                    "dataType": ["string[]"],
                    "description": "Key values that drive decisions"
                },
                {
                    "name": "analyticalScore",
                    "dataType": ["int"],
                    "description": "Analytical decision-making score"
                },
                {
                    "name": "intuitiveScore",
                    "dataType": ["int"],
                    "description": "Intuitive decision-making score"
                },
                {
                    "name": "collaborativeScore",
                    "dataType": ["int"],
                    "description": "Collaborative approach score"
                }
            ]
        }
    
    def _get_organization_schema(self) -> Dict[str, Any]:
        """Organization context schema"""
        return {
            "class": "Organization",
            "description": "Organization context and culture for decision-making",
            "vectorizer": "text2vec-openai",
            "properties": [
                {
                    "name": "organizationId",
                    "dataType": ["string"],
                    "description": "Organization identifier"
                },
                {
                    "name": "name",
                    "dataType": ["text"],
                    "description": "Organization name"
                },
                {
                    "name": "description",
                    "dataType": ["text"],
                    "description": "Organization description"
                },
                {
                    "name": "culture",
                    "dataType": ["text"],
                    "description": "Organizational culture description"
                },
                {
                    "name": "decisionMakingStyle",
                    "dataType": ["text"],
                    "description": "How decisions are typically made"
                },
                {
                    "name": "values",
                    "dataType": ["string[]"],
                    "description": "Organizational values"
                },
                {
                    "name": "industry",
                    "dataType": ["string"],
                    "description": "Industry sector"
                },
                {
                    "name": "size",
                    "dataType": ["int"],
                    "description": "Organization size"
                }
            ]
        }
    
    async def _create_class_if_not_exists(self, schema: Dict[str, Any]) -> None:
        """Create Weaviate class if it doesn't exist"""
        try:
            class_name = schema["class"]
            existing = await asyncio.to_thread(self.client.schema.get, class_name)
            if not existing:
                await asyncio.to_thread(self.client.schema.create_class, schema)
                logger.info(f"Created Weaviate class: {class_name}")
            else:
                logger.info(f"Weaviate class already exists: {class_name}")
        except UnexpectedStatusCodeException as exc:
            if getattr(exc, 'status_code', None) == 404:
                await asyncio.to_thread(self.client.schema.create_class, schema)
                logger.info(f"Created Weaviate class: {schema['class']}")
            else:
                logger.error(f"Failed to create class {schema['class']}: {exc}")
                raise
        except Exception as e:
            if "class name already exists" not in str(e).lower():
                logger.error(f"Failed to create class {schema['class']}: {e}")
                raise
    
    # Meeting Operations
    async def index_meeting(self, meeting_data: Dict[str, Any]) -> str:
        """Index a meeting in Weaviate"""
        try:
            # Prepare data for indexing
            weaviate_data = {
                "meetingId": meeting_data["id"],
                "organizationId": meeting_data["organization_id"],
                "title": meeting_data["title"],
                "description": meeting_data.get("description", ""),
                "agendaText": meeting_data.get("agenda_text", ""),
                "summary": meeting_data.get("ai_summary", ""),
                "keyTopics": meeting_data.get("key_topics", []),
                "meetingType": meeting_data.get("meeting_type", "discussion"),
                "participants": meeting_data.get("participants", []),
                "scheduledAt": meeting_data.get("scheduled_at"),
                "duration": meeting_data.get("duration_minutes", 30),
                "status": meeting_data.get("status", "scheduled"),
                "confidenceScore": meeting_data.get("ai_confidence_score", 0.0)
            }
            
            # Generate unique ID for Weaviate
            uuid = await asyncio.to_thread(
                self.client.data_object.create,
                weaviate_data,
                "Meeting"
            )
            
            logger.info(f"Indexed meeting {meeting_data['id']} in Weaviate")
            return uuid
            
        except Exception as e:
            logger.error(f"Failed to index meeting {meeting_data.get('id')}: {e}")
            raise
    
    async def search_meetings(
        self,
        query: str,
        organization_id: Optional[str] = None,
        meeting_type: Optional[str] = None,
        limit: int = 10,
        min_certainty: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Semantic search for meetings"""
        try:
            # Build where filter
            where_filter = {}
            if organization_id:
                where_filter["organizationId"] = {"equal": organization_id}
            if meeting_type:
                where_filter["meetingType"] = {"equal": meeting_type}
            
            # Perform semantic search
            result = await asyncio.to_thread(
                lambda: (
                    self.client.query
                    .get("Meeting", [
                        "meetingId", "title", "description", "agendaText",
                        "summary", "keyTopics", "meetingType", "participants",
                        "scheduledAt", "duration", "status", "confidenceScore"
                    ])
                    .with_near_text({"concepts": [query]})
                    .with_where(where_filter if where_filter else None)
                    .with_limit(limit)
                    .with_additional(["certainty", "distance"])
                    .do()
                )
            )
            
            meetings = result.get("data", {}).get("Get", {}).get("Meeting", [])
            
            # Filter by certainty
            filtered_meetings = [
                meeting for meeting in meetings
                if meeting.get("_additional", {}).get("certainty", 0) >= min_certainty
            ]
            
            logger.info(f"Found {len(filtered_meetings)} meetings for query: {query}")
            return filtered_meetings
            
        except Exception as e:
            logger.error(f"Failed to search meetings: {e}")
            raise
    
    # Decision Operations
    async def index_decision(self, decision_data: Dict[str, Any]) -> str:
        """Index a decision in Weaviate"""
        try:
            weaviate_data = {
                "decisionId": decision_data["id"],
                "meetingId": decision_data["meeting_id"],
                "organizationId": decision_data["organization_id"],
                "title": decision_data["title"],
                "description": decision_data["description"],
                "reasoning": decision_data.get("reasoning", ""),
                "optionsConsidered": json.dumps(decision_data.get("options_considered", [])),
                "decisionType": decision_data.get("decision_type", "operational"),
                "urgency": decision_data.get("urgency", "medium"),
                "impactAreas": decision_data.get("impact_areas", []),
                "stakeholders": decision_data.get("stakeholders", []),
                "estimatedCost": decision_data.get("estimated_cost", 0.0),
                "timeline": decision_data.get("estimated_timeline", ""),
                "status": decision_data.get("status", "pending"),
                "confidenceScore": decision_data.get("ai_confidence_score", 0.0),
                "decidedAt": decision_data.get("decided_at")
            }
            
            uuid = await asyncio.to_thread(
                self.client.data_object.create,
                weaviate_data,
                "Decision"
            )
            
            logger.info(f"Indexed decision {decision_data['id']} in Weaviate")
            return uuid
            
        except Exception as e:
            logger.error(f"Failed to index decision {decision_data.get('id')}: {e}")
            raise
    
    async def search_decisions(
        self,
        query: str,
        organization_id: Optional[str] = None,
        decision_type: Optional[str] = None,
        limit: int = 10,
        min_certainty: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Semantic search for decisions"""
        try:
            where_filter = {}
            if organization_id:
                where_filter["organizationId"] = {"equal": organization_id}
            if decision_type:
                where_filter["decisionType"] = {"equal": decision_type}
            
            result = await asyncio.to_thread(
                lambda: (
                    self.client.query
                    .get("Decision", [
                        "decisionId", "meetingId", "title", "description",
                        "reasoning", "optionsConsidered", "decisionType",
                        "urgency", "impactAreas", "estimatedCost", "timeline",
                        "status", "confidenceScore", "decidedAt"
                    ])
                    .with_near_text({"concepts": [query]})
                    .with_where(where_filter if where_filter else None)
                    .with_limit(limit)
                    .with_additional(["certainty", "distance"])
                    .do()
                )
            )
            
            decisions = result.get("data", {}).get("Get", {}).get("Decision", [])
            
            filtered_decisions = [
                decision for decision in decisions
                if decision.get("_additional", {}).get("certainty", 0) >= min_certainty
            ]
            
            logger.info(f"Found {len(filtered_decisions)} decisions for query: {query}")
            return filtered_decisions
            
        except Exception as e:
            logger.error(f"Failed to search decisions: {e}")
            raise
    
    # Context and Analytics
    async def get_meeting_context(
        self,
        current_agenda: str,
        organization_id: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Get relevant meeting context for current agenda"""
        try:
            # Search for similar meetings
            similar_meetings = await self.search_meetings(
                query=current_agenda,
                organization_id=organization_id,
                limit=limit,
                min_certainty=0.6
            )
            
            # Get related decisions
            related_decisions = await self.search_decisions(
                query=current_agenda,
                organization_id=organization_id,
                limit=limit,
                min_certainty=0.6
            )
            
            return {
                "similar_meetings": similar_meetings,
                "related_decisions": related_decisions,
                "context_strength": len(similar_meetings) + len(related_decisions)
            }
            
        except Exception as e:
            logger.error(f"Failed to get meeting context: {e}")
            raise
    
    async def analyze_decision_patterns(
        self,
        organization_id: str,
        time_period_days: int = 90
    ) -> Dict[str, Any]:
        """Analyze decision patterns for an organization"""
        try:
            # This would be a complex analysis
            # For now, return basic structure
            return {
                "total_decisions": 0,
                "decision_types": {},
                "success_rate": 0.0,
                "common_themes": [],
                "improvement_areas": []
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze decision patterns: {e}")
            raise
    
    # Utility Methods
    async def delete_by_id(self, class_name: str, object_id: str) -> bool:
        """Delete object by ID"""
        try:
            await asyncio.to_thread(
                self.client.data_object.delete,
                object_id,
                class_name
            )
            return True
        except Exception as e:
            logger.error(f"Failed to delete {class_name} object {object_id}: {e}")
            return False
    
    async def update_object(
        self,
        class_name: str,
        object_id: str,
        updates: Dict[str, Any]
    ) -> bool:
        """Update existing object"""
        try:
            await asyncio.to_thread(
                self.client.data_object.update,
                updates,
                class_name,
                object_id
            )
            return True
        except Exception as e:
            logger.error(f"Failed to update {class_name} object {object_id}: {e}")
            return False
    
    async def cleanup(self) -> None:
        """Cleanup connections"""
        if self.client:
            # Weaviate client doesn't need explicit cleanup
            self.is_connected = False
            logger.info("Weaviate service cleaned up")


# Global service instance
weaviate_service = None


async def get_weaviate_service() -> WeaviateService:
    """Get Weaviate service instance"""
    global weaviate_service
    if not weaviate_service:
        weaviate_service = await WeaviateService.initialize()
    return weaviate_service
