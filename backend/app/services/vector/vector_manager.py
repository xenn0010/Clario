"""
Vector management service for Clario
Handles indexing, updating, and maintenance of vector data
"""

from typing import List, Dict, Any, Optional, Tuple
import asyncio
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
import json

from app.services.vector.weaviate_service import get_weaviate_service
from app.services.vector.embeddings import get_embedding_service
from app.core.logging import get_logger

logger = get_logger("vector_manager")


@dataclass
class IndexingJob:
    """Indexing job configuration"""
    job_id: str
    object_type: str
    object_id: str
    data: Dict[str, Any]
    priority: int = 5  # 1-10, higher is more important
    created_at: datetime = None
    status: str = "pending"  # pending, processing, completed, failed
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()


class VectorManager:
    """Manages vector indexing and maintenance for Clario"""
    
    def __init__(self):
        self.weaviate_service = None
        self.embedding_service = None
        self.indexing_queue = []
        self.processing = False
        
    async def initialize(self) -> None:
        """Initialize vector manager"""
        self.weaviate_service = await get_weaviate_service()
        self.embedding_service = await get_embedding_service()
        logger.info("Vector manager initialized")
    
    # Meeting Operations
    async def index_meeting(
        self,
        meeting_data: Dict[str, Any],
        force_reindex: bool = False
    ) -> bool:
        """Index or update meeting in vector database"""
        try:
            meeting_id = meeting_data.get("id")
            if not meeting_id:
                logger.error("Meeting data missing ID")
                return False
            
            # Check if already indexed
            if not force_reindex:
                existing = await self._check_existing_meeting(meeting_id)
                if existing:
                    logger.info(f"Meeting {meeting_id} already indexed, skipping")
                    return True
            
            # Generate embeddings for meeting content
            embeddings = await self.embedding_service.embed_meeting_content(meeting_data)
            
            # Prepare data for Weaviate
            weaviate_data = await self._prepare_meeting_data(meeting_data, embeddings)
            
            # Index in Weaviate
            uuid = await self.weaviate_service.index_meeting(weaviate_data)
            
            # Index agenda items separately
            if meeting_data.get("agenda_items"):
                await self._index_agenda_items(
                    meeting_data["agenda_items"],
                    meeting_id,
                    meeting_data.get("organization_id")
                )
            
            logger.info(f"Successfully indexed meeting {meeting_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to index meeting {meeting_data.get('id')}: {e}")
            return False
    
    async def index_decision(
        self,
        decision_data: Dict[str, Any],
        force_reindex: bool = False
    ) -> bool:
        """Index or update decision in vector database"""
        try:
            decision_id = decision_data.get("id")
            if not decision_id:
                logger.error("Decision data missing ID")
                return False
            
            # Generate embeddings
            embeddings = await self.embedding_service.embed_decision_content(decision_data)
            
            # Prepare data for Weaviate
            weaviate_data = await self._prepare_decision_data(decision_data, embeddings)
            
            # Index in Weaviate
            uuid = await self.weaviate_service.index_decision(weaviate_data)
            
            logger.info(f"Successfully indexed decision {decision_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to index decision {decision_data.get('id')}: {e}")
            return False
    
    async def index_user_profile(
        self,
        user_data: Dict[str, Any],
        profile_data: Dict[str, Any]
    ) -> bool:
        """Index user decision profile for agent training"""
        try:
            user_id = user_data.get("id")
            if not user_id:
                return False
            
            # Generate profile embedding
            profile_embedding = await self.embedding_service.embed_user_profile(profile_data)
            
            # Prepare data for Weaviate
            weaviate_data = {
                "userId": user_id,
                "organizationId": user_data.get("organization_id"),
                "decisionStyle": profile_data.get("decision_style", ""),
                "communicationStyle": profile_data.get("communication_style", ""),
                "riskTolerance": profile_data.get("risk_tolerance", "medium"),
                "decisionSpeed": profile_data.get("decision_speed", "moderate"),
                "primaryValues": profile_data.get("primary_values", []),
                "analyticalScore": profile_data.get("analytical_score", 0),
                "intuitiveScore": profile_data.get("intuitive_score", 0),
                "collaborativeScore": profile_data.get("collaborative_score", 0)
            }
            
            # Create in Weaviate
            await asyncio.to_thread(
                self.weaviate_service.client.data_object.create,
                weaviate_data,
                "UserProfile"
            )
            
            logger.info(f"Successfully indexed user profile {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to index user profile {user_data.get('id')}: {e}")
            return False
    
    async def index_organization_context(
        self,
        org_data: Dict[str, Any]
    ) -> bool:
        """Index organization context for decision-making"""
        try:
            org_id = org_data.get("id")
            if not org_id:
                return False
            
            # Create organization context text
            context_parts = []
            if org_data.get("name"):
                context_parts.append(f"Organization: {org_data['name']}")
            if org_data.get("description"):
                context_parts.append(f"Description: {org_data['description']}")
            
            context_text = " ".join(context_parts)
            
            # Prepare data for Weaviate
            weaviate_data = {
                "organizationId": org_id,
                "name": org_data.get("name", ""),
                "description": org_data.get("description", ""),
                "culture": org_data.get("culture", ""),
                "decisionMakingStyle": org_data.get("decision_making_style", ""),
                "values": org_data.get("values", []),
                "industry": org_data.get("industry", ""),
                "size": org_data.get("max_team_size", 0)
            }
            
            # Create in Weaviate
            await asyncio.to_thread(
                self.weaviate_service.client.data_object.create,
                weaviate_data,
                "Organization"
            )
            
            logger.info(f"Successfully indexed organization {org_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to index organization {org_data.get('id')}: {e}")
            return False
    
    # Batch Operations
    async def batch_index_meetings(
        self,
        meetings: List[Dict[str, Any]],
        batch_size: int = 10
    ) -> Dict[str, int]:
        """Index multiple meetings in batches"""
        try:
            total = len(meetings)
            successful = 0
            failed = 0
            
            for i in range(0, total, batch_size):
                batch = meetings[i:i + batch_size]
                batch_tasks = []
                
                for meeting in batch:
                    task = self.index_meeting(meeting)
                    batch_tasks.append(task)
                
                # Process batch
                results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                for result in results:
                    if isinstance(result, Exception):
                        failed += 1
                    elif result:
                        successful += 1
                    else:
                        failed += 1
                
                # Small delay between batches
                if i + batch_size < total:
                    await asyncio.sleep(0.5)
            
            logger.info(f"Batch indexing completed: {successful} successful, {failed} failed")
            return {"successful": successful, "failed": failed, "total": total}
            
        except Exception as e:
            logger.error(f"Batch indexing failed: {e}")
            return {"successful": 0, "failed": total, "total": total}
    
    async def batch_index_decisions(
        self,
        decisions: List[Dict[str, Any]],
        batch_size: int = 10
    ) -> Dict[str, int]:
        """Index multiple decisions in batches"""
        try:
            total = len(decisions)
            successful = 0
            failed = 0
            
            for i in range(0, total, batch_size):
                batch = decisions[i:i + batch_size]
                batch_tasks = []
                
                for decision in batch:
                    task = self.index_decision(decision)
                    batch_tasks.append(task)
                
                results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                
                for result in results:
                    if isinstance(result, Exception):
                        failed += 1
                    elif result:
                        successful += 1
                    else:
                        failed += 1
                
                if i + batch_size < total:
                    await asyncio.sleep(0.5)
            
            logger.info(f"Decision batch indexing: {successful} successful, {failed} failed")
            return {"successful": successful, "failed": failed, "total": total}
            
        except Exception as e:
            logger.error(f"Decision batch indexing failed: {e}")
            return {"successful": 0, "failed": total, "total": total}
    
    # Queue Management
    async def queue_indexing_job(self, job: IndexingJob) -> None:
        """Add job to indexing queue"""
        self.indexing_queue.append(job)
        self.indexing_queue.sort(key=lambda x: x.priority, reverse=True)
        
        # Start processing if not already running
        if not self.processing:
            asyncio.create_task(self._process_indexing_queue())
    
    async def _process_indexing_queue(self) -> None:
        """Process indexing queue"""
        if self.processing:
            return
        
        self.processing = True
        
        try:
            while self.indexing_queue:
                job = self.indexing_queue.pop(0)
                job.status = "processing"
                
                try:
                    if job.object_type == "meeting":
                        success = await self.index_meeting(job.data)
                    elif job.object_type == "decision":
                        success = await self.index_decision(job.data)
                    elif job.object_type == "user_profile":
                        success = await self.index_user_profile(
                            job.data.get("user_data", {}),
                            job.data.get("profile_data", {})
                        )
                    else:
                        success = False
                    
                    job.status = "completed" if success else "failed"
                    
                except Exception as e:
                    logger.error(f"Indexing job {job.job_id} failed: {e}")
                    job.status = "failed"
                
                # Small delay between jobs
                await asyncio.sleep(0.1)
        
        finally:
            self.processing = False
    
    # Maintenance Operations
    async def cleanup_old_vectors(self, days_old: int = 365) -> int:
        """Clean up old vector data"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days_old)
            # Implementation would delete old vectors
            # For now, return 0
            return 0
        except Exception as e:
            logger.error(f"Vector cleanup failed: {e}")
            return 0
    
    async def reindex_organization(
        self,
        organization_id: str,
        object_types: List[str] = None
    ) -> Dict[str, int]:
        """Reindex all data for an organization"""
        try:
            if not object_types:
                object_types = ["meeting", "decision", "user_profile"]
            
            results = {}
            
            for object_type in object_types:
                if object_type == "meeting":
                    # Get all meetings for organization from database
                    # For now, skip implementation
                    results["meetings"] = 0
                elif object_type == "decision":
                    # Get all decisions for organization
                    results["decisions"] = 0
                elif object_type == "user_profile":
                    # Get all user profiles for organization
                    results["user_profiles"] = 0
            
            return results
            
        except Exception as e:
            logger.error(f"Organization reindexing failed: {e}")
            return {}
    
    async def get_indexing_status(self) -> Dict[str, Any]:
        """Get current indexing status"""
        return {
            "queue_size": len(self.indexing_queue),
            "processing": self.processing,
            "pending_jobs": [
                {
                    "job_id": job.job_id,
                    "object_type": job.object_type,
                    "priority": job.priority,
                    "status": job.status,
                    "created_at": job.created_at.isoformat()
                }
                for job in self.indexing_queue
            ]
        }
    
    # Helper Methods
    async def _prepare_meeting_data(
        self,
        meeting_data: Dict[str, Any],
        embeddings: Dict[str, List[float]]
    ) -> Dict[str, Any]:
        """Prepare meeting data for Weaviate indexing"""
        return {
            **meeting_data,
            "embeddings": embeddings
        }
    
    async def _prepare_decision_data(
        self,
        decision_data: Dict[str, Any],
        embeddings: Dict[str, List[float]]
    ) -> Dict[str, Any]:
        """Prepare decision data for Weaviate indexing"""
        return {
            **decision_data,
            "embeddings": embeddings
        }
    
    async def _check_existing_meeting(self, meeting_id: str) -> bool:
        """Check if meeting is already indexed"""
        try:
            # Query Weaviate to check if meeting exists
            result = await asyncio.to_thread(
                lambda: (
                    self.weaviate_service.client.query
                    .get("Meeting", ["meetingId"])
                    .with_where({
                        "path": ["meetingId"],
                        "operator": "Equal",
                        "valueString": meeting_id
                    })
                    .with_limit(1)
                    .do()
                )
            )
            
            meetings = result.get("data", {}).get("Get", {}).get("Meeting", [])
            return len(meetings) > 0
            
        except Exception as e:
            logger.error(f"Failed to check existing meeting: {e}")
            return False
    
    async def _index_agenda_items(
        self,
        agenda_items: List[Dict[str, Any]],
        meeting_id: str,
        organization_id: str
    ) -> None:
        """Index agenda items separately"""
        try:
            embedded_items = await self.embedding_service.embed_agenda_items(agenda_items)
            
            for item in embedded_items:
                weaviate_data = {
                    "itemId": item.get("id", ""),
                    "meetingId": meeting_id,
                    "organizationId": organization_id,
                    "title": item.get("title", ""),
                    "description": item.get("description", ""),
                    "itemType": item.get("item_type", "discussion"),
                    "priority": item.get("priority", "medium"),
                    "estimatedDuration": item.get("estimated_duration", 0),
                    "requiresDecision": item.get("requires_decision", False),
                    "orderIndex": item.get("order_index", 0),
                    "status": item.get("status", "pending")
                }
                
                await asyncio.to_thread(
                    self.weaviate_service.client.data_object.create,
                    weaviate_data,
                    "AgendaItem"
                )
            
        except Exception as e:
            logger.error(f"Failed to index agenda items: {e}")


# Global vector manager instance
vector_manager = None


async def get_vector_manager() -> VectorManager:
    """Get vector manager instance"""
    global vector_manager
    if not vector_manager:
        vector_manager = VectorManager()
        await vector_manager.initialize()
    return vector_manager
