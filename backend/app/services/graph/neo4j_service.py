"""
Neo4j graph database service for Clario
Maps relationships between meetings, decisions, topics, and consequences
"""

from neo4j import AsyncGraphDatabase, AsyncDriver
from typing import List, Dict, Any, Optional, Tuple
import asyncio
import json
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger("neo4j")


@dataclass
class GraphNode:
    """Graph node representation"""
    node_type: str
    node_id: str
    properties: Dict[str, Any]


@dataclass
class GraphRelationship:
    """Graph relationship representation"""
    from_node: str
    to_node: str
    relationship_type: str
    properties: Dict[str, Any]


class Neo4jService:
    """Neo4j graph database service for Clario"""
    
    def __init__(self):
        self.driver: Optional[AsyncDriver] = None
        self.is_connected = False
        
    @classmethod
    async def initialize(cls) -> 'Neo4jService':
        """Initialize Neo4j service"""
        service = cls()
        await service.connect()
        await service.setup_constraints()
        return service
    
    async def connect(self) -> None:
        """Connect to Neo4j database"""
        try:
            self.driver = AsyncGraphDatabase.driver(
                settings.NEO4J_URI,
                auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD),
                connection_timeout=30
            )
            
            # Test connection
            await self.driver.verify_connectivity()
            self.is_connected = True
            logger.info("Connected to Neo4j successfully")
            
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            raise
    
    async def setup_constraints(self) -> None:
        """Setup Neo4j constraints and indexes"""
        constraints = [
            # Node uniqueness constraints
            "CREATE CONSTRAINT meeting_id IF NOT EXISTS FOR (m:Meeting) REQUIRE m.id IS UNIQUE",
            "CREATE CONSTRAINT decision_id IF NOT EXISTS FOR (d:Decision) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT topic_name IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE",
            "CREATE CONSTRAINT user_id IF NOT EXISTS FOR (u:User) REQUIRE u.id IS UNIQUE",
            "CREATE CONSTRAINT organization_id IF NOT EXISTS FOR (o:Organization) REQUIRE o.id IS UNIQUE",
            "CREATE CONSTRAINT agent_id IF NOT EXISTS FOR (a:Agent) REQUIRE a.id IS UNIQUE",
            
            # Indexes for performance
            "CREATE INDEX meeting_date IF NOT EXISTS FOR (m:Meeting) ON (m.scheduled_at)",
            "CREATE INDEX decision_date IF NOT EXISTS FOR (d:Decision) ON (d.decided_at)",
            "CREATE INDEX decision_type IF NOT EXISTS FOR (d:Decision) ON (d.decision_type)",
            "CREATE INDEX topic_category IF NOT EXISTS FOR (t:Topic) ON (t.category)",
            "CREATE INDEX outcome_status IF NOT EXISTS FOR (o:Outcome) ON (o.status)"
        ]
        
        async with self.driver.session() as session:
            for constraint in constraints:
                try:
                    await session.run(constraint)
                except Exception as e:
                    # Constraint might already exist
                    if "already exists" not in str(e).lower():
                        logger.warning(f"Failed to create constraint: {e}")
        
        logger.info("Neo4j constraints and indexes set up")
    
    # Node Creation Methods
    async def create_meeting_node(self, meeting_data: Dict[str, Any]) -> bool:
        """Create or update meeting node"""
        try:
            query = """
            MERGE (m:Meeting {id: $id})
            SET m.title = $title,
                m.description = $description,
                m.meeting_type = $meeting_type,
                m.status = $status,
                m.scheduled_at = datetime($scheduled_at),
                m.duration_minutes = $duration_minutes,
                m.confidence_score = $confidence_score,
                m.organization_id = $organization_id,
                m.created_at = datetime($created_at),
                m.updated_at = datetime()
            RETURN m
            """
            
            params = {
                "id": meeting_data["id"],
                "title": meeting_data.get("title", ""),
                "description": meeting_data.get("description", ""),
                "meeting_type": meeting_data.get("meeting_type", "discussion"),
                "status": meeting_data.get("status", "scheduled"),
                "scheduled_at": meeting_data.get("scheduled_at", datetime.utcnow()).isoformat(),
                "duration_minutes": meeting_data.get("duration_minutes", 30),
                "confidence_score": meeting_data.get("ai_confidence_score", 0.0),
                "organization_id": meeting_data.get("organization_id", ""),
                "created_at": meeting_data.get("created_at", datetime.utcnow()).isoformat()
            }
            
            async with self.driver.session() as session:
                result = await session.run(query, params)
                await result.consume()
            
            logger.info(f"Created/updated meeting node: {meeting_data['id']}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create meeting node: {e}")
            return False
    
    async def create_decision_node(self, decision_data: Dict[str, Any]) -> bool:
        """Create or update decision node"""
        try:
            query = """
            MERGE (d:Decision {id: $id})
            SET d.title = $title,
                d.description = $description,
                d.decision_type = $decision_type,
                d.urgency = $urgency,
                d.status = $status,
                d.reasoning = $reasoning,
                d.estimated_cost = $estimated_cost,
                d.timeline = $timeline,
                d.confidence_score = $confidence_score,
                d.meeting_id = $meeting_id,
                d.organization_id = $organization_id,
                d.decided_at = datetime($decided_at),
                d.implementation_progress = $implementation_progress,
                d.success_rating = $success_rating,
                d.created_at = datetime($created_at),
                d.updated_at = datetime()
            RETURN d
            """
            
            params = {
                "id": decision_data["id"],
                "title": decision_data.get("title", ""),
                "description": decision_data.get("description", ""),
                "decision_type": decision_data.get("decision_type", "operational"),
                "urgency": decision_data.get("urgency", "medium"),
                "status": decision_data.get("status", "pending"),
                "reasoning": decision_data.get("reasoning", ""),
                "estimated_cost": decision_data.get("estimated_cost", 0.0),
                "timeline": decision_data.get("estimated_timeline", ""),
                "confidence_score": decision_data.get("ai_confidence_score", 0.0),
                "meeting_id": decision_data.get("meeting_id", ""),
                "organization_id": decision_data.get("organization_id", ""),
                "decided_at": decision_data.get("decided_at", datetime.utcnow()).isoformat(),
                "implementation_progress": decision_data.get("implementation_progress", 0),
                "success_rating": decision_data.get("success_rating", 0.0),
                "created_at": decision_data.get("created_at", datetime.utcnow()).isoformat()
            }
            
            async with self.driver.session() as session:
                result = await session.run(query, params)
                await result.consume()
            
            logger.info(f"Created/updated decision node: {decision_data['id']}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create decision node: {e}")
            return False
    
    async def create_topic_node(self, topic_name: str, category: str = "general", properties: Dict[str, Any] = None) -> bool:
        """Create or update topic node"""
        try:
            if properties is None:
                properties = {}
            
            query = """
            MERGE (t:Topic {name: $name})
            SET t.category = $category,
                t.frequency = COALESCE(t.frequency, 0) + 1,
                t.last_discussed = datetime(),
                t.updated_at = datetime()
            """
            
            # Add custom properties
            for key, value in properties.items():
                query += f", t.{key} = ${key}"
            
            query += " RETURN t"
            
            params = {
                "name": topic_name,
                "category": category,
                **properties
            }
            
            async with self.driver.session() as session:
                result = await session.run(query, params)
                await result.consume()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create topic node: {e}")
            return False
    
    async def create_outcome_node(self, outcome_data: Dict[str, Any]) -> bool:
        """Create outcome node for tracking decision consequences"""
        try:
            query = """
            CREATE (o:Outcome {
                id: $id,
                decision_id: $decision_id,
                description: $description,
                outcome_type: $outcome_type,
                status: $status,
                success_score: $success_score,
                lessons_learned: $lessons_learned,
                occurred_at: datetime($occurred_at),
                created_at: datetime()
            })
            RETURN o
            """
            
            params = {
                "id": outcome_data.get("id", f"outcome_{datetime.utcnow().timestamp()}"),
                "decision_id": outcome_data["decision_id"],
                "description": outcome_data.get("description", ""),
                "outcome_type": outcome_data.get("outcome_type", "result"),
                "status": outcome_data.get("status", "completed"),
                "success_score": outcome_data.get("success_score", 0.0),
                "lessons_learned": outcome_data.get("lessons_learned", ""),
                "occurred_at": outcome_data.get("occurred_at", datetime.utcnow()).isoformat()
            }
            
            async with self.driver.session() as session:
                result = await session.run(query, params)
                await result.consume()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create outcome node: {e}")
            return False
    
    # Relationship Creation Methods
    async def create_meeting_decision_relationship(self, meeting_id: str, decision_id: str) -> bool:
        """Link meeting to decision made during it"""
        try:
            query = """
            MATCH (m:Meeting {id: $meeting_id})
            MATCH (d:Decision {id: $decision_id})
            MERGE (m)-[r:RESULTED_IN]->(d)
            SET r.created_at = datetime()
            RETURN r
            """
            
            params = {
                "meeting_id": meeting_id,
                "decision_id": decision_id
            }
            
            async with self.driver.session() as session:
                result = await session.run(query, params)
                await result.consume()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create meeting-decision relationship: {e}")
            return False
    
    async def create_meeting_topic_relationship(self, meeting_id: str, topic_name: str, relevance: float = 1.0) -> bool:
        """Link meeting to topics discussed"""
        try:
            query = """
            MATCH (m:Meeting {id: $meeting_id})
            MERGE (t:Topic {name: $topic_name})
            MERGE (m)-[r:DISCUSSED]->(t)
            SET r.relevance = $relevance,
                r.created_at = datetime()
            RETURN r
            """
            
            params = {
                "meeting_id": meeting_id,
                "topic_name": topic_name,
                "relevance": relevance
            }
            
            async with self.driver.session() as session:
                result = await session.run(query, params)
                await result.consume()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create meeting-topic relationship: {e}")
            return False
    
    async def create_decision_topic_relationship(self, decision_id: str, topic_name: str, impact: float = 1.0) -> bool:
        """Link decision to topics it affects"""
        try:
            query = """
            MATCH (d:Decision {id: $decision_id})
            MERGE (t:Topic {name: $topic_name})
            MERGE (d)-[r:AFFECTS]->(t)
            SET r.impact = $impact,
                r.created_at = datetime()
            RETURN r
            """
            
            params = {
                "decision_id": decision_id,
                "topic_name": topic_name,
                "impact": impact
            }
            
            async with self.driver.session() as session:
                result = await session.run(query, params)
                await result.consume()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create decision-topic relationship: {e}")
            return False
    
    async def create_decision_outcome_relationship(self, decision_id: str, outcome_id: str) -> bool:
        """Link decision to its outcome/consequence"""
        try:
            query = """
            MATCH (d:Decision {id: $decision_id})
            MATCH (o:Outcome {id: $outcome_id})
            MERGE (d)-[r:LED_TO]->(o)
            SET r.created_at = datetime()
            RETURN r
            """
            
            params = {
                "decision_id": decision_id,
                "outcome_id": outcome_id
            }
            
            async with self.driver.session() as session:
                result = await session.run(query, params)
                await result.consume()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create decision-outcome relationship: {e}")
            return False
    
    async def create_decision_dependency_relationship(
        self, 
        decision_id: str, 
        depends_on_decision_id: str, 
        dependency_type: str = "depends_on"
    ) -> bool:
        """Create dependency between decisions"""
        try:
            query = """
            MATCH (d1:Decision {id: $decision_id})
            MATCH (d2:Decision {id: $depends_on_decision_id})
            MERGE (d1)-[r:DEPENDS_ON]->(d2)
            SET r.dependency_type = $dependency_type,
                r.created_at = datetime()
            RETURN r
            """
            
            params = {
                "decision_id": decision_id,
                "depends_on_decision_id": depends_on_decision_id,
                "dependency_type": dependency_type
            }
            
            async with self.driver.session() as session:
                result = await session.run(query, params)
                await result.consume()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create decision dependency: {e}")
            return False
    
    # Decision Flow Analysis
    async def analyze_decision_flow(self, decision_id: str) -> Dict[str, Any]:
        """Analyze the complete flow of a decision and its consequences"""
        try:
            query = """
            MATCH (d:Decision {id: $decision_id})
            OPTIONAL MATCH (d)-[:LED_TO]->(o:Outcome)
            OPTIONAL MATCH (d)-[:AFFECTS]->(t:Topic)
            OPTIONAL MATCH (m:Meeting)-[:RESULTED_IN]->(d)
            OPTIONAL MATCH (d)-[:DEPENDS_ON]->(dep:Decision)
            OPTIONAL MATCH (future:Decision)-[:DEPENDS_ON]->(d)
            
            RETURN d as decision,
                   collect(DISTINCT o) as outcomes,
                   collect(DISTINCT t) as affected_topics,
                   collect(DISTINCT m) as source_meetings,
                   collect(DISTINCT dep) as dependencies,
                   collect(DISTINCT future) as influenced_decisions
            """
            
            params = {"decision_id": decision_id}
            
            async with self.driver.session() as session:
                result = await session.run(query, params)
                record = await result.single()
                
                if not record:
                    return {}
                
                return {
                    "decision": dict(record["decision"]) if record["decision"] else {},
                    "outcomes": [dict(o) for o in record["outcomes"] if o],
                    "affected_topics": [dict(t) for t in record["affected_topics"] if t],
                    "source_meetings": [dict(m) for m in record["source_meetings"] if m],
                    "dependencies": [dict(d) for d in record["dependencies"] if d],
                    "influenced_decisions": [dict(d) for d in record["influenced_decisions"] if d]
                }
            
        except Exception as e:
            logger.error(f"Failed to analyze decision flow: {e}")
            return {}
    
    async def get_topic_decision_history(self, topic_name: str, organization_id: str) -> List[Dict[str, Any]]:
        """Get history of decisions related to a specific topic"""
        try:
            query = """
            MATCH (t:Topic {name: $topic_name})
            MATCH (d:Decision)-[:AFFECTS]->(t)
            WHERE d.organization_id = $organization_id
            OPTIONAL MATCH (d)-[:LED_TO]->(o:Outcome)
            
            RETURN d as decision,
                   collect(o) as outcomes
            ORDER BY d.decided_at DESC
            """
            
            params = {
                "topic_name": topic_name,
                "organization_id": organization_id
            }
            
            async with self.driver.session() as session:
                result = await session.run(query, params)
                records = await result.data()
                
                history = []
                for record in records:
                    decision_data = dict(record["decision"])
                    outcomes_data = [dict(o) for o in record["outcomes"] if o]
                    
                    history.append({
                        "decision": decision_data,
                        "outcomes": outcomes_data
                    })
                
                return history
            
        except Exception as e:
            logger.error(f"Failed to get topic decision history: {e}")
            return []
    
    async def find_similar_decision_patterns(
        self, 
        decision_type: str, 
        organization_id: str, 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Find decisions with similar patterns and outcomes"""
        try:
            query = """
            MATCH (d:Decision {decision_type: $decision_type})
            WHERE d.organization_id = $organization_id
            OPTIONAL MATCH (d)-[:LED_TO]->(o:Outcome)
            OPTIONAL MATCH (d)-[:AFFECTS]->(t:Topic)
            
            RETURN d as decision,
                   collect(DISTINCT o) as outcomes,
                   collect(DISTINCT t.name) as topics,
                   avg(o.success_score) as avg_success
            ORDER BY avg_success DESC, d.decided_at DESC
            LIMIT $limit
            """
            
            params = {
                "decision_type": decision_type,
                "organization_id": organization_id,
                "limit": limit
            }
            
            async with self.driver.session() as session:
                result = await session.run(query, params)
                records = await result.data()
                
                patterns = []
                for record in records:
                    patterns.append({
                        "decision": dict(record["decision"]),
                        "outcomes": [dict(o) for o in record["outcomes"] if o],
                        "topics": record["topics"],
                        "average_success": record["avg_success"] or 0.0
                    })
                
                return patterns
            
        except Exception as e:
            logger.error(f"Failed to find similar decision patterns: {e}")
            return []
    
    async def get_decision_chain(self, starting_decision_id: str) -> Dict[str, Any]:
        """Get the complete chain of decisions that followed from an initial decision"""
        try:
            query = """
            MATCH path = (start:Decision {id: $starting_decision_id})<-[:DEPENDS_ON*]-(future:Decision)
            RETURN start,
                   collect(DISTINCT future) as future_decisions,
                   length(path) as chain_length
            ORDER BY chain_length DESC
            """
            
            params = {"starting_decision_id": starting_decision_id}
            
            async with self.driver.session() as session:
                result = await session.run(query, params)
                record = await result.single()
                
                if not record:
                    return {}
                
                return {
                    "starting_decision": dict(record["start"]),
                    "future_decisions": [dict(d) for d in record["future_decisions"]],
                    "chain_length": record["chain_length"]
                }
            
        except Exception as e:
            logger.error(f"Failed to get decision chain: {e}")
            return {}
    
    # Analytics and Insights
    async def get_topic_trends(self, organization_id: str, days: int = 90) -> List[Dict[str, Any]]:
        """Get trending topics in meetings and decisions"""
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=days)
            
            query = """
            MATCH (t:Topic)
            OPTIONAL MATCH (m:Meeting)-[:DISCUSSED]->(t)
            WHERE m.organization_id = $organization_id 
              AND m.scheduled_at > datetime($cutoff_date)
            OPTIONAL MATCH (d:Decision)-[:AFFECTS]->(t)
            WHERE d.organization_id = $organization_id 
              AND d.decided_at > datetime($cutoff_date)
            
            WITH t, 
                 count(DISTINCT m) as meeting_mentions,
                 count(DISTINCT d) as decision_mentions,
                 avg(d.success_rating) as avg_success
            
            WHERE meeting_mentions > 0 OR decision_mentions > 0
            
            RETURN t.name as topic,
                   t.category as category,
                   meeting_mentions,
                   decision_mentions,
                   meeting_mentions + decision_mentions as total_mentions,
                   avg_success
            ORDER BY total_mentions DESC, avg_success DESC
            LIMIT 20
            """
            
            params = {
                "organization_id": organization_id,
                "cutoff_date": cutoff_date.isoformat()
            }
            
            async with self.driver.session() as session:
                result = await session.run(query, params)
                records = await result.data()
                
                return [dict(record) for record in records]
            
        except Exception as e:
            logger.error(f"Failed to get topic trends: {e}")
            return []
    
    async def analyze_decision_success_patterns(self, organization_id: str) -> Dict[str, Any]:
        """Analyze patterns in successful vs failed decisions"""
        try:
            query = """
            MATCH (d:Decision)
            WHERE d.organization_id = $organization_id
              AND d.success_rating IS NOT NULL
            OPTIONAL MATCH (d)-[:AFFECTS]->(t:Topic)
            
            WITH d, collect(t.name) as topics
            
            RETURN d.decision_type as decision_type,
                   avg(d.success_rating) as avg_success,
                   count(d) as total_decisions,
                   collect(topics) as common_topics,
                   avg(d.implementation_progress) as avg_progress
            ORDER BY avg_success DESC
            """
            
            params = {"organization_id": organization_id}
            
            async with self.driver.session() as session:
                result = await session.run(query, params)
                records = await result.data()
                
                return {
                    "success_patterns": [dict(record) for record in records],
                    "analysis_date": datetime.utcnow().isoformat()
                }
            
        except Exception as e:
            logger.error(f"Failed to analyze decision success patterns: {e}")
            return {}
    
    # Bulk Operations
    async def index_meeting_with_relationships(self, meeting_data: Dict[str, Any]) -> bool:
        """Index meeting and create all related relationships"""
        try:
            # Create meeting node
            await self.create_meeting_node(meeting_data)
            
            # Extract and create topic relationships
            topics = meeting_data.get("key_topics", [])
            for topic in topics:
                await self.create_topic_node(topic, "meeting_topic")
                await self.create_meeting_topic_relationship(meeting_data["id"], topic)
            
            # If there are decisions from this meeting
            decisions = meeting_data.get("decisions_made", [])
            for decision_id in decisions:
                await self.create_meeting_decision_relationship(meeting_data["id"], decision_id)
            
            logger.info(f"Fully indexed meeting with relationships: {meeting_data['id']}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to index meeting with relationships: {e}")
            return False
    
    async def index_decision_with_relationships(self, decision_data: Dict[str, Any]) -> bool:
        """Index decision and create all related relationships"""
        try:
            # Create decision node
            await self.create_decision_node(decision_data)
            
            # Create topic relationships
            impact_areas = decision_data.get("impact_areas", [])
            for area in impact_areas:
                await self.create_topic_node(area, "impact_area")
                await self.create_decision_topic_relationship(decision_data["id"], area)
            
            # Create meeting relationship if specified
            meeting_id = decision_data.get("meeting_id")
            if meeting_id:
                await self.create_meeting_decision_relationship(meeting_id, decision_data["id"])
            
            logger.info(f"Fully indexed decision with relationships: {decision_data['id']}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to index decision with relationships: {e}")
            return False
    
    # Utility Methods
    async def cleanup(self) -> None:
        """Cleanup Neo4j connections"""
        if self.driver:
            await self.driver.close()
            self.is_connected = False
            logger.info("Neo4j service cleaned up")
    
    async def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            query = """
            MATCH (n)
            OPTIONAL MATCH ()-[r]->()
            RETURN 
                count(DISTINCT n) as total_nodes,
                count(DISTINCT r) as total_relationships,
                collect(DISTINCT labels(n)) as node_types,
                collect(DISTINCT type(r)) as relationship_types
            """
            
            async with self.driver.session() as session:
                result = await session.run(query)
                record = await result.single()
                
                return {
                    "total_nodes": record["total_nodes"],
                    "total_relationships": record["total_relationships"],
                    "node_types": [item for sublist in record["node_types"] for item in sublist],
                    "relationship_types": [t for t in record["relationship_types"] if t]
                }
            
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            return {}


# Global service instance
neo4j_service = None


async def get_neo4j_service() -> Neo4jService:
    """Get Neo4j service instance"""
    global neo4j_service
    if not neo4j_service:
        neo4j_service = await Neo4jService.initialize()
    return neo4j_service
