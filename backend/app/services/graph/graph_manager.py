"""
Graph management service for Clario
Coordinates graph operations and maintains data consistency
"""

from typing import List, Dict, Any, Optional
import asyncio
import logging
from datetime import datetime
import json

from app.services.graph.neo4j_service import get_neo4j_service
from app.services.graph.decision_flow_analyzer import get_decision_flow_analyzer
from app.core.logging import get_logger

logger = get_logger("graph_manager")


class GraphManager:
    """Manages all graph database operations for Clario"""
    
    def __init__(self):
        self.neo4j_service = None
        self.flow_analyzer = None
        
    async def initialize(self) -> None:
        """Initialize graph manager"""
        self.neo4j_service = await get_neo4j_service()
        self.flow_analyzer = await get_decision_flow_analyzer()
        logger.info("Graph manager initialized")
    
    # Meeting Operations
    async def index_meeting_with_full_context(
        self, 
        meeting_data: Dict[str, Any],
        participants: List[str] = None,
        topics_discussed: List[str] = None
    ) -> bool:
        """Index meeting with complete context and relationships"""
        try:
            # Index the basic meeting
            success = await self.neo4j_service.index_meeting_with_relationships(meeting_data)
            
            if not success:
                return False
            
            meeting_id = meeting_data["id"]
            
            # Add participant relationships
            if participants:
                await self._create_participant_relationships(meeting_id, participants)
            
            # Add topic relationships with relevance scoring
            if topics_discussed:
                await self._create_topic_relationships_with_scoring(
                    meeting_id, topics_discussed, meeting_data
                )
            
            # Create temporal relationships (meetings in sequence)
            await self._create_temporal_relationships(meeting_data)
            
            logger.info(f"Fully indexed meeting with context: {meeting_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to index meeting with full context: {e}")
            return False
    
    async def index_decision_with_consequences(
        self, 
        decision_data: Dict[str, Any],
        outcomes: List[Dict[str, Any]] = None,
        dependencies: List[str] = None
    ) -> bool:
        """Index decision with outcomes and dependency relationships"""
        try:
            # Index the basic decision
            success = await self.neo4j_service.index_decision_with_relationships(decision_data)
            
            if not success:
                return False
            
            decision_id = decision_data["id"]
            
            # Create outcome nodes and relationships
            if outcomes:
                for outcome in outcomes:
                    await self.neo4j_service.create_outcome_node(outcome)
                    await self.neo4j_service.create_decision_outcome_relationship(
                        decision_id, outcome.get("id", f"outcome_{datetime.utcnow().timestamp()}")
                    )
            
            # Create dependency relationships
            if dependencies:
                for dep_id in dependencies:
                    await self.neo4j_service.create_decision_dependency_relationship(
                        decision_id, dep_id
                    )
            
            # Analyze impact and create additional relationships
            await self._analyze_and_create_impact_relationships(decision_data)
            
            logger.info(f"Fully indexed decision with consequences: {decision_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to index decision with consequences: {e}")
            return False
    
    # Advanced Relationship Creation
    async def _create_participant_relationships(
        self, 
        meeting_id: str, 
        participants: List[str]
    ) -> None:
        """Create participant relationships with meeting"""
        try:
            for participant in participants:
                query = """
                MATCH (m:Meeting {id: $meeting_id})
                MERGE (p:Participant {name: $participant})
                MERGE (p)-[r:ATTENDED]->(m)
                SET r.created_at = datetime()
                """
                
                params = {
                    "meeting_id": meeting_id,
                    "participant": participant
                }
                
                async with self.neo4j_service.driver.session() as session:
                    await session.run(query, params)
            
        except Exception as e:
            logger.error(f"Failed to create participant relationships: {e}")
    
    async def _create_topic_relationships_with_scoring(
        self,
        meeting_id: str,
        topics: List[str],
        meeting_data: Dict[str, Any]
    ) -> None:
        """Create topic relationships with relevance scoring"""
        try:
            for topic in topics:
                # Calculate relevance based on topic frequency in agenda/description
                relevance = self._calculate_topic_relevance(topic, meeting_data)
                
                await self.neo4j_service.create_meeting_topic_relationship(
                    meeting_id, topic, relevance
                )
                
        except Exception as e:
            logger.error(f"Failed to create topic relationships with scoring: {e}")
    
    async def _create_temporal_relationships(self, meeting_data: Dict[str, Any]) -> None:
        """Create temporal relationships between meetings"""
        try:
            organization_id = meeting_data.get("organization_id")
            scheduled_at = meeting_data.get("scheduled_at")
            
            if not organization_id or not scheduled_at:
                return
            
            # Find the previous meeting in the same organization
            query = """
            MATCH (current:Meeting {id: $current_id})
            MATCH (previous:Meeting)
            WHERE previous.organization_id = $organization_id
              AND previous.scheduled_at < current.scheduled_at
              AND previous.id <> current.id
            WITH previous
            ORDER BY previous.scheduled_at DESC
            LIMIT 1
            
            MATCH (current:Meeting {id: $current_id})
            MERGE (previous)-[r:PRECEDED]->(current)
            SET r.time_gap_hours = duration.between(previous.scheduled_at, current.scheduled_at).hours
            """
            
            params = {
                "current_id": meeting_data["id"],
                "organization_id": organization_id
            }
            
            async with self.neo4j_service.driver.session() as session:
                await session.run(query, params)
                
        except Exception as e:
            logger.error(f"Failed to create temporal relationships: {e}")
    
    async def _analyze_and_create_impact_relationships(
        self, 
        decision_data: Dict[str, Any]
    ) -> None:
        """Analyze decision impact and create additional relationships"""
        try:
            decision_id = decision_data["id"]
            decision_type = decision_data.get("decision_type", "operational")
            
            # Create relationships based on decision type
            if decision_type == "strategic":
                await self._create_strategic_impact_relationships(decision_data)
            elif decision_type == "financial":
                await self._create_financial_impact_relationships(decision_data)
            elif decision_type == "technical":
                await self._create_technical_impact_relationships(decision_data)
            
        except Exception as e:
            logger.error(f"Failed to analyze and create impact relationships: {e}")
    
    # Analysis and Insights
    async def get_comprehensive_decision_analysis(
        self, 
        decision_id: str
    ) -> Dict[str, Any]:
        """Get comprehensive analysis of a decision and its impact"""
        try:
            # Get basic flow analysis
            consequences = await self.flow_analyzer.analyze_decision_consequences(decision_id)
            
            # Get decision chain
            chain = await self.flow_analyzer.trace_decision_chain(decision_id)
            
            # Get related topics and their history
            decision_flow = await self.neo4j_service.analyze_decision_flow(decision_id)
            affected_topics = decision_flow.get("affected_topics", [])
            
            topic_histories = {}
            for topic in affected_topics:
                topic_name = topic.get("name", "")
                if topic_name:
                    history = await self.neo4j_service.get_topic_decision_history(
                        topic_name, 
                        decision_flow.get("decision", {}).get("organization_id", "")
                    )
                    topic_histories[topic_name] = history
            
            return {
                "decision_consequences": consequences,
                "decision_chain": chain,
                "topic_histories": topic_histories,
                "related_decisions": decision_flow.get("influenced_decisions", []),
                "analysis_timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to get comprehensive decision analysis: {e}")
            return {"error": str(e)}
    
    async def generate_organization_insights(
        self, 
        organization_id: str
    ) -> Dict[str, Any]:
        """Generate comprehensive organizational insights"""
        try:
            # Get decision patterns
            patterns = await self.flow_analyzer.map_organizational_decision_patterns(
                organization_id
            )
            
            # Get bottlenecks
            bottlenecks = await self.flow_analyzer.find_decision_bottlenecks(
                organization_id
            )
            
            # Get topic trends
            topic_trends = await self.neo4j_service.get_topic_trends(
                organization_id
            )
            
            # Get database stats for this organization
            org_stats = await self._get_organization_graph_stats(organization_id)
            
            # Generate recommendations
            recommendations = await self._generate_comprehensive_recommendations(
                patterns, bottlenecks, topic_trends
            )
            
            return {
                "organization_id": organization_id,
                "decision_patterns": patterns,
                "bottlenecks": bottlenecks,
                "topic_trends": topic_trends,
                "graph_statistics": org_stats,
                "recommendations": recommendations,
                "insight_score": self._calculate_insight_score(patterns, bottlenecks),
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to generate organization insights: {e}")
            return {"error": str(e)}
    
    # Utility Methods
    def _calculate_topic_relevance(
        self, 
        topic: str, 
        meeting_data: Dict[str, Any]
    ) -> float:
        """Calculate how relevant a topic is to a meeting"""
        relevance = 0.0
        topic_lower = topic.lower()
        
        # Check title
        title = meeting_data.get("title", "").lower()
        if topic_lower in title:
            relevance += 0.4
        
        # Check description
        description = meeting_data.get("description", "").lower()
        if topic_lower in description:
            relevance += 0.3
        
        # Check agenda
        agenda = meeting_data.get("agenda_text", "").lower()
        if topic_lower in agenda:
            relevance += 0.3
        
        return min(1.0, relevance)
    
    async def _create_strategic_impact_relationships(
        self, 
        decision_data: Dict[str, Any]
    ) -> None:
        """Create relationships for strategic decisions"""
        # Strategic decisions often impact multiple areas
        impact_areas = decision_data.get("impact_areas", [])
        for area in impact_areas:
            await self.neo4j_service.create_topic_node(area, "strategic_area")
    
    async def _create_financial_impact_relationships(
        self, 
        decision_data: Dict[str, Any]
    ) -> None:
        """Create relationships for financial decisions"""
        # Financial decisions might affect budget categories
        estimated_cost = decision_data.get("estimated_cost", 0)
        if estimated_cost > 0:
            cost_category = "high_cost" if estimated_cost > 100000 else "medium_cost" if estimated_cost > 10000 else "low_cost"
            await self.neo4j_service.create_topic_node(cost_category, "financial_impact")
    
    async def _create_technical_impact_relationships(
        self, 
        decision_data: Dict[str, Any]
    ) -> None:
        """Create relationships for technical decisions"""
        # Technical decisions might affect systems or technologies
        pass  # Implementation would depend on specific technical categorization
    
    async def _get_organization_graph_stats(self, organization_id: str) -> Dict[str, Any]:
        """Get graph statistics for a specific organization"""
        try:
            query = """
            MATCH (n)
            WHERE n.organization_id = $organization_id
            OPTIONAL MATCH (n)-[r]->()
            WHERE r.organization_id = $organization_id OR NOT EXISTS(r.organization_id)
            
            RETURN 
                count(DISTINCT n) as org_nodes,
                count(DISTINCT r) as org_relationships,
                collect(DISTINCT labels(n)) as org_node_types
            """
            
            params = {"organization_id": organization_id}
            
            async with self.neo4j_service.driver.session() as session:
                result = await session.run(query, params)
                record = await result.single()
                
                return {
                    "organization_nodes": record["org_nodes"] if record else 0,
                    "organization_relationships": record["org_relationships"] if record else 0,
                    "node_types": [item for sublist in record["org_node_types"] for item in sublist] if record else []
                }
                
        except Exception as e:
            logger.error(f"Failed to get organization graph stats: {e}")
            return {}
    
    async def _generate_comprehensive_recommendations(
        self,
        patterns: Dict[str, Any],
        bottlenecks: List[Dict[str, Any]],
        topic_trends: List[Dict[str, Any]]
    ) -> List[Dict[str, str]]:
        """Generate comprehensive recommendations"""
        recommendations = []
        
        # Analyze bottlenecks
        critical_bottlenecks = [b for b in bottlenecks if b.get("bottleneck_severity") == "critical"]
        if critical_bottlenecks:
            recommendations.append({
                "type": "urgent",
                "category": "bottlenecks",
                "title": "Resolve Critical Decision Bottlenecks",
                "description": f"Address {len(critical_bottlenecks)} critical bottlenecks that are blocking organizational progress"
            })
        
        # Analyze decision patterns
        if patterns.get("decision_type_patterns"):
            worst_performing = None
            worst_score = 1.0
            
            for decision_type, data in patterns["decision_type_patterns"].items():
                avg_success = data.get("average_success", 0)
                if avg_success < worst_score:
                    worst_score = avg_success
                    worst_performing = decision_type
            
            if worst_performing and worst_score < 0.6:
                recommendations.append({
                    "type": "improvement",
                    "category": "decision_quality",
                    "title": f"Improve {worst_performing.title()} Decision Process",
                    "description": f"Focus on improving {worst_performing} decisions which have a {worst_score:.1%} success rate"
                })
        
        # Analyze topic trends
        if topic_trends:
            top_topic = topic_trends[0]
            if top_topic.get("total_mentions", 0) > 10:
                recommendations.append({
                    "type": "insight",
                    "category": "trending",
                    "title": f"High Activity on {top_topic['topic']}",
                    "description": f"This topic has {top_topic['total_mentions']} mentions across meetings and decisions"
                })
        
        return recommendations
    
    def _calculate_insight_score(
        self,
        patterns: Dict[str, Any],
        bottlenecks: List[Dict[str, Any]]
    ) -> float:
        """Calculate an overall insight score for the organization"""
        score = 0.5  # Base score
        
        # Boost for good decision patterns
        if patterns.get("decision_type_patterns"):
            avg_success = sum(
                data.get("average_success", 0) 
                for data in patterns["decision_type_patterns"].values()
            ) / len(patterns["decision_type_patterns"])
            score += avg_success * 0.3
        
        # Penalize for bottlenecks
        critical_bottlenecks = len([b for b in bottlenecks if b.get("bottleneck_severity") == "critical"])
        score -= critical_bottlenecks * 0.1
        
        return max(0.0, min(1.0, score))


# Global graph manager instance
graph_manager = None


async def get_graph_manager() -> GraphManager:
    """Get graph manager instance"""
    global graph_manager
    if not graph_manager:
        graph_manager = GraphManager()
        await graph_manager.initialize()
    return graph_manager
