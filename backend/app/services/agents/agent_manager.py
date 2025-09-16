"""
Agent management service for Clario
Coordinates agent creation, meetings, and decision-making
"""

from typing import List, Dict, Any, Optional
import asyncio
import logging
from datetime import datetime
import json

from app.services.agents.strands_service import get_strands_service, AgentConfig
from app.services.vector.weaviate_service import get_weaviate_service
from app.services.graph.neo4j_service import get_neo4j_service
from app.core.logging import get_logger

logger = get_logger("agent_manager")


class AgentManager:
    """Manages AI agents and their participation in meetings"""
    
    def __init__(self):
        self.strands_service = None
        self.weaviate_service = None
        self.neo4j_service = None
        self.active_meetings = {}  # meeting_id -> agent_ids
        
    async def initialize(self) -> None:
        """Initialize agent manager"""
        self.strands_service = await get_strands_service()
        self.weaviate_service = await get_weaviate_service()
        self.neo4j_service = await get_neo4j_service()
        logger.info("Agent manager initialized")
    
    # Agent Creation and Management
    async def create_agent_for_user(
        self,
        user_id: str,
        organization_id: str,
        force_recreate: bool = False
    ) -> Dict[str, Any]:
        """Create an AI agent for a user based on their profile"""
        try:
            # Check if agent already exists
            existing_agent = await self._check_existing_agent(user_id)
            if existing_agent and not force_recreate:
                return {
                    "success": True,
                    "agent_id": existing_agent["agent_id"],
                    "message": "Agent already exists",
                    "existing": True
                }
            
            # Get user data and decision profile
            user_data = await self._get_user_data(user_id)
            decision_profile = await self._get_decision_profile(user_id)
            organization_context = await self._get_organization_context(organization_id)
            
            if not user_data:
                return {
                    "success": False,
                    "error": "User data not found"
                }
            
            # Create agent using Strands service
            agent_id = await self.strands_service.create_agent(
                user_data=user_data,
                decision_profile=decision_profile or {},
                organization_context=organization_context or {}
            )
            
            # Store agent information in database
            await self._store_agent_info(agent_id, user_id, organization_id)
            
            # Index agent context in vector database
            await self._index_agent_context(agent_id, user_data, decision_profile)
            
            return {
                "success": True,
                "agent_id": agent_id,
                "message": "Agent created successfully",
                "existing": False
            }
            
        except Exception as e:
            logger.error(f"Failed to create agent for user {user_id}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def conduct_meeting_with_agents(
        self,
        meeting_id: str,
        agenda_items: List[Dict[str, Any]],
        participant_agents: List[str],
        meeting_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Conduct a meeting with AI agents"""
        try:
            if not participant_agents:
                return {
                    "success": False,
                    "error": "No agents specified for meeting"
                }
            
            # Register active meeting
            self.active_meetings[meeting_id] = participant_agents
            
            meeting_results = {
                "meeting_id": meeting_id,
                "participants": participant_agents,
                "agenda_results": [],
                "decisions": [],
                "discussion_summary": "",
                "started_at": datetime.utcnow().isoformat()
            }
            
            # Process each agenda item
            for i, agenda_item in enumerate(agenda_items):
                logger.info(f"Processing agenda item {i+1}: {agenda_item.get('title', 'Untitled')}")
                
                # Get relevant context from vector database
                agenda_context = await self._get_agenda_context(
                    agenda_item, meeting_context.get("organization_id")
                )
                
                # Conduct agent discussion
                discussion_log = await self.strands_service.conduct_agent_discussion(
                    agent_ids=participant_agents,
                    agenda_item=agenda_item,
                    meeting_context={**meeting_context, **agenda_context},
                    discussion_rounds=3
                )
                
                # Generate decision if required
                decision = None
                if agenda_item.get("requires_decision", False):
                    decision = await self._generate_consensus_decision(
                        participant_agents,
                        agenda_item,
                        discussion_log,
                        meeting_context
                    )
                
                agenda_result = {
                    "agenda_item": agenda_item,
                    "discussion_log": discussion_log,
                    "decision": decision,
                    "context_used": agenda_context,
                    "processed_at": datetime.utcnow().isoformat()
                }
                
                meeting_results["agenda_results"].append(agenda_result)
                
                if decision:
                    meeting_results["decisions"].append(decision)
                
                # Store discussion in graph database
                await self._store_discussion_in_graph(
                    meeting_id, agenda_item, discussion_log, decision
                )
            
            # Generate meeting summary
            meeting_results["discussion_summary"] = await self._generate_meeting_summary(
                meeting_results["agenda_results"]
            )
            
            meeting_results["completed_at"] = datetime.utcnow().isoformat()
            
            # Clean up active meeting
            if meeting_id in self.active_meetings:
                del self.active_meetings[meeting_id]
            
            logger.info(f"Completed meeting {meeting_id} with {len(participant_agents)} agents")
            return {
                "success": True,
                "meeting_results": meeting_results
            }
            
        except Exception as e:
            logger.error(f"Failed to conduct meeting with agents: {e}")
            # Clean up active meeting on error
            if meeting_id in self.active_meetings:
                del self.active_meetings[meeting_id]
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_agent_recommendation(
        self,
        agent_id: str,
        decision_context: Dict[str, Any],
        organization_id: str
    ) -> Dict[str, Any]:
        """Get a recommendation from a specific agent"""
        try:
            # Get historical context
            historical_context = await self._get_decision_historical_context(
                decision_context, organization_id
            )
            
            # Get agent recommendation
            recommendation = await self.strands_service.generate_decision_recommendation(
                agent_id=agent_id,
                agenda_item=decision_context,
                discussion_log=[],  # No prior discussion
                organizational_context=historical_context
            )
            
            return {
                "success": True,
                "recommendation": recommendation
            }
            
        except Exception as e:
            logger.error(f"Failed to get agent recommendation: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    # Context and Data Retrieval
    async def _get_user_data(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user data from database"""
        # In a real implementation, this would query the database
        # For now, return mock data
        return {
            "id": user_id,
            "full_name": f"User {user_id}",
            "email": f"user{user_id}@example.com",
            "job_title": "Software Engineer",
            "department": "Engineering",
            "role": "member"
        }
    
    async def _get_decision_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user's decision profile"""
        # In a real implementation, this would query the decision_profiles table
        return {
            "analytical_score": 75,
            "intuitive_score": 45,
            "collaborative_score": 65,
            "decision_style": "analytical",
            "risk_tolerance": "medium",
            "communication_style": "direct",
            "decision_speed": "moderate"
        }
    
    async def _get_organization_context(self, organization_id: str) -> Optional[Dict[str, Any]]:
        """Get organization context"""
        # In a real implementation, this would query the organizations table
        return {
            "id": organization_id,
            "name": "Sample Organization",
            "description": "A technology company focused on innovation",
            "culture": "collaborative and data-driven",
            "decision_making_style": "consensus-based",
            "values": ["innovation", "transparency", "quality"]
        }
    
    async def _get_agenda_context(
        self,
        agenda_item: Dict[str, Any],
        organization_id: str
    ) -> Dict[str, Any]:
        """Get relevant context for agenda item from vector database"""
        try:
            if not self.weaviate_service:
                return {}
            
            # Search for similar meetings and decisions
            agenda_text = f"{agenda_item.get('title', '')} {agenda_item.get('description', '')}"
            
            context = await self.weaviate_service.get_meeting_context(
                current_agenda=agenda_text,
                organization_id=organization_id
            )
            
            return {
                "similar_meetings": context.get("similar_meetings", []),
                "related_decisions": context.get("related_decisions", []),
                "context_strength": context.get("context_strength", 0)
            }
            
        except Exception as e:
            logger.error(f"Failed to get agenda context: {e}")
            return {}
    
    async def _get_decision_historical_context(
        self,
        decision_context: Dict[str, Any],
        organization_id: str
    ) -> Dict[str, Any]:
        """Get historical context for decision-making"""
        try:
            # Get similar decisions from graph database
            if self.neo4j_service:
                decision_type = decision_context.get("decision_type", "operational")
                patterns = await self.neo4j_service.find_similar_decision_patterns(
                    decision_type=decision_type,
                    organization_id=organization_id,
                    limit=5
                )
                
                return {
                    "similar_decisions": patterns,
                    "organization_id": organization_id
                }
            
            return {}
            
        except Exception as e:
            logger.error(f"Failed to get decision historical context: {e}")
            return {}
    
    # Decision Generation
    async def _generate_consensus_decision(
        self,
        agent_ids: List[str],
        agenda_item: Dict[str, Any],
        discussion_log: List[Dict[str, Any]],
        meeting_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate consensus decision from multiple agents"""
        try:
            recommendations = []
            
            # Get recommendation from each agent
            for agent_id in agent_ids:
                try:
                    recommendation = await self.strands_service.generate_decision_recommendation(
                        agent_id=agent_id,
                        agenda_item=agenda_item,
                        discussion_log=discussion_log,
                        organizational_context=meeting_context
                    )
                    recommendations.append(recommendation)
                except Exception as e:
                    logger.error(f"Failed to get recommendation from agent {agent_id}: {e}")
            
            if not recommendations:
                return {
                    "decision": "No consensus reached",
                    "reasoning": "Unable to get recommendations from agents",
                    "confidence": 0.0,
                    "consensus_level": "none"
                }
            
            # Analyze consensus
            consensus_analysis = self._analyze_consensus(recommendations)
            
            # Generate final decision
            final_decision = {
                "decision": consensus_analysis["consensus_decision"],
                "reasoning": consensus_analysis["consensus_reasoning"],
                "confidence": consensus_analysis["confidence"],
                "consensus_level": consensus_analysis["consensus_level"],
                "individual_recommendations": recommendations,
                "alternative_options": consensus_analysis["alternatives"],
                "generated_at": datetime.utcnow().isoformat()
            }
            
            return final_decision
            
        except Exception as e:
            logger.error(f"Failed to generate consensus decision: {e}")
            return {
                "decision": "Decision generation failed",
                "reasoning": str(e),
                "confidence": 0.0,
                "consensus_level": "error"
            }
    
    def _analyze_consensus(self, recommendations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze consensus among agent recommendations"""
        if not recommendations:
            return {
                "consensus_decision": "No recommendations available",
                "consensus_reasoning": "",
                "confidence": 0.0,
                "consensus_level": "none",
                "alternatives": []
            }
        
        # Extract recommendations and analyze agreement
        decisions = [r.get("recommendation", "") for r in recommendations]
        confidences = [r.get("confidence", 0.0) for r in recommendations]
        
        # Simple consensus analysis - in production, this would be more sophisticated
        avg_confidence = sum(confidences) / len(confidences)
        
        # Check for clear majority
        decision_counts = {}
        for decision in decisions:
            # Normalize decision text for comparison
            normalized = decision.lower().strip()
            decision_counts[normalized] = decision_counts.get(normalized, 0) + 1
        
        if decision_counts:
            most_common = max(decision_counts, key=decision_counts.get)
            consensus_strength = decision_counts[most_common] / len(decisions)
            
            if consensus_strength >= 0.7:
                consensus_level = "strong"
            elif consensus_strength >= 0.5:
                consensus_level = "moderate"
            else:
                consensus_level = "weak"
        else:
            most_common = "No clear consensus"
            consensus_level = "none"
        
        # Combine reasoning from all recommendations
        all_reasoning = [r.get("reasoning", "") for r in recommendations if r.get("reasoning")]
        combined_reasoning = " | ".join(all_reasoning)
        
        return {
            "consensus_decision": most_common,
            "consensus_reasoning": combined_reasoning,
            "confidence": avg_confidence,
            "consensus_level": consensus_level,
            "alternatives": list(set(decisions) - {most_common})
        }
    
    # Storage and Persistence
    async def _check_existing_agent(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Check if agent already exists for user"""
        # In a real implementation, this would query the agents table
        return None
    
    async def _store_agent_info(
        self,
        agent_id: str,
        user_id: str,
        organization_id: str
    ) -> None:
        """Store agent information in database"""
        # In a real implementation, this would store in the agents table
        logger.info(f"Stored agent info: {agent_id} for user {user_id}")
    
    async def _index_agent_context(
        self,
        agent_id: str,
        user_data: Dict[str, Any],
        decision_profile: Dict[str, Any]
    ) -> None:
        """Index agent context in vector database"""
        try:
            if self.weaviate_service and decision_profile:
                # This would index the agent's profile for context retrieval
                logger.info(f"Indexed agent context for {agent_id}")
        except Exception as e:
            logger.error(f"Failed to index agent context: {e}")
    
    async def _store_discussion_in_graph(
        self,
        meeting_id: str,
        agenda_item: Dict[str, Any],
        discussion_log: List[Dict[str, Any]],
        decision: Optional[Dict[str, Any]]
    ) -> None:
        """Store discussion results in graph database"""
        try:
            if not self.neo4j_service:
                return
            
            # Store each discussion entry as a node
            for entry in discussion_log:
                discussion_data = {
                    "id": f"discussion_{meeting_id}_{entry.get('agent_id')}_{entry.get('round')}",
                    "meeting_id": meeting_id,
                    "agent_id": entry.get("agent_id"),
                    "content": entry.get("message", ""),
                    "message_type": "discussion",
                    "reasoning": entry.get("reasoning", ""),
                    "confidence_level": entry.get("confidence", 0.7),
                    "timestamp": entry.get("timestamp")
                }
                
                # Create discussion node (this would need to be implemented in neo4j_service)
                # await self.neo4j_service.create_discussion_node(discussion_data)
            
            logger.info(f"Stored discussion in graph for meeting {meeting_id}")
            
        except Exception as e:
            logger.error(f"Failed to store discussion in graph: {e}")
    
    async def _generate_meeting_summary(
        self,
        agenda_results: List[Dict[str, Any]]
    ) -> str:
        """Generate summary of meeting results"""
        try:
            summary_parts = []
            
            summary_parts.append(f"Meeting completed with {len(agenda_results)} agenda items.")
            
            decisions_made = sum(1 for result in agenda_results if result.get("decision"))
            if decisions_made > 0:
                summary_parts.append(f"{decisions_made} decisions were made.")
            
            # Add key discussion points
            for i, result in enumerate(agenda_results):
                agenda_item = result.get("agenda_item", {})
                title = agenda_item.get("title", f"Item {i+1}")
                
                if result.get("decision"):
                    decision_text = result["decision"].get("decision", "Decision made")
                    summary_parts.append(f"{title}: {decision_text}")
                else:
                    summary_parts.append(f"{title}: Discussion completed")
            
            return " ".join(summary_parts)
            
        except Exception as e:
            logger.error(f"Failed to generate meeting summary: {e}")
            return "Meeting completed successfully."
    
    # Utility Methods
    async def get_active_meetings(self) -> Dict[str, List[str]]:
        """Get currently active meetings"""
        return self.active_meetings.copy()
    
    async def get_agent_list(self, organization_id: str) -> List[Dict[str, Any]]:
        """Get list of agents for organization"""
        if self.strands_service:
            return await self.strands_service.list_agents(organization_id)
        return []
    
    async def cleanup(self) -> None:
        """Cleanup agent manager"""
        if self.strands_service:
            await self.strands_service.cleanup()
        self.active_meetings.clear()
        logger.info("Agent manager cleaned up")


# Global agent manager instance
agent_manager = None


async def get_agent_manager() -> AgentManager:
    """Get agent manager instance"""
    global agent_manager
    if not agent_manager:
        agent_manager = AgentManager()
        await agent_manager.initialize()
    return agent_manager
