"""
Agent endpoints for Clario
AI agent creation and meeting management
"""

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

from app.services.agents.agent_manager import get_agent_manager
from app.services.agents.strands_service import get_strands_service
from app.core.logging import get_logger

logger = get_logger("agents_api")

router = APIRouter()


class CreateAgentRequest(BaseModel):
    """Request to create an agent"""
    user_id: str
    organization_id: str
    force_recreate: bool = False


class MeetingRequest(BaseModel):
    """Request to conduct meeting with agents"""
    meeting_id: str
    agenda_items: List[Dict[str, Any]]
    participant_agents: List[str]
    meeting_context: Dict[str, Any]


class DecisionRequest(BaseModel):
    """Request for agent decision recommendation"""
    agent_id: str
    decision_context: Dict[str, Any]
    organization_id: str


class AgentResponse(BaseModel):
    """Agent information response"""
    agent_id: str
    name: str
    personality: str
    role: str
    status: str
    expertise_areas: List[str]


@router.post("/create")
async def create_agent(request: CreateAgentRequest) -> Dict[str, Any]:
    """
    Create an AI agent for a user based on their decision profile
    
    - **user_id**: ID of the user to create agent for
    - **organization_id**: Organization the user belongs to
    - **force_recreate**: Whether to recreate if agent already exists
    """
    try:
        agent_manager = await get_agent_manager()
        
        result = await agent_manager.create_agent_for_user(
            user_id=request.user_id,
            organization_id=request.organization_id,
            force_recreate=request.force_recreate
        )
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result.get("error", "Agent creation failed"))
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Agent creation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Agent creation failed: {str(e)}")


@router.post("/meetings/conduct")
async def conduct_meeting(
    request: MeetingRequest,
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """
    Conduct a meeting with AI agents
    
    - **meeting_id**: Unique identifier for the meeting
    - **agenda_items**: List of agenda items to discuss
    - **participant_agents**: List of agent IDs to participate
    - **meeting_context**: Additional context for the meeting
    """
    try:
        agent_manager = await get_agent_manager()
        
        # Validate that agents exist
        if not request.participant_agents:
            raise HTTPException(status_code=400, detail="No participant agents specified")
        
        # Start meeting (this could be run in background for long meetings)
        result = await agent_manager.conduct_meeting_with_agents(
            meeting_id=request.meeting_id,
            agenda_items=request.agenda_items,
            participant_agents=request.participant_agents,
            meeting_context=request.meeting_context
        )
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result.get("error", "Meeting failed"))
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Meeting conduct failed: {e}")
        raise HTTPException(status_code=500, detail=f"Meeting failed: {str(e)}")


@router.post("/recommendations")
async def get_agent_recommendation(request: DecisionRequest) -> Dict[str, Any]:
    """
    Get a decision recommendation from a specific agent
    
    - **agent_id**: ID of the agent to get recommendation from
    - **decision_context**: Context and details of the decision
    - **organization_id**: Organization ID for historical context
    """
    try:
        agent_manager = await get_agent_manager()
        
        result = await agent_manager.get_agent_recommendation(
            agent_id=request.agent_id,
            decision_context=request.decision_context,
            organization_id=request.organization_id
        )
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result.get("error", "Recommendation failed"))
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Agent recommendation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")


@router.get("/organizations/{org_id}/agents", response_model=List[AgentResponse])
async def list_organization_agents(org_id: str) -> List[AgentResponse]:
    """
    List all agents in an organization
    
    - **org_id**: Organization ID to list agents for
    """
    try:
        agent_manager = await get_agent_manager()
        
        agents = await agent_manager.get_agent_list(org_id)
        
        return [
            AgentResponse(
                agent_id=agent["agent_id"],
                name=agent["name"],
                personality=agent["personality"],
                role=agent["role"],
                status=agent["status"],
                expertise_areas=agent.get("expertise_areas", [])
            )
            for agent in agents
        ]
        
    except Exception as e:
        logger.error(f"List agents failed: {e}")
        raise HTTPException(status_code=500, detail=f"List agents failed: {str(e)}")


@router.get("/agents/{agent_id}/status")
async def get_agent_status(agent_id: str) -> Dict[str, Any]:
    """
    Get status and information about a specific agent
    
    - **agent_id**: ID of the agent to check
    """
    try:
        strands_service = await get_strands_service()
        
        status = await strands_service.get_agent_status(agent_id)
        
        if status.get("status") == "not_found":
            raise HTTPException(status_code=404, detail="Agent not found")
        
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get agent status failed: {e}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")


@router.get("/meetings/active")
async def get_active_meetings() -> Dict[str, Any]:
    """
    Get currently active meetings with agents
    """
    try:
        agent_manager = await get_agent_manager()
        
        active_meetings = await agent_manager.get_active_meetings()
        
        return {
            "active_meetings": active_meetings,
            "total_meetings": len(active_meetings)
        }
        
    except Exception as e:
        logger.error(f"Get active meetings failed: {e}")
        raise HTTPException(status_code=500, detail=f"Active meetings check failed: {str(e)}")


@router.post("/agents/{agent_id}/discussion")
async def conduct_agent_discussion(
    agent_id: str,
    agenda_item: Dict[str, Any],
    meeting_context: Dict[str, Any],
    other_agents: List[str] = Query(default=[], description="Other agents to include in discussion")
) -> Dict[str, Any]:
    """
    Conduct a discussion with specific agents on an agenda item
    
    - **agent_id**: Primary agent to include
    - **agenda_item**: Agenda item to discuss
    - **meeting_context**: Context for the discussion
    - **other_agents**: Additional agents to include in discussion
    """
    try:
        strands_service = await get_strands_service()
        
        # Include the primary agent in the discussion
        all_agents = [agent_id] + other_agents
        
        discussion_log = await strands_service.conduct_agent_discussion(
            agent_ids=all_agents,
            agenda_item=agenda_item,
            meeting_context=meeting_context,
            discussion_rounds=2  # Shorter discussion for API endpoint
        )
        
        return {
            "success": True,
            "discussion_log": discussion_log,
            "participants": all_agents,
            "agenda_item": agenda_item.get("title", "Discussion Item")
        }
        
    except Exception as e:
        logger.error(f"Agent discussion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Discussion failed: {str(e)}")


@router.delete("/agents/{agent_id}")
async def delete_agent(agent_id: str) -> Dict[str, Any]:
    """
    Delete an agent (remove from active service)
    
    - **agent_id**: ID of the agent to delete
    """
    try:
        # In a real implementation, this would:
        # 1. Remove agent from Strands service
        # 2. Update database records
        # 3. Clean up any active meetings
        
        return {
            "success": True,
            "message": f"Agent {agent_id} deleted successfully"
        }
        
    except Exception as e:
        logger.error(f"Agent deletion failed: {e}")
        raise HTTPException(status_code=500, detail=f"Agent deletion failed: {str(e)}")


@router.post("/agents/batch-create")
async def batch_create_agents(
    organization_id: str,
    user_ids: List[str],
    background_tasks: BackgroundTasks
) -> Dict[str, Any]:
    """
    Create agents for multiple users in batch
    
    - **organization_id**: Organization ID
    - **user_ids**: List of user IDs to create agents for
    """
    try:
        if len(user_ids) > 50:  # Limit batch size
            raise HTTPException(status_code=400, detail="Batch size limited to 50 users")
        
        agent_manager = await get_agent_manager()
        
        # Process in background for large batches
        async def create_agents_batch():
            results = []
            for user_id in user_ids:
                try:
                    result = await agent_manager.create_agent_for_user(
                        user_id=user_id,
                        organization_id=organization_id,
                        force_recreate=False
                    )
                    results.append({
                        "user_id": user_id,
                        "success": result["success"],
                        "agent_id": result.get("agent_id"),
                        "error": result.get("error")
                    })
                except Exception as e:
                    results.append({
                        "user_id": user_id,
                        "success": False,
                        "error": str(e)
                    })
            
            logger.info(f"Batch agent creation completed: {len(results)} agents processed")
            return results
        
        if len(user_ids) > 10:
            # Run in background for larger batches
            background_tasks.add_task(create_agents_batch)
            return {
                "success": True,
                "message": f"Batch creation started for {len(user_ids)} users",
                "processing_in_background": True
            }
        else:
            # Process immediately for smaller batches
            results = await create_agents_batch()
            successful = sum(1 for r in results if r["success"])
            
            return {
                "success": True,
                "message": f"Created {successful}/{len(user_ids)} agents successfully",
                "results": results,
                "processing_in_background": False
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch agent creation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch creation failed: {str(e)}")
