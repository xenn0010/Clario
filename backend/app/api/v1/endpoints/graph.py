"""
Graph database endpoints for Clario
Decision flow analysis and organizational insights
"""

from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

from app.services.graph.graph_manager import get_graph_manager
from app.services.graph.decision_flow_analyzer import get_decision_flow_analyzer
from app.services.graph.neo4j_service import get_neo4j_service
from app.core.logging import get_logger

logger = get_logger("graph_api")

router = APIRouter()


class MeetingGraphRequest(BaseModel):
    """Request to index meeting in graph"""
    meeting_data: Dict[str, Any]
    participants: Optional[List[str]] = None
    topics_discussed: Optional[List[str]] = None


class DecisionGraphRequest(BaseModel):
    """Request to index decision in graph"""
    decision_data: Dict[str, Any]
    outcomes: Optional[List[Dict[str, Any]]] = None
    dependencies: Optional[List[str]] = None


class OutcomeRequest(BaseModel):
    """Request to add decision outcome"""
    outcome_data: Dict[str, Any]


@router.post("/meetings/{meeting_id}/index")
async def index_meeting_graph(
    meeting_id: str,
    request: MeetingGraphRequest
) -> Dict[str, Any]:
    """Index meeting in graph database with full context"""
    try:
        graph_manager = await get_graph_manager()
        
        # Ensure meeting ID matches
        request.meeting_data["id"] = meeting_id
        
        success = await graph_manager.index_meeting_with_full_context(
            meeting_data=request.meeting_data,
            participants=request.participants,
            topics_discussed=request.topics_discussed
        )
        
        return {
            "success": success,
            "meeting_id": meeting_id,
            "message": "Meeting indexed in graph successfully" if success else "Failed to index meeting in graph"
        }
        
    except Exception as e:
        logger.error(f"Meeting graph indexing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Graph indexing failed: {str(e)}")


@router.post("/decisions/{decision_id}/index")
async def index_decision_graph(
    decision_id: str,
    request: DecisionGraphRequest
) -> Dict[str, Any]:
    """Index decision in graph database with consequences"""
    try:
        graph_manager = await get_graph_manager()
        
        # Ensure decision ID matches
        request.decision_data["id"] = decision_id
        
        success = await graph_manager.index_decision_with_consequences(
            decision_data=request.decision_data,
            outcomes=request.outcomes,
            dependencies=request.dependencies
        )
        
        return {
            "success": success,
            "decision_id": decision_id,
            "message": "Decision indexed in graph successfully" if success else "Failed to index decision in graph"
        }
        
    except Exception as e:
        logger.error(f"Decision graph indexing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Graph indexing failed: {str(e)}")


@router.post("/decisions/{decision_id}/outcomes")
async def add_decision_outcome(
    decision_id: str,
    request: OutcomeRequest
) -> Dict[str, Any]:
    """Add outcome/consequence to a decision"""
    try:
        neo4j_service = await get_neo4j_service()
        
        # Ensure decision ID is set
        request.outcome_data["decision_id"] = decision_id
        
        # Create outcome node
        success = await neo4j_service.create_outcome_node(request.outcome_data)
        
        if success:
            # Create relationship
            outcome_id = request.outcome_data.get("id", f"outcome_{decision_id}_{hash(str(request.outcome_data))}")
            await neo4j_service.create_decision_outcome_relationship(decision_id, outcome_id)
        
        return {
            "success": success,
            "decision_id": decision_id,
            "outcome_id": outcome_id if success else None,
            "message": "Outcome added successfully" if success else "Failed to add outcome"
        }
        
    except Exception as e:
        logger.error(f"Add decision outcome failed: {e}")
        raise HTTPException(status_code=500, detail=f"Add outcome failed: {str(e)}")


@router.get("/decisions/{decision_id}/analysis")
async def get_decision_analysis(decision_id: str) -> Dict[str, Any]:
    """Get comprehensive analysis of a decision and its impact"""
    try:
        graph_manager = await get_graph_manager()
        
        analysis = await graph_manager.get_comprehensive_decision_analysis(decision_id)
        
        if "error" in analysis:
            raise HTTPException(status_code=404, detail=analysis["error"])
        
        return analysis
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Decision analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.get("/decisions/{decision_id}/consequences")
async def analyze_decision_consequences(decision_id: str) -> Dict[str, Any]:
    """Analyze the complete consequences of a decision"""
    try:
        flow_analyzer = await get_decision_flow_analyzer()
        
        consequences = await flow_analyzer.analyze_decision_consequences(decision_id)
        
        if "error" in consequences:
            raise HTTPException(status_code=404, detail=consequences["error"])
        
        return consequences
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Consequence analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@router.get("/decisions/{decision_id}/chain")
async def trace_decision_chain(decision_id: str) -> Dict[str, Any]:
    """Trace the complete chain of decisions that followed from this decision"""
    try:
        flow_analyzer = await get_decision_flow_analyzer()
        
        chain = await flow_analyzer.trace_decision_chain(decision_id)
        
        if "error" in chain:
            raise HTTPException(status_code=404, detail=chain["error"])
        
        return chain
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Decision chain tracing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Chain tracing failed: {str(e)}")


@router.post("/decisions/predict")
async def predict_decision_outcome(
    decision_context: Dict[str, Any],
    organization_id: str = Query(..., description="Organization ID")
) -> Dict[str, Any]:
    """Predict likely outcomes of a proposed decision based on historical patterns"""
    try:
        flow_analyzer = await get_decision_flow_analyzer()
        
        prediction = await flow_analyzer.predict_decision_outcome(
            decision_context=decision_context,
            organization_id=organization_id
        )
        
        return prediction
        
    except Exception as e:
        logger.error(f"Decision prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.get("/organizations/{org_id}/insights")
async def get_organization_insights(org_id: str) -> Dict[str, Any]:
    """Get comprehensive organizational decision-making insights"""
    try:
        graph_manager = await get_graph_manager()
        
        insights = await graph_manager.generate_organization_insights(org_id)
        
        if "error" in insights:
            raise HTTPException(status_code=404, detail=insights["error"])
        
        return insights
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Organization insights failed: {e}")
        raise HTTPException(status_code=500, detail=f"Insights generation failed: {str(e)}")


@router.get("/organizations/{org_id}/patterns")
async def get_decision_patterns(
    org_id: str,
    time_period_days: int = Query(default=180, ge=30, le=365, description="Analysis period in days")
) -> Dict[str, Any]:
    """Get organizational decision-making patterns"""
    try:
        flow_analyzer = await get_decision_flow_analyzer()
        
        patterns = await flow_analyzer.map_organizational_decision_patterns(
            organization_id=org_id,
            time_period_days=time_period_days
        )
        
        if "error" in patterns:
            raise HTTPException(status_code=404, detail=patterns["error"])
        
        return patterns
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Decision patterns analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Pattern analysis failed: {str(e)}")


@router.get("/organizations/{org_id}/bottlenecks")
async def get_decision_bottlenecks(org_id: str) -> List[Dict[str, Any]]:
    """Get decision bottlenecks in the organization"""
    try:
        flow_analyzer = await get_decision_flow_analyzer()
        
        bottlenecks = await flow_analyzer.find_decision_bottlenecks(org_id)
        
        return bottlenecks
        
    except Exception as e:
        logger.error(f"Bottleneck analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Bottleneck analysis failed: {str(e)}")


@router.get("/topics/{topic_name}/history")
async def get_topic_decision_history(
    topic_name: str,
    organization_id: str = Query(..., description="Organization ID")
) -> List[Dict[str, Any]]:
    """Get decision history for a specific topic"""
    try:
        neo4j_service = await get_neo4j_service()
        
        history = await neo4j_service.get_topic_decision_history(
            topic_name=topic_name,
            organization_id=organization_id
        )
        
        return history
        
    except Exception as e:
        logger.error(f"Topic history retrieval failed: {e}")
        raise HTTPException(status_code=500, detail=f"History retrieval failed: {str(e)}")


@router.get("/topics/trends")
async def get_topic_trends(
    organization_id: str = Query(..., description="Organization ID"),
    days: int = Query(default=90, ge=7, le=365, description="Analysis period in days")
) -> List[Dict[str, Any]]:
    """Get trending topics in meetings and decisions"""
    try:
        neo4j_service = await get_neo4j_service()
        
        trends = await neo4j_service.get_topic_trends(
            organization_id=organization_id,
            days=days
        )
        
        return trends
        
    except Exception as e:
        logger.error(f"Topic trends analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Trends analysis failed: {str(e)}")


@router.get("/stats")
async def get_graph_statistics() -> Dict[str, Any]:
    """Get overall graph database statistics"""
    try:
        neo4j_service = await get_neo4j_service()
        
        stats = await neo4j_service.get_database_stats()
        
        return {
            "database_stats": stats,
            "timestamp": "2024-01-01T00:00:00Z"  # Would use actual timestamp
        }
        
    except Exception as e:
        logger.error(f"Graph statistics failed: {e}")
        raise HTTPException(status_code=500, detail=f"Statistics retrieval failed: {str(e)}")
