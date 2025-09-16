"""
Main API router for Clario v1
"""

from fastapi import APIRouter

from app.api.v1.endpoints import search, meetings, decisions, organizations, graph, agents, vapi

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(search.router, prefix="/search", tags=["search"])
api_router.include_router(meetings.router, prefix="/meetings", tags=["meetings"])
api_router.include_router(decisions.router, prefix="/decisions", tags=["decisions"])
api_router.include_router(organizations.router, prefix="/organizations", tags=["organizations"])
api_router.include_router(graph.router, prefix="/graph", tags=["graph"])
api_router.include_router(agents.router, prefix="/agents", tags=["agents"])
api_router.include_router(vapi.router, prefix="/voice", tags=["voice"])
