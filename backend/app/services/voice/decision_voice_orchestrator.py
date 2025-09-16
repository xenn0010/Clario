"""
Decision voice orchestrator bridges VAPI with agent and graph intelligence.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from app.core.logging import get_logger
from app.services.voice.vapi_service import VapiService
from app.services.agents.agent_manager import get_agent_manager, AgentManager
from app.services.graph.neo4j_service import get_neo4j_service
from app.services.vector.weaviate_service import get_weaviate_service

logger = get_logger("decision_voice_orchestrator")


class DecisionVoiceOrchestrator:
    """Coordinates context needed to spin up a VAPI decision assistant."""

    def __init__(self) -> None:
        self._agent_manager: Optional[AgentManager] = None
        self._neo4j_service = None
        self._weaviate_service = None

    async def _ensure_dependencies(self) -> None:
        if not self._agent_manager:
            self._agent_manager = await get_agent_manager()
        if not self._neo4j_service:
            self._neo4j_service = await get_neo4j_service()
        if not self._weaviate_service:
            self._weaviate_service = await get_weaviate_service()

    async def create_voice_session(
        self,
        decision_id: str,
        organization_id: str,
        agenda_hint: Optional[str] = None,
        include_similar: bool = True
    ) -> Dict[str, Any]:
        """Generate a VAPI assistant tailored to a decision."""
        await self._ensure_dependencies()

        decision_flow: Dict[str, Any] = {}
        decision_summary: Dict[str, Any] = {}
        try:
            decision_flow = await self._neo4j_service.analyze_decision_flow(decision_id)
            decision_summary = decision_flow.get("decision", {}) or {}
        except Exception as exc:  # pragma: no cover - network failure handled gracefully
            logger.error("Failed to pull decision flow from Neo4j", exc_info=exc)

        title = decision_summary.get("title") or decision_summary.get("id") or decision_id
        description = decision_summary.get("description") or "No rich description captured yet."
        stakeholders = decision_summary.get("stakeholders", [])
        impact_areas = decision_summary.get("impact_areas", [])
        influenced = decision_flow.get("influenced_decisions", []) or []

        similar_decisions: List[Dict[str, Any]] = []
        if include_similar:
            query_text = agenda_hint or description or title
            try:
                similar_decisions = await self._weaviate_service.search_decisions(
                    query=query_text,
                    organization_id=organization_id,
                    limit=3,
                    min_certainty=0.55
                )
            except Exception as exc:  # pragma: no cover - best effort
                logger.warning("Weaviate similar decision search failed", exc_info=exc)

        agents: List[Dict[str, Any]] = []
        try:
            agents = await self._agent_manager.get_agent_list(organization_id)
        except Exception as exc:  # pragma: no cover - best effort
            logger.warning("Agent roster fetch failed", exc_info=exc)

        system_message = self._compose_system_prompt(
            title=title,
            description=description,
            stakeholders=stakeholders,
            impact_areas=impact_areas,
            influenced=influenced,
            similar_decisions=similar_decisions,
            agents=agents
        )

        vapi_service = VapiService()
        try:
            assistant_id = await vapi_service.create_assistant(
                name=f"Decision-{title[:32]}",
                system_message=system_message
            )
        finally:
            await vapi_service.close()

        return {
            "assistant_id": assistant_id,
            "decision": decision_summary,
            "influenced_decisions": influenced,
            "similar_decisions": similar_decisions,
            "agent_roster": [
                {
                    "agent_id": agent.get("agent_id"),
                    "name": agent.get("name"),
                    "role": agent.get("role"),
                    "personality": agent.get("personality")
                }
                for agent in agents
            ]
        }

    def _compose_system_prompt(
        self,
        title: str,
        description: str,
        stakeholders: List[str],
        impact_areas: List[str],
        influenced: List[Dict[str, Any]],
        similar_decisions: List[Dict[str, Any]],
        agents: List[Dict[str, Any]]
    ) -> str:
        """Build a rich prompt for the voice assistant."""
        influenced_lines = [
            f"- {item.get('title') or item.get('id') or 'Follow-on decision'}"
            for item in influenced
            if item
        ]
        similar_lines = [
            f"- {item.get('title') or item.get('decisionId')} (certainty {item.get('_additional', {}).get('certainty', 0):.2f})"
            for item in similar_decisions
            if item
        ]
        agent_lines = [
            f"- {agent.get('name')} — {agent.get('role')} ({agent.get('personality')})"
            for agent in agents[:5]
        ]

        prompt_sections = [
            "You are Clario's decision-strand voice interface. Hold a natural conversation, cite agent perspectives, and ground every answer in data.",
            f"Decision spotlight: {title}.",
            f"Summary: {description}",
        ]
        if stakeholders:
            prompt_sections.append("Stakeholders: " + ", ".join(stakeholders))
        if impact_areas:
            prompt_sections.append("Impact areas: " + ", ".join(impact_areas))
        if influenced_lines:
            prompt_sections.append("Cascade at stake:\n" + "\n".join(influenced_lines))
        if similar_lines:
            prompt_sections.append("Reference similar calls:\n" + "\n".join(similar_lines))
        if agent_lines:
            prompt_sections.append("Agent council in play:\n" + "\n".join(agent_lines))

        prompt_sections.append(
            "Behaviors: (1) retrieve precise context from graph + strands, (2) surface disagreements, (3) invite follow-up, (4) mark action items verbally."
        )
        return "\n\n".join(prompt_sections)


_orchestrator: Optional[DecisionVoiceOrchestrator] = None


async def get_decision_voice_orchestrator() -> DecisionVoiceOrchestrator:
    """Singleton accessor."""
    global _orchestrator
    if not _orchestrator:
        _orchestrator = DecisionVoiceOrchestrator()
    return _orchestrator

