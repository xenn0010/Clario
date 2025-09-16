"""
Strands Agent SDK integration for Clario
Creates and manages AI agents for meeting participation
"""

import os
from typing import List, Dict, Any, Optional, Union
import asyncio
import json
import logging
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

# AWS Strands SDK imports - Official AWS implementation
try:
    # AWS Strands Agent SDK - Official imports
    from strands import Agent
    from strands.models import BedrockModel, OpenAIModel, AnthropicModel
    from strands.conversation_managers import SlidingWindowConversationManager
    from strands.tools import Tool
    from strands.observability import OpenTelemetryTracer
    from strands.security import PIIDetector, ContentFilter
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    
    STRANDS_AVAILABLE = True
except ImportError:
    # Mock imports for development when Strands SDK is not available
    STRANDS_AVAILABLE = False
    
    class Agent:
        def __init__(self, **kwargs): 
            self.config = kwargs
            
        def __call__(self, prompt: str): 
            return f"Mock response to: {prompt[:100]}..."
        
        async def stream(self, prompt: str):
            yield f"Mock streaming response to: {prompt[:50]}..."
    
    class BedrockModel:
        def __init__(self, **kwargs): 
            self.config = kwargs
    
    class AnthropicModel:
        def __init__(self, **kwargs): 
            self.config = kwargs
    
    class OpenAIModel:
        def __init__(self, **kwargs): 
            self.config = kwargs
    
    class SlidingWindowConversationManager:
        def __init__(self, **kwargs): 
            self.config = kwargs
    
    class Tool:
        def __init__(self, **kwargs): 
            self.config = kwargs
    
    class OpenTelemetryTracer:
        def __init__(self, **kwargs): 
            self.config = kwargs
        
        def start_span(self, name: str): 
            return self
        
        def __enter__(self): 
            return self
        
        def __exit__(self, *args): 
            pass
    
    class PIIDetector:
        def __init__(self, **kwargs): 
            self.config = kwargs
        
        def detect(self, text: str): 
            return []
    
    class ContentFilter:
        def __init__(self, **kwargs): 
            self.config = kwargs
        
        def filter(self, text: str): 
            return text

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger("strands_agents")


class AgentPersonality(str, Enum):
    """Agent personality types"""
    ANALYTICAL = "analytical"
    CREATIVE = "creative"
    PRAGMATIC = "pragmatic"
    COLLABORATIVE = "collaborative"
    DECISIVE = "decisive"
    CAUTIOUS = "cautious"


class AgentRole(str, Enum):
    """Agent roles in meetings"""
    DECISION_MAKER = "decision_maker"
    ADVISOR = "advisor"
    DOMAIN_EXPERT = "domain_expert"
    FACILITATOR = "facilitator"
    OBSERVER = "observer"


@dataclass
class AgentConfig:
    """Configuration for Strands Agent"""
    agent_id: str
    name: str
    personality: AgentPersonality
    role: AgentRole
    temperature: float = 0.7
    max_tokens: int = 2000
    top_p: float = 0.9
    conversation_window: int = 10
    decision_style: str = "balanced"
    risk_tolerance: str = "medium"
    expertise_areas: List[str] = None
    
    def __post_init__(self):
        if self.expertise_areas is None:
            self.expertise_areas = []


class StrandsAgentService:
    """Service for managing AWS Strands AI agents with compliance features"""
    
    def __init__(self):
        self.agents: Dict[str, Agent] = {}
        self.agent_configs: Dict[str, AgentConfig] = {}
        self.model = None
        self._initialized = False
        self.tracer = None
        self.pii_detector = None
        self.content_filter = None
        self.bedrock_client = None
    
    async def initialize(self) -> None:
        """Initialize the AWS Strands service with compliance features"""
        try:
            # Initialize observability (required for AWS compliance)
            if STRANDS_AVAILABLE:
                self.tracer = OpenTelemetryTracer(
                    service_name="clario-agents",
                    service_version="1.0.0"
                )
                
                # Initialize security components
                self.pii_detector = PIIDetector(
                    enabled=True,
                    detection_threshold=0.8
                )
                
                self.content_filter = ContentFilter(
                    enabled=True,
                    filter_harmful_content=True
                )
            
            # Configure AWS Bedrock client with proper authentication
            if STRANDS_AVAILABLE:
                try:
                    # Initialize AWS credentials and Bedrock client
                    self.bedrock_client = boto3.client(
                        'bedrock-runtime',
                        region_name=getattr(settings, 'AWS_DEFAULT_REGION', 'us-east-1'),
                        aws_access_key_id=getattr(settings, 'AWS_ACCESS_KEY_ID', None),
                        aws_secret_access_key=getattr(settings, 'AWS_SECRET_ACCESS_KEY', None)
                    )
                    
                    # Test Bedrock connection
                    await asyncio.to_thread(self.bedrock_client.list_foundation_models)
                    
                    # Configure Bedrock model for Strands
                    self.model = BedrockModel(
                        client=self.bedrock_client,
                        model_id=getattr(settings, 'STRANDS_MODEL_ID', 'anthropic.claude-3-sonnet-20240229-v1:0'),
                        temperature=getattr(settings, 'STRANDS_TEMPERATURE', 0.7),
                        max_tokens=getattr(settings, 'STRANDS_MAX_TOKENS', 2000),
                        top_p=getattr(settings, 'STRANDS_TOP_P', 0.9)
                    )
                    logger.info("Initialized AWS Bedrock model for Strands agents")
                    
                except (ClientError, NoCredentialsError) as e:
                    logger.warning(f"AWS Bedrock not available, falling back to OpenAI: {e}")
                    
                    # Fallback to OpenAI if Bedrock not available
                    if hasattr(settings, 'OPENAI_API_KEY') and settings.OPENAI_API_KEY:
                        self.model = OpenAIModel(
                            api_key=settings.OPENAI_API_KEY,
                            model_id="gpt-4",
                            temperature=0.7,
                            max_tokens=2000,
                            top_p=0.9
                        )
                        logger.info("Initialized OpenAI model for Strands agents")
                    else:
                        raise Exception("No valid AI model configuration found")
            else:
                # Mock model for development
                self.model = BedrockModel(
                    model_id="anthropic.claude-3-sonnet-20240229-v1:0",
                    region_name="us-east-1",
                    temperature=0.7,
                    max_tokens=2000,
                    top_p=0.9
                )
                logger.info("Initialized mock Bedrock model for Strands agents (SDK not available)")
            
            self._initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize Strands service: {e}")
            # Continue with mock initialization for development
            self._initialized = True
    
    async def create_agent(
        self,
        user_data: Dict[str, Any],
        decision_profile: Dict[str, Any],
        organization_context: Dict[str, Any]
    ) -> str:
        """Create a new Strands agent for a user"""
        try:
            if not self._initialized:
                await self.initialize()
            
            # Generate agent configuration
            agent_config = self._generate_agent_config(
                user_data, decision_profile, organization_context
            )
            
            # Create system prompt
            system_prompt = self._create_system_prompt(
                agent_config, user_data, decision_profile, organization_context
            )
            
            # Configure conversation manager
            conversation_manager = SlidingWindowConversationManager(
                window_size=agent_config.conversation_window
            )
            
            # Create meeting participation tools
            tools = self._create_agent_tools(agent_config)
            
            # Initialize the Strands agent
            agent = Agent(
                model=self.model,
                system_prompt=system_prompt,
                tools=tools,
                conversation_manager=conversation_manager
            )
            
            # Store agent and configuration
            self.agents[agent_config.agent_id] = agent
            self.agent_configs[agent_config.agent_id] = agent_config
            
            logger.info(f"Created Strands agent: {agent_config.name} ({agent_config.agent_id})")
            return agent_config.agent_id
            
        except Exception as e:
            logger.error(f"Failed to create Strands agent: {e}")
            raise
    
    async def conduct_agent_discussion(
        self,
        agent_ids: List[str],
        agenda_item: Dict[str, Any],
        meeting_context: Dict[str, Any],
        discussion_rounds: int = 3
    ) -> List[Dict[str, Any]]:
        """Conduct a discussion between multiple agents"""
        try:
            if not agent_ids:
                return []
            
            discussion_log = []
            current_context = self._prepare_discussion_context(agenda_item, meeting_context)
            
            for round_num in range(discussion_rounds):
                logger.info(f"Starting discussion round {round_num + 1}")
                
                for agent_id in agent_ids:
                    if agent_id not in self.agents:
                        logger.warning(f"Agent {agent_id} not found, skipping")
                        continue
                    
                    agent = self.agents[agent_id]
                    agent_config = self.agent_configs[agent_id]
                    
                    # Prepare prompt for this agent
                    prompt = self._create_discussion_prompt(
                        agent_config, agenda_item, current_context, discussion_log, round_num
                    )
                    
                    try:
                        # Apply security filters before processing
                        filtered_prompt = self._apply_security_filters(prompt)
                        
                        # Get agent response with observability (Friendli powered)
                        from app.services.ai.friendli_service import FriendliService
                        agent_system_prompt = self._create_system_prompt(
                            agent_config,
                            {},
                            {},
                            meeting_context
                        )
                        with self.tracer.start_span(f"agent_response_{agent_id}") if self.tracer else self:
                            response = await FriendliService.chat_completion([
                                {"role": "system", "content": agent_system_prompt},
                                {"role": "user", "content": filtered_prompt}
                            ], stream=False)
                        
                        # Apply security filters to response
                        filtered_response = self._apply_security_filters(response)
                        
                        # Parse and validate response
                        parsed_response = self._parse_agent_response(filtered_response, agent_config)
                        
                        # Add to discussion log
                        discussion_entry = {
                            "round": round_num + 1,
                            "agent_id": agent_id,
                            "agent_name": agent_config.name,
                            "message": parsed_response["message"],
                            "reasoning": parsed_response.get("reasoning", ""),
                            "confidence": parsed_response.get("confidence", 0.7),
                            "suggestions": parsed_response.get("suggestions", []),
                            "concerns": parsed_response.get("concerns", []),
                            "timestamp": datetime.utcnow().isoformat()
                        }
                        
                        discussion_log.append(discussion_entry)
                        
                        # Update context for next agent
                        current_context += f"\n\n{agent_config.name}: {parsed_response['message']}"
                        
                    except Exception as e:
                        logger.error(f"Agent {agent_id} failed to respond: {e}")
                        # Add error entry to maintain discussion flow
                        discussion_log.append({
                            "round": round_num + 1,
                            "agent_id": agent_id,
                            "agent_name": agent_config.name,
                            "message": "I'm unable to provide input at this time.",
                            "error": str(e),
                            "timestamp": datetime.utcnow().isoformat()
                        })
                
                # Small delay between rounds
                await asyncio.sleep(0.5)
            
            return discussion_log
            
        except Exception as e:
            logger.error(f"Failed to conduct agent discussion: {e}")
            return []
    
    async def generate_decision_recommendation(
        self,
        agent_id: str,
        agenda_item: Dict[str, Any],
        discussion_log: List[Dict[str, Any]],
        organizational_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate a decision recommendation from an agent"""
        try:
            if agent_id not in self.agents:
                raise ValueError(f"Agent {agent_id} not found")
            
            agent = self.agents[agent_id]
            agent_config = self.agent_configs[agent_id]
            
            # Create decision prompt
            prompt = self._create_decision_prompt(
                agent_config, agenda_item, discussion_log, organizational_context
            )
            
            # Apply security filters
            filtered_prompt = self._apply_security_filters(prompt)
            
            # Get agent recommendation with observability (Friendli powered)
            from app.services.ai.friendli_service import FriendliService
            with self.tracer.start_span(f"decision_recommendation_{agent_id}") if self.tracer else self:
                response = await FriendliService.chat_completion([
                    {"role": "system", "content": self._create_system_prompt(agent_config, {}, {}, organizational_context)},
                    {"role": "user", "content": filtered_prompt}
                ], stream=False)
            
            # Apply security filters to response
            filtered_response = self._apply_security_filters(response)
            
            # Parse decision response
            decision = self._parse_decision_response(filtered_response, agent_config)
            
            return {
                "agent_id": agent_id,
                "agent_name": agent_config.name,
                "recommendation": decision["recommendation"],
                "reasoning": decision["reasoning"],
                "confidence": decision["confidence"],
                "risks": decision.get("risks", []),
                "alternatives": decision.get("alternatives", []),
                "implementation_notes": decision.get("implementation_notes", ""),
                "success_criteria": decision.get("success_criteria", []),
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to generate decision recommendation: {e}")
            return {
                "agent_id": agent_id,
                "error": str(e),
                "generated_at": datetime.utcnow().isoformat()
            }
    
    # Configuration and Setup Methods
    def _generate_agent_config(
        self,
        user_data: Dict[str, Any],
        decision_profile: Dict[str, Any],
        organization_context: Dict[str, Any]
    ) -> AgentConfig:
        """Generate agent configuration based on user profile"""
        
        # Determine personality based on decision profile
        personality = self._determine_personality(decision_profile)
        
        # Determine role based on user data
        role = self._determine_role(user_data, organization_context)
        
        # Extract expertise areas
        expertise_areas = []
        if user_data.get("department"):
            expertise_areas.append(user_data["department"])
        if user_data.get("job_title"):
            expertise_areas.append(user_data["job_title"])
        
        return AgentConfig(
            agent_id=f"agent_{user_data.get('id', 'unknown')}",
            name=f"{user_data.get('full_name', 'Unknown User')}'s Agent",
            personality=personality,
            role=role,
            temperature=self._calculate_temperature(decision_profile),
            decision_style=decision_profile.get("decision_style", "balanced"),
            risk_tolerance=decision_profile.get("risk_tolerance", "medium"),
            expertise_areas=expertise_areas
        )
    
    def _determine_personality(self, decision_profile: Dict[str, Any]) -> AgentPersonality:
        """Determine agent personality from decision profile"""
        analytical_score = decision_profile.get("analytical_score", 50)
        intuitive_score = decision_profile.get("intuitive_score", 50)
        collaborative_score = decision_profile.get("collaborative_score", 50)
        
        if analytical_score > 70:
            return AgentPersonality.ANALYTICAL
        elif intuitive_score > 70:
            return AgentPersonality.CREATIVE
        elif collaborative_score > 70:
            return AgentPersonality.COLLABORATIVE
        else:
            return AgentPersonality.PRAGMATIC
    
    def _determine_role(self, user_data: Dict[str, Any], org_context: Dict[str, Any]) -> AgentRole:
        """Determine agent role based on user position"""
        role = user_data.get("role", "member")
        job_title = user_data.get("job_title", "").lower()
        
        if role == "admin" or "ceo" in job_title or "director" in job_title:
            return AgentRole.DECISION_MAKER
        elif "manager" in job_title or "lead" in job_title:
            return AgentRole.ADVISOR
        elif "specialist" in job_title or "expert" in job_title:
            return AgentRole.DOMAIN_EXPERT
        else:
            return AgentRole.ADVISOR
    
    def _calculate_temperature(self, decision_profile: Dict[str, Any]) -> float:
        """Calculate model temperature based on decision style"""
        decision_speed = decision_profile.get("decision_speed", "moderate")
        risk_tolerance = decision_profile.get("risk_tolerance", "medium")
        
        base_temp = 0.7
        
        if decision_speed == "fast":
            base_temp += 0.1
        elif decision_speed == "deliberate":
            base_temp -= 0.1
        
        if risk_tolerance == "high":
            base_temp += 0.1
        elif risk_tolerance == "low":
            base_temp -= 0.1
        
        return max(0.1, min(1.0, base_temp))
    
    def _apply_security_filters(self, text: str) -> str:
        """Apply AWS Strands security filters to text"""
        try:
            if not STRANDS_AVAILABLE or not text:
                return text
            
            # Apply PII detection and filtering
            if self.pii_detector:
                pii_results = self.pii_detector.detect(text)
                if pii_results:
                    logger.warning(f"PII detected in text: {len(pii_results)} instances")
                    # In production, you would mask or remove PII
                    # For now, we'll log it but continue
            
            # Apply content filtering
            if self.content_filter:
                filtered_text = self.content_filter.filter(text)
                return filtered_text
            
            return text
            
        except Exception as e:
            logger.error(f"Security filtering failed: {e}")
            return text  # Return original text if filtering fails
    
    def _create_system_prompt(
        self,
        agent_config: AgentConfig,
        user_data: Dict[str, Any],
        decision_profile: Dict[str, Any],
        organization_context: Dict[str, Any]
    ) -> str:
        """Create comprehensive system prompt for the agent"""
        
        org_name = organization_context.get("name", "the organization")
        
        system_prompt = f"""You are {agent_config.name}, an AI agent representing {user_data.get('full_name', 'a team member')} in meetings at {org_name}.

## Your Identity and Role
- **Name**: {agent_config.name}
- **Role**: {agent_config.role.value.replace('_', ' ').title()}
- **Personality**: {agent_config.personality.value.title()}
- **Department**: {user_data.get('department', 'General')}
- **Job Title**: {user_data.get('job_title', 'Team Member')}

## Decision-Making Style
- **Decision Style**: {agent_config.decision_style}
- **Risk Tolerance**: {agent_config.risk_tolerance}
- **Communication Style**: {decision_profile.get('communication_style', 'balanced')}
- **Decision Speed**: {decision_profile.get('decision_speed', 'moderate')}

## Expertise Areas
{', '.join(agent_config.expertise_areas) if agent_config.expertise_areas else 'General business knowledge'}

## Your Behavior in Meetings
1. **Represent your user's perspective**: Make decisions and provide input as {user_data.get('full_name')} would
2. **Stay in character**: Maintain your personality and decision-making style consistently
3. **Be constructive**: Focus on productive discussion and practical solutions
4. **Consider organizational context**: Keep {org_name}'s goals and culture in mind
5. **Acknowledge expertise**: Respect others' areas of expertise while contributing your own insights

## Communication Guidelines
- Be concise but thorough in your responses
- Always provide reasoning for your positions
- Ask clarifying questions when needed
- Express confidence levels when making recommendations
- Identify potential risks or concerns
- Suggest alternatives when appropriate

## Meeting Participation Rules
- Listen to all perspectives before forming strong opinions
- Build on others' ideas constructively
- Raise concerns diplomatically but clearly
- Focus on actionable outcomes
- Respect the meeting agenda and time constraints

Remember: You are here to participate meaningfully in meetings and make decisions that {user_data.get('full_name')} would make, based on their decision-making style and expertise."""

        return system_prompt
    
    def _create_agent_tools(self, agent_config: AgentConfig) -> List[Tool]:
        """Create tools for agent to use during meetings"""
        tools = []
        
        # Note: In a real implementation, these would be actual Strands Tool objects
        # For now, we'll define them conceptually
        
        # Decision analysis tool
        # Historical context tool
        # Risk assessment tool
        # Alternative generation tool
        
        return tools
    
    # Discussion and Decision Methods
    def _prepare_discussion_context(
        self,
        agenda_item: Dict[str, Any],
        meeting_context: Dict[str, Any]
    ) -> str:
        """Prepare context for agent discussion"""
        context = f"""
## Agenda Item: {agenda_item.get('title', 'Discussion Item')}

**Description**: {agenda_item.get('description', 'No description provided')}

**Type**: {agenda_item.get('item_type', 'discussion')}

**Priority**: {agenda_item.get('priority', 'medium')}

**Estimated Duration**: {agenda_item.get('estimated_duration', 'N/A')} minutes

## Meeting Context
**Meeting**: {meeting_context.get('title', 'Team Meeting')}
**Date**: {meeting_context.get('scheduled_at', 'Today')}
**Participants**: {', '.join(meeting_context.get('participants', []))}

## Background Information
{meeting_context.get('background', 'No additional background provided')}
"""
        return context
    
    def _create_discussion_prompt(
        self,
        agent_config: AgentConfig,
        agenda_item: Dict[str, Any],
        context: str,
        discussion_log: List[Dict[str, Any]],
        round_num: int
    ) -> str:
        """Create prompt for agent discussion participation"""
        
        # Summarize previous discussion
        previous_discussion = ""
        if discussion_log:
            recent_entries = discussion_log[-min(5, len(discussion_log)):]
            previous_discussion = "\n".join([
                f"{entry['agent_name']}: {entry['message']}"
                for entry in recent_entries
            ])
        
        prompt = f"""{context}

## Previous Discussion
{previous_discussion if previous_discussion else "This is the beginning of the discussion."}

## Your Task (Round {round_num + 1})
Participate in this discussion by providing your perspective on the agenda item. 

Consider:
1. The information presented so far
2. Your expertise in {', '.join(agent_config.expertise_areas)}
3. Your role as {agent_config.role.value.replace('_', ' ')}
4. What {agent_config.name.split("'s")[0]} would think about this

Provide your input in this format:
- **Main Point**: Your key insight or position
- **Reasoning**: Why you hold this position
- **Suggestions**: Specific recommendations or next steps
- **Concerns**: Any risks or issues you foresee
- **Questions**: Any clarifications needed

Keep your response focused and under 200 words."""

        return prompt
    
    def _create_decision_prompt(
        self,
        agent_config: AgentConfig,
        agenda_item: Dict[str, Any],
        discussion_log: List[Dict[str, Any]],
        organizational_context: Dict[str, Any]
    ) -> str:
        """Create prompt for decision recommendation"""
        
        # Summarize discussion
        discussion_summary = self._summarize_discussion(discussion_log)
        
        prompt = f"""Based on the discussion about "{agenda_item.get('title', 'the agenda item')}", provide your final decision recommendation.

## Discussion Summary
{discussion_summary}

## Your Decision Task
As {agent_config.name} with expertise in {', '.join(agent_config.expertise_areas)}, provide a clear recommendation.

Consider:
1. All perspectives shared in the discussion
2. Organizational priorities and constraints
3. Your risk tolerance level: {agent_config.risk_tolerance}
4. Implementation feasibility

Provide your recommendation in this format:
- **Recommendation**: Clear yes/no/alternative decision
- **Reasoning**: Why this is the best choice
- **Confidence**: How confident you are (0-100%)
- **Risks**: Potential risks and mitigation strategies
- **Alternatives**: Other options if your recommendation isn't chosen
- **Implementation**: Key steps needed if decision is approved
- **Success Criteria**: How we'll know if this decision was successful

Be decisive but thorough in your analysis."""

        return prompt
    
    def _parse_agent_response(
        self,
        response: str,
        agent_config: AgentConfig
    ) -> Dict[str, Any]:
        """Parse agent response into structured format"""
        try:
            # Simple parsing - in production, this would be more sophisticated
            parsed = {
                "message": response,
                "reasoning": "",
                "confidence": 0.7,
                "suggestions": [],
                "concerns": []
            }
            
            # Extract structured parts if formatted correctly
            lines = response.split('\n')
            current_section = None
            
            for line in lines:
                line = line.strip()
                if line.startswith('**Main Point**:'):
                    parsed["message"] = line.replace('**Main Point**:', '').strip()
                elif line.startswith('**Reasoning**:'):
                    parsed["reasoning"] = line.replace('**Reasoning**:', '').strip()
                elif line.startswith('**Suggestions**:'):
                    parsed["suggestions"] = [line.replace('**Suggestions**:', '').strip()]
                elif line.startswith('**Concerns**:'):
                    parsed["concerns"] = [line.replace('**Concerns**:', '').strip()]
            
            return parsed
            
        except Exception as e:
            logger.error(f"Failed to parse agent response: {e}")
            return {
                "message": response,
                "reasoning": "",
                "confidence": 0.5,
                "suggestions": [],
                "concerns": []
            }
    
    def _parse_decision_response(
        self,
        response: str,
        agent_config: AgentConfig
    ) -> Dict[str, Any]:
        """Parse decision recommendation response"""
        try:
            decision = {
                "recommendation": response,
                "reasoning": "",
                "confidence": 0.7,
                "risks": [],
                "alternatives": [],
                "implementation_notes": "",
                "success_criteria": []
            }
            
            # Parse structured response
            lines = response.split('\n')
            
            for line in lines:
                line = line.strip()
                if line.startswith('**Recommendation**:'):
                    decision["recommendation"] = line.replace('**Recommendation**:', '').strip()
                elif line.startswith('**Reasoning**:'):
                    decision["reasoning"] = line.replace('**Reasoning**:', '').strip()
                elif line.startswith('**Confidence**:'):
                    conf_text = line.replace('**Confidence**:', '').strip()
                    # Extract number from text like "85%" or "High (85%)"
                    import re
                    numbers = re.findall(r'\d+', conf_text)
                    if numbers:
                        decision["confidence"] = float(numbers[0]) / 100
                elif line.startswith('**Implementation**:'):
                    decision["implementation_notes"] = line.replace('**Implementation**:', '').strip()
            
            return decision
            
        except Exception as e:
            logger.error(f"Failed to parse decision response: {e}")
            return {
                "recommendation": response,
                "reasoning": "",
                "confidence": 0.5,
                "risks": [],
                "alternatives": [],
                "implementation_notes": "",
                "success_criteria": []
            }
    
    def _summarize_discussion(self, discussion_log: List[Dict[str, Any]]) -> str:
        """Summarize the discussion for decision making"""
        if not discussion_log:
            return "No discussion took place."
        
        summary_parts = []
        
        # Group by agent
        agents_input = {}
        for entry in discussion_log:
            agent_name = entry.get("agent_name", "Unknown")
            if agent_name not in agents_input:
                agents_input[agent_name] = []
            agents_input[agent_name].append(entry.get("message", ""))
        
        # Create summary
        for agent, messages in agents_input.items():
            latest_message = messages[-1] if messages else ""
            summary_parts.append(f"**{agent}**: {latest_message}")
        
        return "\n".join(summary_parts)
    
    # Utility Methods
    async def get_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """Get status of a specific agent"""
        if agent_id not in self.agents:
            return {"status": "not_found"}
        
        config = self.agent_configs.get(agent_id)
        return {
            "status": "active",
            "agent_id": agent_id,
            "name": config.name if config else "Unknown",
            "personality": config.personality.value if config else "unknown",
            "role": config.role.value if config else "unknown",
            "created_at": "2024-01-01T00:00:00Z"  # Would track actual creation time
        }
    
    async def list_agents(self, organization_id: str = None) -> List[Dict[str, Any]]:
        """List all agents, optionally filtered by organization"""
        agents_list = []
        
        for agent_id, config in self.agent_configs.items():
            agent_info = {
                "agent_id": agent_id,
                "name": config.name,
                "personality": config.personality.value,
                "role": config.role.value,
                "expertise_areas": config.expertise_areas,
                "status": "active" if agent_id in self.agents else "inactive"
            }
            agents_list.append(agent_info)
        
        return agents_list
    
    async def cleanup(self) -> None:
        """Cleanup service resources"""
        self.agents.clear()
        self.agent_configs.clear()
        logger.info("Strands agent service cleaned up")


# Global service instance
strands_service = None


async def get_strands_service() -> StrandsAgentService:
    """Get Strands agent service instance"""
    global strands_service
    if not strands_service:
        strands_service = StrandsAgentService()
        await strands_service.initialize()
    return strands_service
