"""
AI Agent models
"""

from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, ForeignKey, JSON, Float
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from enum import Enum
import uuid
from datetime import datetime

from app.core.database import Base


class AgentStatus(str, Enum):
    """Agent status options"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    TRAINING = "training"
    ERROR = "error"


class AgentType(str, Enum):
    """Agent type options"""
    PERSONAL = "personal"  # Represents individual user
    ROLE_BASED = "role_based"  # Represents a role/function
    SPECIALIST = "specialist"  # Domain expert agent


class Agent(Base):
    """AI Agent model representing users in meetings"""
    __tablename__ = "agents"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    organization_id = Column(String, ForeignKey("organizations.id"), nullable=False)
    user_id = Column(String, ForeignKey("users.id"), nullable=False)  # User this agent represents
    
    # Agent identity
    name = Column(String, nullable=False)
    description = Column(Text)
    agent_type = Column(String, default=AgentType.PERSONAL)
    avatar_url = Column(String)
    
    # Agent configuration
    system_prompt = Column(Text, nullable=False)  # Core personality and decision-making instructions
    decision_framework = Column(JSON)  # Structured decision-making framework
    communication_style = Column(JSON)  # How the agent communicates
    
    # Decision-making parameters
    risk_tolerance = Column(String)  # low, medium, high
    decision_speed = Column(String)  # fast, moderate, deliberate
    collaboration_style = Column(String)  # independent, collaborative, consensus-seeking
    authority_level = Column(String)  # observer, participant, decision-maker
    
    # Learning and adaptation
    learning_enabled = Column(Boolean, default=True)
    adaptation_rate = Column(Float, default=0.1)  # How quickly agent adapts (0-1)
    training_data = Column(JSON)  # Historical decisions and outcomes
    
    # Performance metrics
    decision_accuracy = Column(Float, default=0.0)  # 0-1
    user_satisfaction = Column(Float, default=0.0)  # 0-1
    meetings_participated = Column(Integer, default=0)
    decisions_made = Column(Integer, default=0)
    
    # Status and configuration
    status = Column(String, default=AgentStatus.TRAINING)
    is_active = Column(Boolean, default=True)
    
    # AI model configuration
    model_provider = Column(String, default="friendliai")  # friendliai, openai, etc.
    model_name = Column(String, default="meta-llama-3.1-70b-instruct")
    temperature = Column(Float, default=0.7)
    max_tokens = Column(Integer, default=2000)
    
    # Strands SDK configuration
    strands_agent_id = Column(String)  # ID from Strands SDK
    strands_config = Column(JSON)  # Strands-specific configuration
    
    # Version and updates
    version = Column(String, default="1.0.0")
    last_training_at = Column(DateTime)
    last_updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    created_at = Column(DateTime, server_default=func.now())
    
    # Relationships
    organization = relationship("Organization", back_populates="agents")
    user = relationship("User", back_populates="agents")
    meeting_participants = relationship("MeetingParticipant", back_populates="agent")
    decisions = relationship("Decision", back_populates="agent")
    agent_interactions = relationship("AgentInteraction", back_populates="agent")
    performance_metrics = relationship("AgentPerformance", back_populates="agent")
    
    def __repr__(self):
        return f"<Agent(id={self.id}, name={self.name}, user={self.user_id})>"
    
    @property
    def is_ready(self) -> bool:
        """Check if agent is ready for meetings"""
        return (
            self.status == AgentStatus.ACTIVE and 
            self.system_prompt and 
            self.decision_framework
        )
    
    @property
    def effectiveness_score(self) -> float:
        """Calculate overall effectiveness score"""
        if self.decisions_made == 0:
            return 0.0
        
        accuracy_weight = 0.4
        satisfaction_weight = 0.4
        participation_weight = 0.2
        
        participation_score = min(self.meetings_participated / 10, 1.0)  # Normalize to 0-1
        
        return (
            self.decision_accuracy * accuracy_weight +
            self.user_satisfaction * satisfaction_weight +
            participation_score * participation_weight
        )


class AgentInteraction(Base):
    """Log of agent interactions and decisions"""
    __tablename__ = "agent_interactions"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    agent_id = Column(String, ForeignKey("agents.id"), nullable=False)
    meeting_id = Column(String, ForeignKey("meetings.id"))
    
    # Interaction details
    interaction_type = Column(String, nullable=False)  # decision, discussion, question, clarification
    input_data = Column(JSON)  # Input that triggered the interaction
    output_data = Column(JSON)  # Agent's response/decision
    
    # Context
    context = Column(JSON)  # Surrounding context and information
    reasoning = Column(Text)  # Agent's reasoning process
    confidence_score = Column(Float)  # 0-1
    
    # Performance
    processing_time_ms = Column(Integer)
    tokens_used = Column(Integer)
    cost_estimate = Column(Float)  # Estimated cost in USD
    
    # Feedback and learning
    feedback_score = Column(Float)  # Human feedback on this interaction
    feedback_comments = Column(Text)
    was_overridden = Column(Boolean, default=False)
    override_reason = Column(Text)
    
    timestamp = Column(DateTime, server_default=func.now())
    
    # Relationships
    agent = relationship("Agent", back_populates="agent_interactions")
    meeting = relationship("Meeting")
    
    def __repr__(self):
        return f"<AgentInteraction(id={self.id}, agent={self.agent_id}, type={self.interaction_type})>"


class AgentPerformance(Base):
    """Agent performance tracking over time"""
    __tablename__ = "agent_performance"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    agent_id = Column(String, ForeignKey("agents.id"), nullable=False)
    
    # Performance period
    period_start = Column(DateTime, nullable=False)
    period_end = Column(DateTime, nullable=False)
    
    # Metrics
    meetings_count = Column(Integer, default=0)
    decisions_count = Column(Integer, default=0)
    correct_decisions = Column(Integer, default=0)
    user_feedback_avg = Column(Float, default=0.0)
    response_time_avg = Column(Float, default=0.0)  # Average response time in seconds
    
    # Detailed metrics (JSON)
    accuracy_by_decision_type = Column(JSON)
    interaction_patterns = Column(JSON)
    improvement_areas = Column(JSON)
    strengths = Column(JSON)
    
    created_at = Column(DateTime, server_default=func.now())
    
    # Relationships
    agent = relationship("Agent", back_populates="performance_metrics")
    
    def __repr__(self):
        return f"<AgentPerformance(agent={self.agent_id}, period={self.period_start}-{self.period_end})>"


class AgentTemplate(Base):
    """Templates for creating agents with specific characteristics"""
    __tablename__ = "agent_templates"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    organization_id = Column(String, ForeignKey("organizations.id"))
    
    # Template details
    name = Column(String, nullable=False)
    description = Column(Text)
    category = Column(String)  # manager, analyst, creative, technical, etc.
    
    # Template configuration
    base_system_prompt = Column(Text, nullable=False)
    decision_framework_template = Column(JSON)
    communication_style_template = Column(JSON)
    
    # Default parameters
    default_risk_tolerance = Column(String, default="medium")
    default_decision_speed = Column(String, default="moderate")
    default_collaboration_style = Column(String, default="collaborative")
    
    # Usage tracking
    usage_count = Column(Integer, default=0)
    rating = Column(Float, default=0.0)
    
    is_public = Column(Boolean, default=False)  # Available to all organizations
    is_active = Column(Boolean, default=True)
    
    created_by = Column(String, ForeignKey("users.id"))
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    # Relationships
    organization = relationship("Organization")
    creator = relationship("User")
    
    def __repr__(self):
        return f"<AgentTemplate(id={self.id}, name={self.name}, category={self.category})>"
