"""
Decision and decision profiling models
"""

from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, ForeignKey, JSON, Float
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from enum import Enum
import uuid
from datetime import datetime

from app.core.database import Base


class DecisionStatus(str, Enum):
    """Decision status options"""
    PENDING = "pending"
    PROPOSED = "proposed"
    APPROVED = "approved"
    REJECTED = "rejected"
    IMPLEMENTED = "implemented"
    CANCELLED = "cancelled"


class DecisionType(str, Enum):
    """Decision type categories"""
    STRATEGIC = "strategic"
    OPERATIONAL = "operational"
    FINANCIAL = "financial"
    TECHNICAL = "technical"
    PERSONNEL = "personnel"
    POLICY = "policy"
    RESOURCE = "resource"


class DecisionUrgency(str, Enum):
    """Decision urgency levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Decision(Base):
    """Decision model for tracking meeting decisions"""
    __tablename__ = "decisions"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    meeting_id = Column(String, ForeignKey("meetings.id"), nullable=False)
    agenda_item_id = Column(String, ForeignKey("agenda_items.id"))
    organization_id = Column(String, ForeignKey("organizations.id"), nullable=False)
    
    # Decision content
    title = Column(String, nullable=False)
    description = Column(Text, nullable=False)
    decision_type = Column(String, default=DecisionType.OPERATIONAL)
    urgency = Column(String, default=DecisionUrgency.MEDIUM)
    
    # Decision details
    options_considered = Column(JSON)  # List of options that were considered
    chosen_option = Column(JSON)  # The selected option with details
    reasoning = Column(Text)  # Detailed reasoning behind the decision
    assumptions = Column(JSON)  # Key assumptions made
    risks = Column(JSON)  # Identified risks and mitigation strategies
    
    # Impact assessment
    impact_areas = Column(JSON)  # Areas affected by this decision
    estimated_cost = Column(Float)  # Financial impact
    estimated_timeline = Column(String)  # Implementation timeline
    success_metrics = Column(JSON)  # How success will be measured
    
    # Decision process
    decision_method = Column(String)  # consensus, majority, authority, etc.
    participants_count = Column(Integer)
    ai_agents_count = Column(Integer)
    human_participants_count = Column(Integer)
    
    # AI Analysis
    ai_confidence_score = Column(Float)  # 0-1, AI's confidence in this decision
    ai_reasoning = Column(Text)  # AI's reasoning process
    agent_id = Column(String, ForeignKey("agents.id"))  # Primary agent that made/influenced decision
    
    # Approval workflow
    requires_approval = Column(Boolean, default=True)
    approved_by = Column(String, ForeignKey("users.id"))
    approved_at = Column(DateTime)
    approval_notes = Column(Text)
    
    # Status and tracking
    status = Column(String, default=DecisionStatus.PENDING)
    priority = Column(Integer, default=5)  # 1-10 scale
    
    # Implementation
    implementation_start = Column(DateTime)
    implementation_deadline = Column(DateTime)
    implementation_notes = Column(Text)
    implementation_progress = Column(Integer, default=0)  # 0-100%
    
    # Review and feedback
    review_scheduled_at = Column(DateTime)
    actual_outcome = Column(Text)
    lessons_learned = Column(Text)
    success_rating = Column(Float)  # 0-1, how successful was this decision
    
    # Timestamps
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    decided_at = Column(DateTime)
    
    # Relationships
    meeting = relationship("Meeting", back_populates="decisions")
    agenda_item = relationship("AgendaItem")
    organization = relationship("Organization")
    agent = relationship("Agent", back_populates="decisions")
    approver = relationship("User")
    action_items = relationship("ActionItem", back_populates="decision")
    
    def __repr__(self):
        return f"<Decision(id={self.id}, title={self.title}, status={self.status})>"
    
    @property
    def is_approved(self) -> bool:
        """Check if decision is approved"""
        return self.status == DecisionStatus.APPROVED
    
    @property
    def is_implemented(self) -> bool:
        """Check if decision is implemented"""
        return self.status == DecisionStatus.IMPLEMENTED
    
    @property
    def days_since_decision(self) -> int:
        """Get days since decision was made"""
        if not self.decided_at:
            return 0
        return (datetime.utcnow() - self.decided_at).days
    
    def get_implementation_status(self) -> str:
        """Get human-readable implementation status"""
        if self.implementation_progress == 0:
            return "Not started"
        elif self.implementation_progress < 100:
            return f"In progress ({self.implementation_progress}%)"
        else:
            return "Completed"


class DecisionProfile(Base):
    """User decision-making profile based on questionnaire"""
    __tablename__ = "decision_profiles"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False, unique=True)
    
    # Decision-making style assessment
    analytical_score = Column(Integer, default=0)  # 0-100
    intuitive_score = Column(Integer, default=0)
    directive_score = Column(Integer, default=0)
    conceptual_score = Column(Integer, default=0)
    behavioral_score = Column(Integer, default=0)
    
    # Risk assessment
    risk_tolerance = Column(String)  # low, medium, high
    risk_assessment_approach = Column(String)  # thorough, balanced, quick
    
    # Information processing
    information_preference = Column(String)  # detailed, summary, visual
    decision_speed = Column(String)  # fast, moderate, deliberate
    consultation_style = Column(String)  # independent, selective, collaborative
    
    # Values and priorities
    primary_values = Column(JSON)  # ["efficiency", "quality", "innovation", "stability"]
    decision_factors = Column(JSON)  # Weighted factors for decisions
    
    # Communication preferences
    communication_style = Column(String)  # direct, diplomatic, analytical
    feedback_preference = Column(String)  # immediate, periodic, minimal
    
    # Context preferences
    meeting_participation = Column(String)  # active, moderate, observer
    group_dynamics_preference = Column(String)  # leader, collaborator, supporter
    
    # Learning and adaptation
    learning_style = Column(String)  # experiential, analytical, observational
    change_adaptability = Column(String)  # high, medium, low
    
    # Questionnaire responses (raw data)
    questionnaire_responses = Column(JSON)
    questionnaire_version = Column(String, default="1.0")
    
    # Computed scores
    decision_consistency = Column(Float, default=0.0)  # How consistent their decisions are
    confidence_calibration = Column(Float, default=0.0)  # How well-calibrated their confidence is
    
    # Profile status
    is_complete = Column(Boolean, default=False)
    last_updated = Column(DateTime, server_default=func.now(), onupdate=func.now())
    created_at = Column(DateTime, server_default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="decision_profiles")
    
    def __repr__(self):
        return f"<DecisionProfile(user_id={self.user_id}, complete={self.is_complete})>"
    
    @property
    def dominant_style(self) -> str:
        """Get the dominant decision-making style"""
        scores = {
            'analytical': self.analytical_score,
            'intuitive': self.intuitive_score,
            'directive': self.directive_score,
            'conceptual': self.conceptual_score,
            'behavioral': self.behavioral_score
        }
        return max(scores, key=scores.get)
    
    @property
    def profile_completeness(self) -> float:
        """Calculate profile completeness percentage"""
        required_fields = [
            'risk_tolerance', 'decision_speed', 'consultation_style',
            'communication_style', 'meeting_participation'
        ]
        completed = sum(1 for field in required_fields if getattr(self, field))
        return (completed / len(required_fields)) * 100


class ActionItem(Base):
    """Action items resulting from decisions"""
    __tablename__ = "action_items"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    decision_id = Column(String, ForeignKey("decisions.id"), nullable=False)
    meeting_id = Column(String, ForeignKey("meetings.id"), nullable=False)
    
    # Action details
    title = Column(String, nullable=False)
    description = Column(Text)
    
    # Assignment
    assigned_to = Column(String, ForeignKey("users.id"))
    assigned_by = Column(String, ForeignKey("users.id"))
    
    # Timeline
    due_date = Column(DateTime)
    estimated_effort = Column(String)  # hours, days, weeks
    
    # Status
    status = Column(String, default="pending")  # pending, in_progress, completed, cancelled
    priority = Column(String, default="medium")  # low, medium, high
    progress = Column(Integer, default=0)  # 0-100%
    
    # Updates and notes
    notes = Column(Text)
    completion_notes = Column(Text)
    
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    completed_at = Column(DateTime)
    
    # Relationships
    decision = relationship("Decision", back_populates="action_items")
    meeting = relationship("Meeting")
    assignee = relationship("User", foreign_keys=[assigned_to])
    assigner = relationship("User", foreign_keys=[assigned_by])
    
    def __repr__(self):
        return f"<ActionItem(id={self.id}, title={self.title}, status={self.status})>"
    
    @property
    def is_overdue(self) -> bool:
        """Check if action item is overdue"""
        if not self.due_date or self.status == "completed":
            return False
        return datetime.utcnow() > self.due_date
    
    @property
    def days_until_due(self) -> int:
        """Get days until due date"""
        if not self.due_date:
            return 0
        return (self.due_date - datetime.utcnow()).days
