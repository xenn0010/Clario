"""
Meeting and related models
"""

from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, ForeignKey, JSON, Float
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from enum import Enum
import uuid
from datetime import datetime

from app.core.database import Base


class MeetingStatus(str, Enum):
    """Meeting status options"""
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class MeetingType(str, Enum):
    """Meeting type options"""
    DISCUSSION = "discussion"
    DECISION = "decision"
    BRAINSTORMING = "brainstorming"
    REVIEW = "review"
    PLANNING = "planning"
    STATUS_UPDATE = "status_update"


class Meeting(Base):
    """Meeting model"""
    __tablename__ = "meetings"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    organization_id = Column(String, ForeignKey("organizations.id"), nullable=False)
    
    # Meeting basic info
    title = Column(String, nullable=False)
    description = Column(Text)
    meeting_type = Column(String, default=MeetingType.DISCUSSION)
    status = Column(String, default=MeetingStatus.SCHEDULED)
    
    # Scheduling
    scheduled_at = Column(DateTime)
    duration_minutes = Column(Integer, default=30)
    timezone = Column(String, default="UTC")
    
    # Meeting content
    agenda_text = Column(Text)
    agenda_file_url = Column(String)  # PDF or document URL
    agenda_parsed_content = Column(JSON)  # Parsed agenda items
    
    # AI processing
    ai_summary = Column(Text)
    ai_confidence_score = Column(Float)
    processing_status = Column(String, default="pending")  # pending, processing, completed, failed
    
    # Meeting outcomes
    decisions_made = Column(JSON)  # List of decision IDs
    action_items = Column(JSON)  # List of action items
    key_points = Column(JSON)  # Key discussion points
    
    # Creator and participants
    created_by = Column(String, ForeignKey("users.id"), nullable=False)
    
    # Settings
    allow_ai_agents = Column(Boolean, default=True)
    require_human_approval = Column(Boolean, default=True)
    auto_generate_summary = Column(Boolean, default=True)
    
    # Timestamps
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    started_at = Column(DateTime)
    completed_at = Column(DateTime)
    
    # Relationships
    organization = relationship("Organization", back_populates="meetings")
    creator = relationship("User")
    participants = relationship("MeetingParticipant", back_populates="meeting")
    decisions = relationship("Decision", back_populates="meeting")
    discussion_logs = relationship("DiscussionLog", back_populates="meeting")
    
    def __repr__(self):
        return f"<Meeting(id={self.id}, title={self.title}, status={self.status})>"
    
    @property
    def is_completed(self) -> bool:
        """Check if meeting is completed"""
        return self.status == MeetingStatus.COMPLETED
    
    @property
    def has_decisions(self) -> bool:
        """Check if meeting has decisions"""
        return bool(self.decisions_made)
    
    @property
    def human_participants(self) -> list:
        """Get human participants"""
        return [p for p in self.participants if not p.is_ai_agent]
    
    @property
    def ai_participants(self) -> list:
        """Get AI agent participants"""
        return [p for p in self.participants if p.is_ai_agent]


class MeetingParticipant(Base):
    """Meeting participants (humans and AI agents)"""
    __tablename__ = "meeting_participants"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    meeting_id = Column(String, ForeignKey("meetings.id"), nullable=False)
    user_id = Column(String, ForeignKey("users.id"))  # Nullable for AI agents
    agent_id = Column(String, ForeignKey("agents.id"))  # Nullable for humans
    
    # Participant info
    role = Column(String, default="participant")  # participant, facilitator, observer
    is_required = Column(Boolean, default=True)
    is_ai_agent = Column(Boolean, default=False)
    
    # Participation tracking
    joined_at = Column(DateTime)
    left_at = Column(DateTime)
    participation_score = Column(Float)  # 0-1 based on engagement
    
    # Response tracking for humans
    invitation_sent_at = Column(DateTime)
    invitation_responded_at = Column(DateTime)
    response_status = Column(String)  # pending, accepted, declined
    
    created_at = Column(DateTime, server_default=func.now())
    
    # Relationships
    meeting = relationship("Meeting", back_populates="participants")
    user = relationship("User", back_populates="meeting_participants")
    agent = relationship("Agent", back_populates="meeting_participants")
    
    def __repr__(self):
        participant_type = "AI Agent" if self.is_ai_agent else "Human"
        return f"<MeetingParticipant(meeting={self.meeting_id}, type={participant_type})>"


class AgendaItem(Base):
    """Individual agenda items"""
    __tablename__ = "agenda_items"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    meeting_id = Column(String, ForeignKey("meetings.id"), nullable=False)
    
    # Item details
    title = Column(String, nullable=False)
    description = Column(Text)
    order_index = Column(Integer, nullable=False)
    
    # Timing
    estimated_duration = Column(Integer)  # minutes
    actual_duration = Column(Integer)
    
    # Item type and priority
    item_type = Column(String)  # discussion, decision, information, action
    priority = Column(String, default="medium")  # high, medium, low
    
    # Processing
    requires_decision = Column(Boolean, default=False)
    requires_vote = Column(Boolean, default=False)
    
    # Outcomes
    status = Column(String, default="pending")  # pending, discussed, decided, deferred
    notes = Column(Text)
    decision_id = Column(String, ForeignKey("decisions.id"))
    
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    # Relationships
    meeting = relationship("Meeting")
    decision = relationship("Decision")
    
    def __repr__(self):
        return f"<AgendaItem(id={self.id}, title={self.title}, meeting={self.meeting_id})>"


class DiscussionLog(Base):
    """Log of AI agent discussions"""
    __tablename__ = "discussion_logs"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    meeting_id = Column(String, ForeignKey("meetings.id"), nullable=False)
    agenda_item_id = Column(String, ForeignKey("agenda_items.id"))
    
    # Discussion details
    speaker_agent_id = Column(String, ForeignKey("agents.id"))
    message_content = Column(Text, nullable=False)
    message_type = Column(String)  # statement, question, response, decision
    
    # Context
    responding_to_id = Column(String, ForeignKey("discussion_logs.id"))  # For threaded discussions
    confidence_level = Column(Float)  # 0-1
    reasoning = Column(Text)  # AI reasoning behind the statement
    
    # Metadata
    tokens_used = Column(Integer)
    processing_time_ms = Column(Integer)
    
    timestamp = Column(DateTime, server_default=func.now())
    
    # Relationships
    meeting = relationship("Meeting", back_populates="discussion_logs")
    agenda_item = relationship("AgendaItem")
    speaker_agent = relationship("Agent")
    responding_to = relationship("DiscussionLog", remote_side=[id])
    
    def __repr__(self):
        return f"<DiscussionLog(id={self.id}, meeting={self.meeting_id}, speaker={self.speaker_agent_id})>"
