"""
Organization and team models
"""

from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, ForeignKey, Table
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from enum import Enum
import uuid
from datetime import datetime

from app.core.database import Base


# Association table for organization members
organization_members = Table(
    'organization_members',
    Base.metadata,
    Column('organization_id', String, ForeignKey('organizations.id'), primary_key=True),
    Column('user_id', String, ForeignKey('users.id'), primary_key=True),
    Column('role', String, default='member'),
    Column('joined_at', DateTime, server_default=func.now())
)


class Organization(Base):
    """Organization model"""
    __tablename__ = "organizations"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    description = Column(Text)
    
    # Organization settings
    max_team_size = Column(Integer, default=50)
    subscription_plan = Column(String, default="free")  # free, pro, enterprise
    
    # Owner information
    owner_id = Column(String, ForeignKey("users.id"), nullable=False)
    
    # Organization preferences
    default_meeting_duration = Column(Integer, default=30)  # minutes
    timezone = Column(String, default="UTC")
    working_hours_start = Column(String, default="09:00")
    working_hours_end = Column(String, default="17:00")
    
    # AI settings
    ai_decision_confidence_threshold = Column(Integer, default=80)  # 0-100
    enable_voice_interactions = Column(Boolean, default=True)
    enable_auto_decisions = Column(Boolean, default=False)
    
    # Status
    is_active = Column(Boolean, default=True)
    
    # Timestamps
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    # Relationships
    owner = relationship("User", back_populates="owned_organizations")
    members = relationship(
        "User", 
        secondary=organization_members, 
        back_populates="organizations"
    )
    meetings = relationship("Meeting", back_populates="organization")
    agents = relationship("Agent", back_populates="organization")
    invitations = relationship("Invitation", back_populates="organization")
    
    def __repr__(self):
        return f"<Organization(id={self.id}, name={self.name})>"
    
    @property
    def member_count(self) -> int:
        """Get number of members"""
        return len(self.members)
    
    @property
    def is_full(self) -> bool:
        """Check if organization is at capacity"""
        return self.member_count >= self.max_team_size
    
    def can_add_members(self, count: int = 1) -> bool:
        """Check if can add more members"""
        return (self.member_count + count) <= self.max_team_size
    
    def get_member_role(self, user_id: str) -> str:
        """Get user's role in organization"""
        # This would need to be implemented with a proper query
        # returning the role from organization_members table
        return "member"  # Default for now
    
    def is_owner(self, user_id: str) -> bool:
        """Check if user is owner"""
        return self.owner_id == user_id
    
    def is_member(self, user_id: str) -> bool:
        """Check if user is member"""
        return any(member.id == user_id for member in self.members)


class Team(Base):
    """Team within organization for structured groups"""
    __tablename__ = "teams"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    organization_id = Column(String, ForeignKey("organizations.id"), nullable=False)
    name = Column(String, nullable=False)
    description = Column(Text)
    
    # Team settings
    team_lead_id = Column(String, ForeignKey("users.id"))
    max_size = Column(Integer, default=10)
    
    # Meeting preferences
    default_meeting_type = Column(String, default="discussion")
    auto_schedule_meetings = Column(Boolean, default=False)
    
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    # Relationships
    organization = relationship("Organization")
    team_lead = relationship("User")
    
    def __repr__(self):
        return f"<Team(id={self.id}, name={self.name}, org={self.organization_id})>"


# Association table for team members
team_members = Table(
    'team_members',
    Base.metadata,
    Column('team_id', String, ForeignKey('teams.id'), primary_key=True),
    Column('user_id', String, ForeignKey('users.id'), primary_key=True),
    Column('role', String, default='member'),
    Column('joined_at', DateTime, server_default=func.now())
)
