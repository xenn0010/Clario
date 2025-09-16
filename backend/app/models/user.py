"""
User model
"""

from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, Enum as SQLEnum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from enum import Enum
import uuid
from datetime import datetime

from app.core.database import Base


class UserRole(str, Enum):
    """User roles in the system"""
    ADMIN = "admin"
    MEMBER = "member"
    VIEWER = "viewer"


class User(Base):
    """User model"""
    __tablename__ = "users"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, unique=True, index=True, nullable=False)
    full_name = Column(String, nullable=False)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    role = Column(SQLEnum(UserRole), default=UserRole.MEMBER)
    
    # Profile information
    job_title = Column(String)
    department = Column(String)
    bio = Column(Text)
    avatar_url = Column(String)
    
    # Decision-making profile
    decision_style = Column(String)  # analytical, intuitive, collaborative, etc.
    risk_tolerance = Column(String)  # low, medium, high
    communication_preference = Column(String)  # direct, diplomatic, detailed
    
    # Timestamps
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    last_login = Column(DateTime)
    
    # API access
    api_key_hash = Column(String)
    api_key_created_at = Column(DateTime)
    
    # Relationships
    organizations = relationship(
        "Organization", 
        secondary="organization_members", 
        back_populates="members"
    )
    owned_organizations = relationship(
        "Organization", 
        back_populates="owner",
        foreign_keys="Organization.owner_id"
    )
    agents = relationship("Agent", back_populates="user")
    meeting_participants = relationship("MeetingParticipant", back_populates="user")
    decision_profiles = relationship("DecisionProfile", back_populates="user")
    invitations_sent = relationship(
        "Invitation", 
        back_populates="inviter",
        foreign_keys="Invitation.inviter_id"
    )
    invitations_received = relationship(
        "Invitation", 
        back_populates="invitee",
        foreign_keys="Invitation.invitee_id"
    )
    
    def __repr__(self):
        return f"<User(id={self.id}, email={self.email}, full_name={self.full_name})>"
    
    @property
    def is_admin(self) -> bool:
        """Check if user is admin"""
        return self.role == UserRole.ADMIN
    
    @property
    def display_name(self) -> str:
        """Get display name"""
        return self.full_name or self.email.split("@")[0]
    
    def has_completed_profile(self) -> bool:
        """Check if user has completed their decision profile"""
        return bool(self.decision_style and self.risk_tolerance and self.communication_preference)
    
    def can_access_organization(self, org_id: str) -> bool:
        """Check if user can access organization"""
        return any(org.id == org_id for org in self.organizations)


class UserProfile(Base):
    """Extended user profile for decision-making analysis"""
    __tablename__ = "user_profiles"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, nullable=False, index=True)
    
    # Decision-making characteristics
    analytical_score = Column(Integer, default=0)  # 0-100
    intuitive_score = Column(Integer, default=0)
    collaborative_score = Column(Integer, default=0)
    directive_score = Column(Integer, default=0)
    
    # Communication style
    directness_level = Column(Integer, default=50)  # 0-100
    detail_preference = Column(Integer, default=50)
    formality_level = Column(Integer, default=50)
    
    # Meeting preferences
    preferred_meeting_length = Column(Integer, default=30)  # minutes
    max_participants = Column(Integer, default=8)
    agenda_structure_preference = Column(String)  # strict, flexible, none
    
    # Decision factors (JSON stored as text)
    important_factors = Column(Text)  # JSON: ["cost", "timeline", "risk", "impact"]
    decision_speed = Column(String)  # fast, moderate, deliberate
    consensus_requirement = Column(String)  # majority, consensus, authority
    
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    def __repr__(self):
        return f"<UserProfile(user_id={self.user_id})>"
