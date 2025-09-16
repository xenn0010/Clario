"""
Invitation models for team onboarding
"""

from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, ForeignKey, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from enum import Enum
import uuid
from datetime import datetime, timedelta

from app.core.database import Base


class InvitationStatus(str, Enum):
    """Invitation status options"""
    PENDING = "pending"
    SENT = "sent"
    ACCEPTED = "accepted"
    DECLINED = "declined"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


class InvitationType(str, Enum):
    """Invitation type options"""
    ORGANIZATION = "organization"
    TEAM = "team"
    MEETING = "meeting"


class Invitation(Base):
    """Invitation model for onboarding team members"""
    __tablename__ = "invitations"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    organization_id = Column(String, ForeignKey("organizations.id"), nullable=False)
    
    # Invitation details
    invitation_type = Column(String, default=InvitationType.ORGANIZATION)
    email = Column(String, nullable=False, index=True)
    token = Column(String, unique=True, nullable=False, index=True)
    
    # Inviter and invitee
    inviter_id = Column(String, ForeignKey("users.id"), nullable=False)
    invitee_id = Column(String, ForeignKey("users.id"))  # Set when user accepts
    
    # Role and permissions
    role = Column(String, default="member")  # member, admin, viewer
    permissions = Column(JSON)  # Specific permissions if needed
    
    # Personal message
    message = Column(Text)
    welcome_message = Column(Text)
    
    # Status and timing
    status = Column(String, default=InvitationStatus.PENDING)
    expires_at = Column(DateTime)
    
    # Tracking
    sent_at = Column(DateTime)
    opened_at = Column(DateTime)  # When invitation email was opened
    responded_at = Column(DateTime)
    
    # Response details
    response_message = Column(Text)
    decline_reason = Column(String)
    
    # Onboarding configuration
    requires_profile_completion = Column(Boolean, default=True)
    requires_questionnaire = Column(Boolean, default=True)
    auto_create_agent = Column(Boolean, default=True)
    
    # Metadata
    ip_address = Column(String)  # IP when responded
    user_agent = Column(String)  # Browser info when responded
    
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    # Relationships
    organization = relationship("Organization", back_populates="invitations")
    inviter = relationship("User", back_populates="invitations_sent", foreign_keys=[inviter_id])
    invitee = relationship("User", back_populates="invitations_received", foreign_keys=[invitee_id])
    
    def __repr__(self):
        return f"<Invitation(id={self.id}, email={self.email}, status={self.status})>"
    
    @property
    def is_expired(self) -> bool:
        """Check if invitation is expired"""
        if not self.expires_at:
            return False
        return datetime.utcnow() > self.expires_at
    
    @property
    def is_pending(self) -> bool:
        """Check if invitation is still pending"""
        return self.status == InvitationStatus.PENDING and not self.is_expired
    
    @property
    def days_until_expiry(self) -> int:
        """Get days until expiry"""
        if not self.expires_at:
            return 0
        return max(0, (self.expires_at - datetime.utcnow()).days)
    
    def can_be_accepted(self) -> bool:
        """Check if invitation can be accepted"""
        return (
            self.status in [InvitationStatus.PENDING, InvitationStatus.SENT] and
            not self.is_expired
        )
    
    def set_expiry(self, days: int = 7) -> None:
        """Set expiry date"""
        self.expires_at = datetime.utcnow() + timedelta(days=days)


class OnboardingStep(Base):
    """Track user onboarding progress"""
    __tablename__ = "onboarding_steps"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, ForeignKey("users.id"), nullable=False)
    organization_id = Column(String, ForeignKey("organizations.id"), nullable=False)
    invitation_id = Column(String, ForeignKey("invitations.id"))
    
    # Step details
    step_name = Column(String, nullable=False)
    step_order = Column(Integer, nullable=False)
    is_required = Column(Boolean, default=True)
    
    # Status
    is_completed = Column(Boolean, default=False)
    completed_at = Column(DateTime)
    
    # Data collected in this step
    step_data = Column(JSON)
    
    # Progress tracking
    attempts = Column(Integer, default=0)
    last_attempt_at = Column(DateTime)
    
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    # Relationships
    user = relationship("User")
    organization = relationship("Organization")
    invitation = relationship("Invitation")
    
    def __repr__(self):
        return f"<OnboardingStep(user={self.user_id}, step={self.step_name}, completed={self.is_completed})>"


class OnboardingTemplate(Base):
    """Templates for different onboarding flows"""
    __tablename__ = "onboarding_templates"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    organization_id = Column(String, ForeignKey("organizations.id"))
    
    # Template details
    name = Column(String, nullable=False)
    description = Column(Text)
    
    # Template configuration
    steps = Column(JSON, nullable=False)  # Ordered list of onboarding steps
    estimated_duration = Column(Integer)  # Estimated minutes to complete
    
    # Usage settings
    is_default = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    
    # For role-specific onboarding
    target_roles = Column(JSON)  # Which roles this template applies to
    
    created_by = Column(String, ForeignKey("users.id"))
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    
    # Relationships
    organization = relationship("Organization")
    creator = relationship("User")
    
    def __repr__(self):
        return f"<OnboardingTemplate(id={self.id}, name={self.name})>"


class InvitationAnalytics(Base):
    """Analytics for invitation campaigns"""
    __tablename__ = "invitation_analytics"
    
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    organization_id = Column(String, ForeignKey("organizations.id"), nullable=False)
    
    # Time period
    period_start = Column(DateTime, nullable=False)
    period_end = Column(DateTime, nullable=False)
    
    # Metrics
    invitations_sent = Column(Integer, default=0)
    invitations_opened = Column(Integer, default=0)
    invitations_accepted = Column(Integer, default=0)
    invitations_declined = Column(Integer, default=0)
    invitations_expired = Column(Integer, default=0)
    
    # Calculated rates
    open_rate = Column(Float, default=0.0)  # opened / sent
    acceptance_rate = Column(Float, default=0.0)  # accepted / sent
    decline_rate = Column(Float, default=0.0)  # declined / sent
    
    # Average times
    avg_response_time_hours = Column(Float, default=0.0)
    avg_onboarding_completion_time_hours = Column(Float, default=0.0)
    
    # Breakdown by role
    analytics_by_role = Column(JSON)
    
    created_at = Column(DateTime, server_default=func.now())
    
    # Relationships
    organization = relationship("Organization")
    
    def __repr__(self):
        return f"<InvitationAnalytics(org={self.organization_id}, period={self.period_start}-{self.period_end})>"
