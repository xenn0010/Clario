"""
Security utilities for authentication and authorization
"""

from datetime import datetime, timedelta
from typing import Any, Union, Optional
from passlib.context import CryptContext
from jose import jwt, JWTError
from fastapi import HTTPException, status
import secrets
import hashlib

from .config import settings

# Password context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT settings
ALGORITHM = "HS256"


class SecurityManager:
    """Security utilities manager"""
    
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash"""
        return pwd_context.verify(plain_password, hashed_password)
    
    @staticmethod
    def get_password_hash(password: str) -> str:
        """Generate password hash"""
        return pwd_context.hash(password)
    
    @staticmethod
    def create_access_token(
        subject: Union[str, Any], 
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create JWT access token"""
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(
                minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
            )
        
        to_encode = {
            "exp": expire,
            "sub": str(subject),
            "type": "access"
        }
        encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    
    @staticmethod
    def create_refresh_token(subject: Union[str, Any]) -> str:
        """Create JWT refresh token"""
        expire = datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
        to_encode = {
            "exp": expire,
            "sub": str(subject),
            "type": "refresh"
        }
        encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    
    @staticmethod
    def verify_token(token: str, token_type: str = "access") -> Optional[str]:
        """Verify JWT token and return subject"""
        try:
            payload = jwt.decode(
                token, settings.SECRET_KEY, algorithms=[ALGORITHM]
            )
            user_id: str = payload.get("sub")
            token_type_claim: str = payload.get("type", "access")
            
            if user_id is None or token_type_claim != token_type:
                return None
            
            return user_id
        except JWTError:
            return None
    
    @staticmethod
    def generate_api_key() -> str:
        """Generate secure API key"""
        return secrets.token_urlsafe(32)
    
    @staticmethod
    def generate_invitation_token() -> str:
        """Generate invitation token"""
        return secrets.token_urlsafe(16)
    
    @staticmethod
    def hash_api_key(api_key: str) -> str:
        """Hash API key for storage"""
        return hashlib.sha256(api_key.encode()).hexdigest()
    
    @staticmethod
    def verify_api_key(api_key: str, hashed_key: str) -> bool:
        """Verify API key against hash"""
        return hashlib.sha256(api_key.encode()).hexdigest() == hashed_key


class RateLimiter:
    """Rate limiting utilities"""
    
    def __init__(self, redis_client=None):
        self.redis_client = redis_client
    
    async def check_rate_limit(
        self, 
        key: str, 
        limit: int, 
        window: int = 60
    ) -> bool:
        """Check if request is within rate limit"""
        if not self.redis_client:
            return True  # No rate limiting if Redis not available
        
        try:
            current = await self.redis_client.get(key)
            if current is None:
                await self.redis_client.setex(key, window, 1)
                return True
            
            current_count = int(current)
            if current_count >= limit:
                return False
            
            await self.redis_client.incr(key)
            return True
        except Exception:
            return True  # Allow request if Redis fails


def create_security_headers() -> dict:
    """Create security headers for responses"""
    return {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Referrer-Policy": "strict-origin-when-cross-origin"
    }


# Convenience functions
verify_password = SecurityManager.verify_password
get_password_hash = SecurityManager.get_password_hash
create_access_token = SecurityManager.create_access_token
create_refresh_token = SecurityManager.create_refresh_token
verify_token = SecurityManager.verify_token
generate_api_key = SecurityManager.generate_api_key
