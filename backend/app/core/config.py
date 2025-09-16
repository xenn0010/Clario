"""
Clario Application Configuration
Centralized configuration management using Pydantic Settings
"""

from typing import List, Optional, Union
from pydantic import AnyHttpUrl, Field, validator
from pydantic_settings import BaseSettings
import secrets
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings"""
    
    # Basic App Info
    PROJECT_NAME: str = "Clario AI Meetings Platform"
    VERSION: str = "0.1.0"
    DESCRIPTION: str = "AI-powered meetings platform with intelligent agents"
    ENVIRONMENT: str = Field(default="development", env="ENVIRONMENT")
    DEBUG: bool = Field(default=True, env="DEBUG")
    
    # API Configuration
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = Field(default_factory=lambda: secrets.token_urlsafe(32))
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 days
    REFRESH_TOKEN_EXPIRE_DAYS: int = 30
    
    # Security
    ALLOWED_HOSTS: List[str] = ["localhost", "127.0.0.1", "0.0.0.0"]
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = [
        "http://localhost:3000",  # React dev server
        "http://localhost:8080",  # Vue dev server
        "http://localhost:4200",  # Angular dev server
    ]
    
    @validator("BACKEND_CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)
    
    # Database Configuration
    DATABASE_URL: str = Field(
        default="postgresql://clario_user:clario_password@localhost:5432/clario",
        env="DATABASE_URL"
    )
    ASYNC_DATABASE_URL: str = Field(
        default="postgresql+asyncpg://clario_user:clario_password@localhost:5432/clario",
        env="ASYNC_DATABASE_URL"
    )
    
    # Vector Database (Weaviate)
    WEAVIATE_URL: str = Field(default="http://localhost:8080", env="WEAVIATE_URL")
    WEAVIATE_API_KEY: Optional[str] = Field(default=None, env="WEAVIATE_API_KEY")
    WEAVIATE_TIMEOUT: int = Field(default=60, env="WEAVIATE_TIMEOUT")
    WEAVIATE_BATCH_SIZE: int = Field(default=100, env="WEAVIATE_BATCH_SIZE")
    
    # Graph Database (Neo4j)
    NEO4J_URI: str = Field(default="bolt://localhost:7687", env="NEO4J_URI")
    NEO4J_USER: str = Field(default="neo4j", env="NEO4J_USER")
    NEO4J_PASSWORD: str = Field(default="clario_password", env="NEO4J_PASSWORD")
    NEO4J_DATABASE: str = Field(default="neo4j", env="NEO4J_DATABASE")
    NEO4J_ENCRYPTED: bool = Field(default=False, env="NEO4J_ENCRYPTED")
    
    # Redis Configuration
    REDIS_URL: str = Field(default="redis://localhost:6379", env="REDIS_URL")
    
    # File Storage (MinIO/S3)
    MINIO_ENDPOINT: str = Field(default="localhost:9000", env="MINIO_ENDPOINT")
    MINIO_ACCESS_KEY: str = Field(default="clario_admin", env="MINIO_ACCESS_KEY")
    MINIO_SECRET_KEY: str = Field(default="clario_password", env="MINIO_SECRET_KEY")
    MINIO_BUCKET_NAME: str = Field(default="clario-files", env="MINIO_BUCKET_NAME")
    MINIO_SECURE: bool = Field(default=False, env="MINIO_SECURE")
    
    # AI Service Configuration
    
    # FriendliAI
    FRIENDLIAI_TOKEN: str = Field(..., env="FRIENDLIAI_TOKEN")
    FRIENDLIAI_BASE_URL: str = Field(
        default="https://api.friendli.ai/v1", 
        env="FRIENDLIAI_BASE_URL"
    )
    FRIENDLIAI_MODEL: str = Field(
        default="meta-llama-3.1-70b-instruct",
        env="FRIENDLIAI_MODEL"
    )
    
    # Strands Agent SDK
    STRANDS_API_KEY: str = Field(..., env="STRANDS_API_KEY")
    STRANDS_BASE_URL: str = Field(
        default="https://api.strands.ai/v1",
        env="STRANDS_BASE_URL"
    )
    
    # VAPI (Voice AI)
    VAPI_API_KEY: str = Field(..., env="VAPI_API_KEY")
    VAPI_BASE_URL: str = Field(
        default="https://api.vapi.ai/v1",
        env="VAPI_BASE_URL"
    )
    VAPI_DEFAULT_VOICE: str = Field(default="nova", env="VAPI_DEFAULT_VOICE")
    VAPI_FIRST_MESSAGE: str = Field(default="Hello! I'm Clario's decision companion.", env="VAPI_FIRST_MESSAGE")
    VAPI_SYSTEM_MESSAGE: str = Field(
        default="You are Clario's AI assistant helping users discuss meeting decisions.",
        env="VAPI_SYSTEM_MESSAGE"
    )
    VAPI_SILENCE_TIMEOUT: int = Field(default=30, env="VAPI_SILENCE_TIMEOUT")
    VAPI_MAX_DURATION: int = Field(default=900, env="VAPI_MAX_DURATION")
    VAPI_WEBHOOK_URL: Optional[str] = Field(default=None, env="VAPI_WEBHOOK_URL")
    VAPI_WEBHOOK_SECRET: Optional[str] = Field(default=None, env="VAPI_WEBHOOK_SECRET")
    
    # OpenAI (Backup/Additional capabilities)
    OPENAI_API_KEY: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    
    # Email Configuration
    SMTP_TLS: bool = Field(default=True, env="SMTP_TLS")
    SMTP_PORT: Optional[int] = Field(default=587, env="SMTP_PORT")
    SMTP_HOST: Optional[str] = Field(default="smtp.gmail.com", env="SMTP_HOST")
    SMTP_USER: Optional[str] = Field(default=None, env="SMTP_USER")
    SMTP_PASSWORD: Optional[str] = Field(default=None, env="SMTP_PASSWORD")
    EMAILS_FROM_EMAIL: Optional[str] = Field(default=None, env="EMAILS_FROM_EMAIL")
    EMAILS_FROM_NAME: Optional[str] = Field(default="Clario", env="EMAILS_FROM_NAME")
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = Field(default=60, env="RATE_LIMIT_PER_MINUTE")
    
    # Logging
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FORMAT: str = Field(default="json", env="LOG_FORMAT")  # json or text
    
    # Celery (Background Tasks)
    CELERY_BROKER_URL: str = Field(default="redis://localhost:6379/0", env="CELERY_BROKER_URL")
    CELERY_RESULT_BACKEND: str = Field(default="redis://localhost:6379/0", env="CELERY_RESULT_BACKEND")
    
    # Monitoring
    ENABLE_METRICS: bool = Field(default=True, env="ENABLE_METRICS")
    METRICS_PORT: int = Field(default=8001, env="METRICS_PORT")
    
    # Feature Flags
    ENABLE_AGENT_DISCUSSIONS: bool = Field(default=True, env="ENABLE_AGENT_DISCUSSIONS")
    ENABLE_VOICE_INTERACTIONS: bool = Field(default=True, env="ENABLE_VOICE_INTERACTIONS")
    ENABLE_ANALYTICS: bool = Field(default=True, env="ENABLE_ANALYTICS")
    MOCK_EXTERNAL_SERVICES: bool = Field(default=False, env="MOCK_EXTERNAL_SERVICES")
    
    # Application Limits
    MAX_TEAM_SIZE: int = Field(default=100, env="MAX_TEAM_SIZE")
    MAX_MEETINGS_PER_ORG: int = Field(default=1000, env="MAX_MEETINGS_PER_ORG")
    MAX_FILE_SIZE_MB: int = Field(default=50, env="MAX_FILE_SIZE_MB")
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        
    # Computed properties
    @property
    def async_database_url(self) -> str:
        """Get async database URL"""
        if self.ASYNC_DATABASE_URL:
            return self.ASYNC_DATABASE_URL
        return self.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")
    
    @property
    def is_development(self) -> bool:
        """Check if running in development mode"""
        return self.ENVIRONMENT.lower() == "development"
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode"""
        return self.ENVIRONMENT.lower() == "production"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Global settings instance
settings = get_settings()




