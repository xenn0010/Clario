"""
Database configuration and session management
"""

from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from typing import AsyncGenerator
import logging

from .config import settings

logger = logging.getLogger(__name__)

# SQLAlchemy engine
engine = create_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    pool_pre_ping=True,
    pool_recycle=300,
)

# Async SQLAlchemy engine
async_engine = create_async_engine(
    settings.async_database_url,
    echo=settings.DEBUG,
    pool_pre_ping=True,
    pool_recycle=300,
)

# Session makers
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# Base class for models
Base = declarative_base()

# Metadata for Alembic
metadata = MetaData()


def get_db():
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency to get async database session"""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db() -> None:
    """Initialize database tables"""
    try:
        # Import all models to register them with SQLAlchemy
        from app.models import user, organization, meeting, agent, decision, invitation
        
        # Create tables
        async with async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


class DatabaseManager:
    """Database manager for advanced operations"""
    
    @staticmethod
    async def create_database():
        """Create database if it doesn't exist"""
        # This would be implemented for production deployment
        pass
    
    @staticmethod
    async def drop_database():
        """Drop all tables (use with caution!)"""
        async with async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
    
    @staticmethod
    async def reset_database():
        """Reset database (drop and recreate)"""
        await DatabaseManager.drop_database()
        await init_db()
    
    @staticmethod
    async def check_connection():
        """Check database connection"""
        try:
            async with AsyncSessionLocal() as session:
                await session.execute("SELECT 1")
            return True
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return False
