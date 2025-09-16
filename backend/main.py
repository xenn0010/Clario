"""
Clario - AI Meetings Platform
Main FastAPI application entry point
"""

from fastapi import FastAPI, Middleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import uvicorn
import logging

from app.core.config import settings
from app.core.database import init_db
from app.core.logging import setup_logging
from app.api.v1.api import api_router
from app.middleware.error_handler import ErrorHandlerMiddleware
from app.middleware.request_logging import RequestLoggingMiddleware

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logger.info("Starting up Clario API server...")
    
    # Initialize database
    await init_db()
    
    # Initialize services (skip externals when mocking)
    from app.services.ai.friendli_service import FriendliService
    try:
        await FriendliService.initialize()
        logger.info("Friendli service initialized")
    except Exception as e:
        logger.warning(f"Friendli init warning: {e}")

    if not getattr(settings, "MOCK_EXTERNAL_SERVICES", False):
        from app.services.vector.weaviate_service import WeaviateService
        from app.services.graph.neo4j_service import Neo4jService
        try:
            await WeaviateService.initialize()
            await Neo4jService.initialize()
            logger.info("External services initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize external services: {e}")
            raise
    else:
        logger.info("MOCK_EXTERNAL_SERVICES enabled: skipping Weaviate/Neo4j initialization")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Clario API server...")
    await FriendliService.cleanup()
    if not getattr(settings, "MOCK_EXTERNAL_SERVICES", False):
        from app.services.vector.weaviate_service import WeaviateService
        from app.services.graph.neo4j_service import Neo4jService
        await WeaviateService.cleanup()
        await Neo4jService.cleanup()


# Create FastAPI application
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="AI-powered meetings platform that replaces human attendance with intelligent agents",
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=settings.ALLOWED_HOSTS
)

app.add_middleware(RequestLoggingMiddleware)
app.add_middleware(ErrorHandlerMiddleware)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include API routes
app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/", tags=["root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to Clario - The Future of AI Meetings",
        "docs": "/docs",
        "version": settings.VERSION,
        "status": "operational"
    }


@app.get("/health", tags=["health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True if settings.ENVIRONMENT == "development" else False,
        log_level="info",
        access_log=True
    )
