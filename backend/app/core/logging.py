"""
Logging configuration for Clario
"""

import logging
import sys
from typing import Dict, Any
import structlog
from pythonjsonlogger import jsonlogger

from .config import settings


def setup_logging() -> None:
    """Setup application logging"""
    
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer() if settings.LOG_FORMAT == "json" else structlog.processors.KeyValueRenderer(),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure standard logging
    log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    
    if settings.LOG_FORMAT == "json":
        formatter = jsonlogger.JsonFormatter(
            '%(asctime)s %(name)s %(levelname)s %(message)s'
        )
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(log_level)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(console_handler)
    
    # Silence noisy loggers in development
    if settings.is_development:
        logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)


class RequestLogger:
    """Request logging utilities"""
    
    @staticmethod
    def log_request(request_id: str, method: str, path: str, **kwargs) -> None:
        """Log incoming request"""
        logger = structlog.get_logger("request")
        logger.info(
            "Request received",
            request_id=request_id,
            method=method,
            path=path,
            **kwargs
        )
    
    @staticmethod
    def log_response(request_id: str, status_code: int, duration: float, **kwargs) -> None:
        """Log response"""
        logger = structlog.get_logger("response")
        logger.info(
            "Request completed",
            request_id=request_id,
            status_code=status_code,
            duration_ms=round(duration * 1000, 2),
            **kwargs
        )
    
    @staticmethod
    def log_error(request_id: str, error: Exception, **kwargs) -> None:
        """Log request error"""
        logger = structlog.get_logger("error")
        logger.error(
            "Request failed",
            request_id=request_id,
            error=str(error),
            error_type=type(error).__name__,
            **kwargs
        )


class AILogger:
    """AI operations logging"""
    
    @staticmethod
    def log_agent_creation(agent_id: str, user_id: str, **kwargs) -> None:
        """Log agent creation"""
        logger = structlog.get_logger("ai.agent")
        logger.info(
            "Agent created",
            agent_id=agent_id,
            user_id=user_id,
            **kwargs
        )
    
    @staticmethod
    def log_meeting_discussion(meeting_id: str, participants: list, **kwargs) -> None:
        """Log meeting discussion"""
        logger = structlog.get_logger("ai.meeting")
        logger.info(
            "Meeting discussion started",
            meeting_id=meeting_id,
            participants=participants,
            **kwargs
        )
    
    @staticmethod
    def log_decision_made(decision_id: str, meeting_id: str, confidence: float, **kwargs) -> None:
        """Log decision generation"""
        logger = structlog.get_logger("ai.decision")
        logger.info(
            "Decision generated",
            decision_id=decision_id,
            meeting_id=meeting_id,
            confidence=confidence,
            **kwargs
        )
    
    @staticmethod
    def log_inference_call(service: str, model: str, tokens: int, duration: float, **kwargs) -> None:
        """Log AI inference call"""
        logger = structlog.get_logger("ai.inference")
        logger.info(
            "AI inference completed",
            service=service,
            model=model,
            tokens=tokens,
            duration_ms=round(duration * 1000, 2),
            **kwargs
        )


def get_logger(name: str) -> structlog.BoundLogger:
    """Get structured logger instance"""
    return structlog.get_logger(name)
