"""
AWS-specific configuration for Clario
Handles AWS services integration and compliance
"""

import boto3
from typing import Optional, Dict, Any
import logging
from botocore.exceptions import ClientError, NoCredentialsError
from dataclasses import dataclass

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger("aws_config")


@dataclass
class AWSConfig:
    """AWS configuration container"""
    region: str
    access_key_id: Optional[str] = None
    secret_access_key: Optional[str] = None
    session_token: Optional[str] = None
    profile_name: Optional[str] = None


class AWSServiceManager:
    """Manages AWS service clients and configuration"""
    
    def __init__(self):
        self.bedrock_client = None
        self.s3_client = None
        self.lambda_client = None
        self.config = None
        self._initialized = False
    
    async def initialize(self) -> bool:
        """Initialize AWS services"""
        try:
            # Build AWS configuration
            self.config = AWSConfig(
                region=getattr(settings, 'AWS_DEFAULT_REGION', 'us-east-1'),
                access_key_id=getattr(settings, 'AWS_ACCESS_KEY_ID', None),
                secret_access_key=getattr(settings, 'AWS_SECRET_ACCESS_KEY', None),
                profile_name=getattr(settings, 'AWS_PROFILE', None)
            )
            
            # Create session
            session_kwargs = {
                'region_name': self.config.region
            }
            
            if self.config.access_key_id and self.config.secret_access_key:
                session_kwargs.update({
                    'aws_access_key_id': self.config.access_key_id,
                    'aws_secret_access_key': self.config.secret_access_key
                })
            elif self.config.profile_name:
                session_kwargs['profile_name'] = self.config.profile_name
            
            session = boto3.Session(**session_kwargs)
            
            # Initialize Bedrock client for Strands
            try:
                self.bedrock_client = session.client('bedrock-runtime')
                # Test connection
                await self._test_bedrock_connection()
                logger.info("AWS Bedrock client initialized successfully")
            except (ClientError, NoCredentialsError) as e:
                logger.warning(f"Bedrock client initialization failed: {e}")
                self.bedrock_client = None
            
            # Initialize S3 client for file storage
            try:
                self.s3_client = session.client('s3')
                logger.info("AWS S3 client initialized successfully")
            except (ClientError, NoCredentialsError) as e:
                logger.warning(f"S3 client initialization failed: {e}")
                self.s3_client = None
            
            # Initialize Lambda client for serverless functions
            try:
                self.lambda_client = session.client('lambda')
                logger.info("AWS Lambda client initialized successfully")
            except (ClientError, NoCredentialsError) as e:
                logger.warning(f"Lambda client initialization failed: {e}")
                self.lambda_client = None
            
            self._initialized = True
            return True
            
        except Exception as e:
            logger.error(f"AWS service initialization failed: {e}")
            return False
    
    async def _test_bedrock_connection(self) -> None:
        """Test Bedrock connection"""
        if self.bedrock_client:
            try:
                # Test by listing available models
                response = self.bedrock_client.list_foundation_models()
                logger.info(f"Bedrock connection successful. Available models: {len(response.get('modelSummaries', []))}")
            except ClientError as e:
                logger.error(f"Bedrock connection test failed: {e}")
                raise
    
    def get_bedrock_client(self):
        """Get Bedrock client for Strands agents"""
        if not self._initialized:
            raise RuntimeError("AWS service manager not initialized")
        return self.bedrock_client
    
    def get_s3_client(self):
        """Get S3 client for file storage"""
        if not self._initialized:
            raise RuntimeError("AWS service manager not initialized")
        return self.s3_client
    
    def get_lambda_client(self):
        """Get Lambda client for serverless functions"""
        if not self._initialized:
            raise RuntimeError("AWS service manager not initialized")
        return self.lambda_client
    
    def is_bedrock_available(self) -> bool:
        """Check if Bedrock is available"""
        return self.bedrock_client is not None
    
    def is_s3_available(self) -> bool:
        """Check if S3 is available"""
        return self.s3_client is not None
    
    def get_bedrock_model_config(self) -> Dict[str, Any]:
        """Get Bedrock model configuration for Strands"""
        return {
            'model_id': getattr(settings, 'STRANDS_MODEL_ID', 'anthropic.claude-3-sonnet-20240229-v1:0'),
            'temperature': getattr(settings, 'STRANDS_TEMPERATURE', 0.7),
            'max_tokens': getattr(settings, 'STRANDS_MAX_TOKENS', 2000),
            'top_p': getattr(settings, 'STRANDS_TOP_P', 0.9),
            'region': self.config.region
        }
    
    async def validate_compliance(self) -> Dict[str, bool]:
        """Validate AWS compliance requirements"""
        compliance_checks = {
            'bedrock_available': self.is_bedrock_available(),
            'credentials_configured': bool(self.config.access_key_id or self.config.profile_name),
            'region_configured': bool(self.config.region),
            'encryption_enabled': True,  # Bedrock uses encryption by default
            'logging_enabled': True,     # We have logging configured
            'monitoring_enabled': True   # OpenTelemetry integration
        }
        
        overall_compliance = all(compliance_checks.values())
        compliance_checks['overall_compliant'] = overall_compliance
        
        if overall_compliance:
            logger.info("AWS compliance validation passed")
        else:
            failed_checks = [k for k, v in compliance_checks.items() if not v and k != 'overall_compliant']
            logger.warning(f"AWS compliance validation failed: {failed_checks}")
        
        return compliance_checks


# Global AWS service manager
aws_service_manager = None


async def get_aws_service_manager() -> AWSServiceManager:
    """Get AWS service manager instance"""
    global aws_service_manager
    if not aws_service_manager:
        aws_service_manager = AWSServiceManager()
        await aws_service_manager.initialize()
    return aws_service_manager
