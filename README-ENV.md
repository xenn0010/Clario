# Clario Environment Configuration Guide

This guide explains how to configure all the required environment variables for the Clario AI Meetings Platform.

## 📋 Quick Setup

1. **Copy the appropriate environment file:**
   ```bash
   # For development
   cp .env.development .env
   
   # For production
   cp .env.production.example .env
   ```

2. **Fill in your API keys and credentials** (see sections below)

3. **Start the services:**
   ```bash
   docker-compose up -d
   ```

## 🔑 Required API Keys & Services

### 1. FriendliAI (AI Inference) - REQUIRED ⭐
**Purpose**: 3x faster LLM inference for AI agents

- **Get API Key**: [FriendliAI Console](https://suite.friendli.ai/)
- **Free Tier**: Available for development
- **Environment Variables**:
  ```env
  FRIENDLIAI_TOKEN="your-friendliai-token-here"
  FRIENDLIAI_MODEL="meta-llama-3.1-70b-instruct"
  ```

### 2. AWS Strands Agent SDK - REQUIRED ⭐
**Purpose**: AI agent framework and decision-making

- **Get Access**: [AWS Strands Documentation](https://docs.aws.amazon.com/prescriptive-guidance/latest/agentic-ai-frameworks/strands-agents.html)
- **Setup AWS Account**: [AWS Console](https://aws.amazon.com/console/)
- **Environment Variables**:
  ```env
  STRANDS_API_KEY="your-strands-api-key-here"
  AWS_ACCESS_KEY_ID="your-aws-access-key"
  AWS_SECRET_ACCESS_KEY="your-aws-secret-key"
  AWS_DEFAULT_REGION="us-east-1"
  ```

### 3. VAPI (Voice AI) - REQUIRED ⭐
**Purpose**: Voice interactions with meeting decisions

- **Get API Key**: [VAPI Dashboard](https://vapi.ai/)
- **Free Trial**: Available
- **Environment Variables**:
  ```env
  VAPI_API_KEY="your-vapi-api-key-here"
  VAPI_WEBHOOK_URL="https://yourdomain.com/api/v1/vapi/webhook"
  ```

### 4. OpenAI (Backup/Embeddings) - OPTIONAL
**Purpose**: Fallback for AI operations and text embeddings

- **Get API Key**: [OpenAI Platform](https://platform.openai.com/)
- **Environment Variables**:
  ```env
  OPENAI_API_KEY="your-openai-api-key-here"
  ```

## 🗄️ Database Services

### Local Development (Docker Compose)
All databases run locally via Docker - no additional setup required:

```bash
docker-compose up -d postgres weaviate neo4j redis minio
```

### Production Recommendations

#### Weaviate (Vector Database)
- **Recommended**: [Weaviate Cloud Service](https://weaviate.io/pricing)
- **Alternative**: Self-hosted Weaviate
- **Configuration**:
  ```env
  WEAVIATE_URL="https://your-cluster.weaviate.network"
  WEAVIATE_API_KEY="your-weaviate-api-key"
  ```

#### Neo4j (Graph Database)
- **Recommended**: [Neo4j AuraDB](https://neo4j.com/cloud/aura/)
- **Alternative**: Self-hosted Neo4j
- **Configuration**:
  ```env
  NEO4J_URI="neo4j+s://your-instance.databases.neo4j.io"
  NEO4J_PASSWORD="your-secure-password"
  ```

#### PostgreSQL (Primary Database)
- **Recommended**: AWS RDS, Google Cloud SQL, or Azure Database
- **Alternative**: Self-hosted PostgreSQL
- **Configuration**:
  ```env
  DATABASE_URL="postgresql://user:password@host:5432/database"
  ```

## 🔧 Service-Specific Configuration

### FriendliAI Models
Choose the appropriate model based on your needs:

```env
# Development (Faster, cheaper)
FRIENDLIAI_MODEL="meta-llama-3.1-8b-instruct"

# Production (More capable)
FRIENDLIAI_MODEL="meta-llama-3.1-70b-instruct"

# High performance (Most capable)
FRIENDLIAI_MODEL="meta-llama-3.1-405b-instruct"
```

### AWS Strands Configuration
Configure based on your AWS setup:

```env
# Bedrock Model Options
STRANDS_MODEL_ID="anthropic.claude-3-sonnet-20240229-v1:0"  # Balanced
STRANDS_MODEL_ID="anthropic.claude-3-haiku-20240307-v1:0"   # Fast/Cheap
STRANDS_MODEL_ID="anthropic.claude-3-opus-20240229-v1:0"    # Most Capable

# Agent Behavior
STRANDS_TEMPERATURE=0.7          # Creativity (0.0-1.0)
STRANDS_DISCUSSION_ROUNDS=3      # Meeting discussion rounds
STRANDS_MAX_AGENTS_PER_MEETING=10 # Max agents per meeting
```

### VAPI Voice Settings
Configure voice characteristics:

```env
# Voice Options
VAPI_DEFAULT_VOICE="nova"        # Options: nova, alloy, echo, fable, onyx, shimmer
VAPI_VOICE_SPEED=1.0            # Speech speed (0.25-4.0)
VAPI_VOICE_PITCH=0.0            # Voice pitch (-20 to 20)

# Call Settings
VAPI_MAX_CALL_DURATION=1800     # 30 minutes max
VAPI_SILENCE_TIMEOUT=30         # Seconds before timeout
```

## 🔐 Security Configuration

### Development
```env
SECRET_KEY="dev-secret-key-not-for-production"
JWT_SECRET="dev-jwt-secret"
DEBUG=true
```

### Production
```env
SECRET_KEY="your-super-secure-production-secret-key"
JWT_SECRET="your-production-jwt-secret"
ENCRYPTION_KEY="your-production-encryption-key"
DEBUG=false
```

## 📊 Optional Integrations

### Email (Production)
```env
# SendGrid
SMTP_HOST="smtp.sendgrid.net"
SMTP_USER="apikey"
SMTP_PASSWORD="your-sendgrid-api-key"

# Gmail
SMTP_HOST="smtp.gmail.com"
SMTP_USER="your-email@gmail.com"
SMTP_PASSWORD="your-app-password"
```

### Slack Integration
```env
SLACK_BOT_TOKEN="xoxb-your-slack-bot-token"
SLACK_SIGNING_SECRET="your-slack-signing-secret"
```

### Microsoft Teams
```env
TEAMS_APP_ID="your-teams-app-id"
TEAMS_APP_PASSWORD="your-teams-app-password"
```

## 🚀 Deployment Configurations

### Development
- Use `.env.development` 
- All services run locally via Docker
- Mock mode enabled for external APIs
- Verbose logging and debugging

### Staging
- Use production-like services
- Enable all features for testing
- Moderate logging

### Production
- Use `.env.production.example` as template
- Cloud-hosted databases
- Strict security settings
- Error tracking (Sentry)
- Performance monitoring

## ⚠️ Security Best Practices

1. **Never commit `.env` files** to version control
2. **Use different API keys** for development/staging/production
3. **Rotate API keys** regularly
4. **Use environment-specific secrets** management
5. **Enable encryption** for all database connections in production
6. **Use HTTPS** for all external webhook URLs
7. **Implement rate limiting** appropriate for your environment

## 🧪 Testing Configuration

### Running Tests
```env
TEST_DATABASE_URL="postgresql://clario_user:clario_password@localhost:5432/clario_test"
MOCK_EXTERNAL_SERVICES=true
```

### Mock Mode
For development without external API dependencies:
```env
STRANDS_MOCK_MODE=true
VAPI_MOCK_MODE=true
MOCK_EXTERNAL_SERVICES=true
```

## 📞 Support

- **Environment Issues**: Check the Docker Compose logs
- **API Key Issues**: Verify with service providers
- **Database Issues**: Check connection strings and credentials
- **Need Help**: See the main README.md for troubleshooting

## 🔄 Quick Validation

Test your configuration:
```bash
# Check database connections
docker-compose exec backend python -c "
from app.core.database import DatabaseManager
import asyncio
asyncio.run(DatabaseManager.check_connection())
"

# Check AI services
curl -X GET "http://localhost:8000/health"

# Check API documentation
open http://localhost:8000/docs
```
