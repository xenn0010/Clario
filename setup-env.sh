#!/bin/bash

# =====================================================
# Clario Environment Setup Script
# =====================================================

echo "🚀 Setting up Clario environment configuration..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "docker-compose.yml" ]; then
    print_error "docker-compose.yml not found. Please run this script from the Clario root directory."
    exit 1
fi

# Create env-templates directory if it doesn't exist
mkdir -p env-templates

# Function to copy environment file
copy_env_file() {
    local env_type=$1
    local source_file="env-templates/env.$env_type"
    local target_file=".env"
    
    if [ -f "$source_file" ]; then
        cp "$source_file" "$target_file"
        print_success "Copied $source_file to $target_file"
    else
        print_error "Template file $source_file not found!"
        return 1
    fi
}

# Ask user which environment to set up
echo ""
print_status "Which environment would you like to set up?"
echo "1) Development (local Docker services)"
echo "2) Production (cloud services)"
echo "3) Example (template with all options)"
echo ""
read -p "Enter your choice (1-3): " env_choice

case $env_choice in
    1)
        print_status "Setting up development environment..."
        copy_env_file "development"
        ENV_TYPE="development"
        ;;
    2)
        print_status "Setting up production environment..."
        copy_env_file "production"
        ENV_TYPE="production"
        ;;
    3)
        print_status "Setting up example environment..."
        copy_env_file "example"
        ENV_TYPE="example"
        ;;
    *)
        print_error "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
print_success "Environment file created: .env"
print_warning "⚠️  IMPORTANT: You need to fill in your API keys and credentials!"

echo ""
print_status "Required API keys and services:"
echo ""

# FriendliAI
echo "🤖 FriendliAI (AI Inference) - REQUIRED"
echo "   Get API key: https://suite.friendli.ai/"
echo "   Variable: FRIENDLIAI_TOKEN"
echo ""

# AWS Strands
echo "🧠 AWS Strands Agent SDK - REQUIRED"
echo "   Setup AWS account: https://aws.amazon.com/console/"
echo "   Get Strands access: https://docs.aws.amazon.com/prescriptive-guidance/latest/agentic-ai-frameworks/strands-agents.html"
echo "   Variables: STRANDS_API_KEY, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY"
echo ""

# VAPI
echo "🗣️  VAPI (Voice AI) - REQUIRED"
echo "   Get API key: https://vapi.ai/"
echo "   Variable: VAPI_API_KEY"
echo ""

# Optional services
echo "📚 OpenAI (Optional - for embeddings/fallback)"
echo "   Get API key: https://platform.openai.com/"
echo "   Variable: OPENAI_API_KEY"
echo ""

if [ "$ENV_TYPE" = "production" ]; then
    echo "☁️  Production Database Services:"
    echo "   Weaviate Cloud: https://weaviate.io/pricing"
    echo "   Neo4j AuraDB: https://neo4j.com/cloud/aura/"
    echo "   PostgreSQL: AWS RDS, Google Cloud SQL, Azure Database"
    echo ""
fi

# Next steps
echo ""
print_status "Next steps:"
echo "1. Edit .env file and fill in your API keys"
echo "2. Start the services:"
if [ "$ENV_TYPE" = "development" ]; then
    echo "   docker-compose up -d"
else
    echo "   Configure your cloud databases first, then:"
    echo "   docker-compose up -d backend"
fi
echo "3. Check the health endpoint: http://localhost:8000/health"
echo "4. View API docs: http://localhost:8000/docs"
echo ""

# Offer to open the .env file
read -p "Would you like to open the .env file now? (y/n): " open_env

if [ "$open_env" = "y" ] || [ "$open_env" = "Y" ]; then
    # Try to open with common editors
    if command -v code &> /dev/null; then
        code .env
    elif command -v nano &> /dev/null; then
        nano .env
    elif command -v vim &> /dev/null; then
        vim .env
    else
        print_warning "No suitable editor found. Please edit .env manually."
    fi
fi

echo ""
print_success "Environment setup complete! 🎉"
print_warning "Remember to never commit your .env file to version control!"

# Create .gitignore entry if it doesn't exist
if [ ! -f ".gitignore" ]; then
    echo ".env" > .gitignore
    print_success "Created .gitignore with .env entry"
elif ! grep -q "^\.env$" .gitignore; then
    echo ".env" >> .gitignore
    print_success "Added .env to .gitignore"
fi
