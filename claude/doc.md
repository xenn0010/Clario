# FriendliAI Platform - Complete Documentation

## Table of Contents
1. [Overview](#friendliai-overview)
2. [Installation](#friendliai-installation)
3. [Python SDK](#friendliai-python-sdk)
4. [JavaScript/TypeScript SDK](#friendliai-javascript-sdk)
5. [Model Deployment](#model-deployment)
6. [Optimization Features](#optimization-features)
7. [Integration Examples](#friendliai-integrations)
8. [Best Practices](#friendliai-best-practices)

## FriendliAI Overview

**FriendliAI** is a generative AI infrastructure platform providing up to 3x faster inference with 50-90% GPU cost reduction through proprietary optimization techniques.

### Key Features
- **Iteration Batching**: Revolutionary continuous batching technology
- **Native Quantization**: FP8, INT8, AWQ without accuracy loss  
- **Multi-LoRA Support**: Multiple adapters on single endpoint
- **Scale-to-Zero**: Eliminate costs during idle periods
- **3x Faster Output**: Industry-leading inference speeds

## FriendliAI Installation

### Python SDK

```bash
# PyPI
pip install friendli

# With extras
pip install friendli[langchain]
pip install friendli[llamaindex]

# Development
git clone https://github.com/friendliai/friendli-python
cd friendli-python
pip install -e ".[dev]"
```

### JavaScript/TypeScript

```bash
# NPM
npm install @friendliai/ai-provider

# Yarn
yarn add @friendliai/ai-provider

# PNPM
pnpm add @friendliai/ai-provider
```

## FriendliAI Python SDK

### Complete Python Implementation

```python
from friendli import Friendli, AsyncFriendli
from friendli.core import ChatCompletionRequest, CompletionRequest
from friendli.schema import (
    Message, 
    ModelConfig, 
    DedicatedEndpointConfig,
    ServerlessEndpointConfig,
    ContainerConfig,
    QuantizationConfig,
    LoRAConfig,
    GenerationConfig
)
from typing import List, Dict, Any, Optional, AsyncIterator, Union
import asyncio
import json
import logging
from dataclasses import dataclass
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Available model types"""
    LLAMA_3_1_8B = "meta-llama-3.1-8b-instruct"
    LLAMA_3_1_70B = "meta-llama-3.1-70b-instruct"
    LLAMA_3_1_405B = "meta-llama-3.1-405b-instruct"
    MIXTRAL_8X7B = "mixtral-8x7b-instruct"
    QWEN_72B = "qwen-2.5-72b-instruct"
    DEEPSEEK_V3 = "deepseek-r1"
    EXAONE_32B = "lgai-exaone-4.0.1-32b-instruct"
    CUSTOM = "custom"

class DeploymentType(Enum):
    """Deployment options"""
    SERVERLESS = "serverless"
    DEDICATED = "dedicated"
    CONTAINER = "container"

@dataclass
class FriendliConfig:
    """FriendliAI configuration"""
    token: str
    base_url: str = "https://api.friendli.ai/v1"
    deployment_type: DeploymentType = DeploymentType.SERVERLESS
    model_type: ModelType = ModelType.LLAMA_3_1_70B
    timeout: int = 300

class FriendliClient:
    """Complete FriendliAI client implementation"""
    
    def __init__(self, config: FriendliConfig):
        self.config = config
        self.client = Friendli(
            token=config.token,
            base_url=config.base_url,
            timeout=config.timeout
        )
        self.async_client = AsyncFriendli(
            token=config.token,
            base_url=config.base_url,
            timeout=config.timeout
        )
    
    # Chat Completions
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        top_p: float = 0.9,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        stream: bool = False,
        tools: Optional[List[Dict]] = None,
        response_format: Optional[Dict] = None
    ) -> Union[Dict, AsyncIterator]:
        """Generate chat completion"""
        
        model = model or self.config.model_type.value
        
        request = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "stream": stream
        }
        
        if tools:
            request["tools"] = tools
        
        if response_format:
            request["response_format"] = response_format
        
        if stream:
            return self._stream_chat(request)
        else:
            response = self.client.completions.create(**request)
            return self._parse_response(response)
    
    def _stream_chat(self, request: Dict) -> AsyncIterator[str]:
        """Stream chat responses"""
        stream = self.client.completions.create(**request)
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    async def async_chat_completion(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> Union[Dict, AsyncIterator]:
        """Async chat completion"""
        model = kwargs.get("model", self.config.model_type.value)
        
        response = await self.async_client.completions.create(
            model=model,
            messages=messages,
            **kwargs
        )
        
        if kwargs.get("stream", False):
            return self._async_stream_chat(response)
        else:
            return self._parse_response(response)
    
    async def _async_stream_chat(self, stream) -> AsyncIterator[str]:
        """Async stream handler"""
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    def _parse_response(self, response) -> Dict[str, Any]:
        """Parse API response"""
        return {
            "id": response.id,
            "model": response.model,
            "content": response.choices[0].message.content,
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            },
            "finish_reason": response.choices[0].finish_reason
        }
    
    # Text Completions
    def text_completion(
        self,
        prompt: str,
        model: Optional[str] = None,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None,
        stream: bool = False
    ) -> Union[str, AsyncIterator[str]]:
        """Generate text completion"""
        
        model = model or self.config.model_type.value
        
        response = self.client.completions.create(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=stop,
            stream=stream
        )
        
        if stream:
            return self._stream_text(response)
        else:
            return response.choices[0].text
    
    def _stream_text(self, stream) -> AsyncIterator[str]:
        """Stream text completions"""
        for chunk in stream:
            if chunk.choices and chunk.choices[0].text:
                yield chunk.choices[0].text
    
    # Embeddings
    def create_embeddings(
        self,
        input_texts: Union[str, List[str]],
        model: str = "embed-multilingual-v3.0"
    ) -> List[List[float]]:
        """Create embeddings for texts"""
        
        if isinstance(input_texts, str):
            input_texts = [input_texts]
        
        response = self.client.embeddings.create(
            model=model,
            input=input_texts
        )
        
        return [item.embedding for item in response.data]

# Dedicated Endpoint Management
class DedicatedEndpointManager:
    """Manage dedicated FriendliAI endpoints"""
    
    def __init__(self, client: FriendliClient):
        self.client = client
        self.endpoints = {}
    
    async def create_endpoint(
        self,
        name: str,
        model: ModelType,
        gpu_type: str = "A100",
        min_replicas: int = 1,
        max_replicas: int = 10,
        quantization: Optional[str] = "FP8",
        auto_scaling: bool = True
    ) -> Dict[str, Any]:
        """Create dedicated endpoint"""
        
        config = DedicatedEndpointConfig(
            name=name,
            model=model.value,
            gpu_type=gpu_type,
            min_replicas=min_replicas,
            max_replicas=max_replicas,
            quantization=QuantizationConfig(type=quantization) if quantization else None,
            auto_scaling=auto_scaling,
            scaling_metric="requests_per_second",
            target_value=100
        )
        
        # API call to create endpoint
        endpoint = await self.client.async_client.endpoints.create(config.dict())
        self.endpoints[name] = endpoint
        
        return endpoint
    
    async def update_endpoint(
        self,
        endpoint_id: str,
        **updates
    ) -> Dict[str, Any]:
        """Update endpoint configuration"""
        
        endpoint = await self.client.async_client.endpoints.update(
            endpoint_id,
            **updates
        )
        
        return endpoint
    
    async def scale_endpoint(
        self,
        endpoint_id: str,
        replicas: int
    ) -> Dict[str, Any]:
        """Scale endpoint replicas"""
        
        return await self.update_endpoint(
            endpoint_id,
            min_replicas=replicas,
            max_replicas=replicas
        )
    
    async def delete_endpoint(self, endpoint_id: str) -> bool:
        """Delete endpoint"""
        
        await self.client.async_client.endpoints.delete(endpoint_id)
        return True
    
    async def list_endpoints(self) -> List[Dict[str, Any]]:
        """List all endpoints"""
        
        endpoints = await self.client.async_client.endpoints.list()
        return endpoints

# Fine-tuning and LoRA
class FineTuningManager:
    """Manage fine-tuning and LoRA adapters"""
    
    def __init__(self, client: FriendliClient):
        self.client = client
        self.jobs = {}
        self.adapters = {}
    
    async def create_fine_tuning_job(
        self,
        base_model: ModelType,
        training_file: str,
        validation_file: Optional[str] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        wandb_project: Optional[str] = None
    ) -> Dict[str, Any]:
        """Create fine-tuning job"""
        
        default_hyperparameters = {
            "learning_rate": 1e-4,
            "batch_size": 8,
            "num_epochs": 3,
            "warmup_steps": 100,
            "gradient_accumulation_steps": 4
        }
        
        if hyperparameters:
            default_hyperparameters.update(hyperparameters)
        
        job_config = {
            "base_model": base_model.value,
            "training_file": training_file,
            "validation_file": validation_file,
            "hyperparameters": default_hyperparameters,
            "wandb_project": wandb_project
        }
        
        job = await self.client.async_client.fine_tuning.create(**job_config)
        self.jobs[job["id"]] = job
        
        return job
    
    async def create_lora_adapter(
        self,
        name: str,
        base_model: ModelType,
        weights_path: str,
        rank: int = 16,
        alpha: int = 32,
        dropout: float = 0.05
    ) -> Dict[str, Any]:
        """Create LoRA adapter"""
        
        adapter_config = LoRAConfig(
            name=name,
            base_model=base_model.value,
            weights_path=weights_path,
            rank=rank,
            alpha=alpha,
            dropout=dropout
        )
        
        adapter = await self.client.async_client.adapters.create(adapter_config.dict())
        self.adapters[name] = adapter
        
        return adapter
    
    async def deploy_multi_lora(
        self,
        endpoint_id: str,
        adapters: List[str]
    ) -> Dict[str, Any]:
        """Deploy multiple LoRA adapters to endpoint"""
        
        config = {
            "adapters": adapters,
            "routing_strategy": "dynamic"
        }
        
        return await self.client.async_client.endpoints.update_adapters(
            endpoint_id,
            config
        )

# Container Deployment
class ContainerManager:
    """Manage Friendli Container deployments"""
    
    def __init__(self, client: FriendliClient):
        self.client = client
    
    def generate_container_config(
        self,
        model: ModelType,
        quantization: str = "AWQ",
        gpu_type: str = "A100",
        num_gpus: int = 1,
        port: int = 8000
    ) -> Dict[str, Any]:
        """Generate container configuration"""
        
        config = {
            "image": "friendliai/friendli-container:latest",
            "model": model.value,
            "quantization": quantization,
            "environment": {
                "FRIENDLI_MODEL_ID": model.value,
                "FRIENDLI_QUANTIZATION": quantization,
                "FRIENDLI_GPU_TYPE": gpu_type,
                "FRIENDLI_NUM_GPUS": str(num_gpus),
                "FRIENDLI_PORT": str(port),
                "FRIENDLI_MAX_BATCH_SIZE": "32",
                "FRIENDLI_MAX_SEQ_LEN": "4096"
            },
            "resources": {
                "gpu": {
                    "type": gpu_type,
                    "count": num_gpus
                },
                "memory": "32Gi",
                "cpu": "8"
            }
        }
        
        return config
    
    def generate_kubernetes_manifest(
        self,
        name: str,
        config: Dict[str, Any]
    ) -> str:
        """Generate Kubernetes deployment manifest"""
        
        manifest = f"""
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {name}-friendli
spec:
  replicas: 1
  selector:
    matchLabels:
      app: {name}-friendli
  template:
    metadata:
      labels:
        app: {name}-friendli
    spec:
      containers:
      - name: friendli-container
        image: {config['image']}
        ports:
        - containerPort: {config['environment']['FRIENDLI_PORT']}
        env:
"""
        
        for key, value in config['environment'].items():
            manifest += f"""
        - name: {key}
          value: "{value}"
"""
        
        manifest += f"""
        resources:
          limits:
            nvidia.com/gpu: {config['resources']['gpu']['count']}
            memory: {config['resources']['memory']}
            cpu: {config['resources']['cpu']}
---
apiVersion: v1
kind: Service
metadata:
  name: {name}-friendli-service
spec:
  selector:
    app: {name}-friendli
  ports:
  - port: 80
    targetPort: {config['environment']['FRIENDLI_PORT']}
  type: LoadBalancer
"""
        
        return manifest

# Advanced Features
class AdvancedFriendliClient(FriendliClient):
    """Advanced FriendliAI features"""
    
    def __init__(self, config: FriendliConfig):
        super().__init__(config)
        self.cache = {}
        self.metrics = {
            "total_requests": 0,
            "total_tokens": 0,
            "cache_hits": 0
        }
    
    async def cached_completion(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> Dict[str, Any]:
        """Cached completion with deduplication"""
        
        # Generate cache key
        cache_key = self._generate_cache_key(messages, kwargs)
        
        # Check cache
        if cache_key in self.cache:
            self.metrics["cache_hits"] += 1
            return self.cache[cache_key]
        
        # Generate completion
        response = await self.async_chat_completion(messages, **kwargs)
        
        # Update metrics
        self.metrics["total_requests"] += 1
        self.metrics["total_tokens"] += response["usage"]["total_tokens"]
        
        # Cache response
        self.cache[cache_key] = response
        
        return response
    
    def _generate_cache_key(
        self,
        messages: List[Dict[str, str]],
        kwargs: Dict[str, Any]
    ) -> str:
        """Generate cache key for request"""
        import hashlib
        
        key_data = {
            "messages": messages,
            "params": kwargs
        }
        
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()
    
    async def batch_completion(
        self,
        batch_messages: List[List[Dict[str, str]]],
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Process batch of completions"""
        
        tasks = []
        for messages in batch_messages:
            task = self.async_chat_completion(messages, **kwargs)
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks)
        return responses
    
    async def structured_output(
        self,
        messages: List[Dict[str, str]],
        schema: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """Generate structured output with JSON schema"""
        
        response_format = {
            "type": "json_schema",
            "json_schema": schema
        }
        
        response = await self.async_chat_completion(
            messages,
            response_format=response_format,
            **kwargs
        )
        
        # Parse and validate JSON
        try:
            structured_data = json.loads(response["content"])
            return structured_data
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse structured output: {e}")
            return {}
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get usage metrics"""
        return self.metrics

# Integration Examples
class FriendliLangChainIntegration:
    """LangChain integration for FriendliAI"""
    
    @staticmethod
    def create_llm(token: str, model: str = "meta-llama-3.1-70b-instruct"):
        """Create LangChain LLM"""
        from langchain_community.llms import FriendliAI
        
        llm = FriendliAI(
            friendli_token=token,
            model=model,
            temperature=0.7,
            max_tokens=2000
        )
        
        return llm
    
    @staticmethod
    def create_chat_model(token: str, model: str = "meta-llama-3.1-70b-instruct"):
        """Create LangChain chat model"""
        from langchain_community.chat_models import ChatFriendliAI
        
        chat_model = ChatFriendliAI(
            friendli_token=token,
            model=model,
            temperature=0.7
        )
        
        return chat_model

# Example Usage
async def main():
    # Initialize client
    config = FriendliConfig(
        token="your-token",
        deployment_type=DeploymentType.SERVERLESS,
        model_type=ModelType.LLAMA_3_1_70B
    )
    
    client = AdvancedFriendliClient(config)
    
    # Chat completion
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain quantum computing"}
    ]
    
    response = await client.async_chat_completion(messages)
    print("Response:", response["content"])
    
    # Streaming
    stream = await client.async_chat_completion(messages, stream=True)
    async for chunk in stream:
        print(chunk, end="")
    
    # Structured output
    schema = {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "key_points": {
                "type": "array",
                "items": {"type": "string"}
            },
            "difficulty": {
                "type": "string",
                "enum": ["easy", "medium", "hard"]
            }
        },
        "required": ["summary", "key_points", "difficulty"]
    }
    
    structured = await client.structured_output(messages, schema)
    print("Structured:", structured)
    
    # Batch processing
    batch = [messages] * 5
    batch_responses = await client.batch_completion(batch)
    print(f"Processed {len(batch_responses)} requests")
    
    # Metrics
    print("Metrics:", client.get_metrics())

if __name__ == "__main__":
    asyncio.run(main())
```

## FriendliAI JavaScript SDK

```typescript
// friendli-client.ts
import { FriendliAI } from '@friendliai/ai-provider';
import { generateText, streamText, generateObject } from 'ai';
import { z } from 'zod';

interface FriendliConfig {
  token: string;
  baseUrl?: string;
  model?: string;
  temperature?: number;
  maxTokens?: number;
}

export class FriendliClient {
  private client: FriendliAI;
  private config: FriendliConfig;

  constructor(config: FriendliConfig) {
    this.config = config;
    this.client = new FriendliAI({
      apiKey: config.token,
      baseURL: config.baseUrl || 'https://api.friendli.ai/v1'
    });
  }

  async generateText(prompt: string, options: any = {}) {
    const result = await generateText({
      model: this.client(this.config.model || 'meta-llama-3.1-70b-instruct'),
      prompt,
      temperature: options.temperature || this.config.temperature,
      maxTokens: options.maxTokens || this.config.maxTokens
    });
    
    return result.text;
  }

  async streamText(prompt: string, onChunk: (chunk: string) => void) {
    const stream = await streamText({
      model: this.client(this.config.model || 'meta-llama-3.1-70b-instruct'),
      prompt
    });

    for await (const chunk of stream.textStream) {
      onChunk(chunk);
    }
  }

  async generateStructured<T>(
    prompt: string,
    schema: z.ZodType<T>
  ): Promise<T> {
    const result = await generateObject({
      model: this.client(this.config.model || 'meta-llama-3.1-70b-instruct'),
      prompt,
      schema
    });

    return result.object;
  }
}
```

---

# Soveren Data Governance - Complete Documentation

## Table of Contents
1. [Overview](#soveren-overview)
2. [Installation](#soveren-installation)
3. [Sensor Configuration](#sensor-configuration)
4. [API Implementation](#soveren-api)
5. [Data Discovery](#data-discovery)
6. [Compliance Management](#compliance-management)
7. [Integration Examples](#soveren-integrations)
8. [Best Practices](#soveren-best-practices)

## Soveren Overview

**Soveren** is a real-time Data Security Posture Management (DSPM) and Data Detection & Response (DDR) platform for Kubernetes environments with 98% detection accuracy.

### Key Features
- **Real-time Discovery**: Automatic detection of sensitive data
- **98% Accuracy**: ML-powered classification
- **Zero Performance Impact**: eBPF-based monitoring
- **Compliance Ready**: GDPR, CCPA, PCI DSS, SOC-2
- **45+ Data Types**: Across 30+ countries

## Soveren Installation

### Helm Installation

```bash
# Add Soveren repository
helm repo add soveren https://soverenio.github.io/helm-charts
helm repo update

# Create namespaces
kubectl create namespace soverenio-dim-sensor
kubectl create namespace soverenio-dar-sensor

# Install DIM (Data-in-Motion) Sensor
helm install -n soverenio-dim-sensor soveren-dim-sensor \
  soveren/soveren-dim-sensor \
  --set sensor.token="<YOUR_TOKEN>" \
  --set sensor.collectorUrl="https://api.soveren.io" \
  --create-namespace

# Install DAR (Data-at-Rest) Sensor
helm install -n soverenio-dar-sensor soveren-dar-sensor \
  soveren/soveren-dar-sensor \
  --set crawler.token="<YOUR_TOKEN>" \
  --create-namespace
```

### Advanced Configuration

```yaml
# values.yaml
sensor:
  token: "<YOUR_TOKEN>"
  collectorUrl: "https://api.soveren.io"
  
  # Performance tuning
  resources:
    limits:
      memory: "2Gi"
      cpu: "1000m"
    requests:
      memory: "512Mi"
      cpu: "250m"
  
  # Node selection
  nodeSelector:
    node-role.kubernetes.io/monitoring: "true"
  
  # Security context
  securityContext:
    privileged: true
    capabilities:
      add:
        - SYS_ADMIN
        - SYS_RESOURCE
        - NET_ADMIN
        - NET_RAW
  
  # Data sampling
  sampling:
    rate: 1.0  # 100% sampling
    maxEventsPerSecond: 10000
  
  # Filtering
  filters:
    includeNamespaces:
      - production
      - staging
    excludeNamespaces:
      - kube-system
      - test
    includeServices:
      - api-gateway
      - user-service
    excludeServices:
      - monitoring
      - logging

crawler:
  token: "<YOUR_TOKEN>"
  schedule: "0 2 * * *"  # Daily at 2 AM
  
  # Database connections
  databases:
    - name: postgres-prod
      type: postgresql
      host: postgres.example.com
      port: 5432
      database: production
      credentials:
        secretName: postgres-credentials
    
    - name: mysql-prod
      type: mysql
      host: mysql.example.com
      port: 3306
      database: production
      credentials:
        secretName: mysql-credentials
  
  # S3 buckets
  s3Buckets:
    - name: prod-data
      region: us-east-1
      credentials:
        secretName: aws-credentials
  
  # Kafka topics
  kafkaTopics:
    - name: events
      bootstrapServers: kafka:9092
      credentials:
        secretName: kafka-credentials
```

## Sensor Configuration

### Complete Sensor Implementation

```python
# soveren_client.py
import requests
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataType(Enum):
    """Sensitive data types"""
    CREDIT_CARD = "credit_card"
    SSN = "ssn"
    EMAIL = "email"
    PHONE = "phone"
    PASSPORT = "passport"
    DRIVER_LICENSE = "driver_license"
    BANK_ACCOUNT = "bank_account"
    API_KEY = "api_key"
    PASSWORD = "password"
    CUSTOM = "custom"

class Severity(Enum):
    """Data sensitivity levels"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class SoverenConfig:
    """Soveren configuration"""
    api_token: str
    base_url: str = "https://api.soveren.io"
    timeout: int = 30

class SoverenClient:
    """Soveren API client"""
    
    def __init__(self, config: SoverenConfig):
        self.config = config
        self.headers = {
            "Authorization": f"Bearer {config.api_token}",
            "Content-Type": "application/json"
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)
    
    # Data Discovery
    def get_data_types(self) -> List[Dict[str, Any]]:
        """Get supported data types"""
        response = self.session.get(
            f"{self.config.base_url}/api/v1/data-types",
            timeout=self.config.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def get_assets(self, filters: Optional[Dict] = None) -> List[Dict[str, Any]]:
        """Get discovered infrastructure assets"""
        params = filters or {}
        response = self.session.get(
            f"{self.config.base_url}/api/v1/assets",
            params=params,
            timeout=self.config.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def get_data_flows(self, asset_id: str) -> List[Dict[str, Any]]:
        """Get data flows for an asset"""
        response = self.session.get(
            f"{self.config.base_url}/api/v1/assets/{asset_id}/data-flows",
            timeout=self.config.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def get_sensitive_data(
        self,
        asset_id: Optional[str] = None,
        data_type: Optional[DataType] = None,
        severity: Optional[Severity] = None
    ) -> List[Dict[str, Any]]:
        """Get detected sensitive data"""
        params = {}
        if asset_id:
            params["assetId"] = asset_id
        if data_type:
            params["dataType"] = data_type.value
        if severity:
            params["severity"] = severity.value
        
        response = self.session.get(
            f"{self.config.base_url}/api/v1/sensitive-data",
            params=params,
            timeout=self.config.timeout
        )
        response.raise_for_status()
        return response.json()
    
    # Compliance
    def get_compliance_status(self, framework: str = "GDPR") -> Dict[str, Any]:
        """Get compliance status for framework"""
        response = self.session.get(
            f"{self.config.base_url}/api/v1/compliance/{framework}",
            timeout=self.config.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def get_violations(
        self,
        framework: Optional[str] = None,
        severity: Optional[Severity] = None
    ) -> List[Dict[str, Any]]:
        """Get compliance violations"""
        params = {}
        if framework:
            params["framework"] = framework
        if severity:
            params["severity"] = severity.value
        
        response = self.session.get(
            f"{self.config.base_url}/api/v1/violations",
            params=params,
            timeout=self.config.timeout
        )
        response.raise_for_status()
        return response.json()
    
    # Policies
    def create_policy(
        self,
        name: str,
        description: str,
        rules: List[Dict[str, Any]],
        enabled: bool = True
    ) -> Dict[str, Any]:
        """Create detection policy"""
        policy = {
            "name": name,
            "description": description,
            "rules": rules,
            "enabled": enabled
        }
        
        response = self.session.post(
            f"{self.config.base_url}/api/v1/policies",
            json=policy,
            timeout=self.config.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def update_policy(
        self,
        policy_id: str,
        updates: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Update policy"""
        response = self.session.patch(
            f"{self.config.base_url}/api/v1/policies/{policy_id}",
            json=updates,
            timeout=self.config.timeout
        )
        response.raise_for_status()
        return response.json()
    
    # Alerts
    def get_alerts(
        self,
        status: Optional[str] = None,
        severity: Optional[Severity] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get alerts"""
        params = {"limit": limit}
        if status:
            params["status"] = status
        if severity:
            params["severity"] = severity.value
        
        response = self.session.get(
            f"{self.config.base_url}/api/v1/alerts",
            params=params,
            timeout=self.config.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def acknowledge_alert(self, alert_id: str, notes: str = "") -> Dict[str, Any]:
        """Acknowledge an alert"""
        response = self.session.post(
            f"{self.config.base_url}/api/v1/alerts/{alert_id}/acknowledge",
            json={"notes": notes},
            timeout=self.config.timeout
        )
        response.raise_for_status()
        return response.json()
    
    # Reports
    def generate_report(
        self,
        report_type: str = "compliance",
        framework: Optional[str] = None,
        format: str = "pdf"
    ) -> bytes:
        """Generate compliance report"""
        params = {
            "type": report_type,
            "format": format
        }
        if framework:
            params["framework"] = framework
        
        response = self.session.post(
            f"{self.config.base_url}/api/v1/reports/generate",
            params=params,
            timeout=self.config.timeout * 2  # Longer timeout for reports
        )
        response.raise_for_status()
        return response.content

# Advanced Features
class SoverenMonitor:
    """Real-time monitoring with Soveren"""
    
    def __init__(self, client: SoverenClient):
        self.client = client
        self.baseline = {}
        self.thresholds = {
            "high_risk_data": 10,
            "medium_risk_data": 50,
            "low_risk_data": 100
        }
    
    def establish_baseline(self) -> Dict[str, Any]:
        """Establish baseline for sensitive data"""
        assets = self.client.get_assets()
        
        for asset in assets:
            asset_id = asset["id"]
            sensitive_data = self.client.get_sensitive_data(asset_id=asset_id)
            
            self.baseline[asset_id] = {
                "high": len([d for d in sensitive_data if d["severity"] == "high"]),
                "medium": len([d for d in sensitive_data if d["severity"] == "medium"]),
                "low": len([d for d in sensitive_data if d["severity"] == "low"])
            }
        
        return self.baseline
    
    def check_anomalies(self) -> List[Dict[str, Any]]:
        """Check for anomalies in sensitive data"""
        anomalies = []
        assets = self.client.get_assets()
        
        for asset in assets:
            asset_id = asset["id"]
            current_data = self.client.get_sensitive_data(asset_id=asset_id)
            
            current_counts = {
                "high": len([d for d in current_data if d["severity"] == "high"]),
                "medium": len([d for d in current_data if d["severity"] == "medium"]),
                "low": len([d for d in current_data if d["severity"] == "low"])
            }
            
            if asset_id in self.baseline:
                baseline = self.baseline[asset_id]
                
                for severity in ["high", "medium", "low"]:
                    if current_counts[severity] > baseline[severity] * 1.5:
                        anomalies.append({
                            "asset_id": asset_id,
                            "asset_name": asset["name"],
                            "severity": severity,
                            "baseline_count": baseline[severity],
                            "current_count": current_counts[severity],
                            "increase_percentage": (
                                (current_counts[severity] - baseline[severity]) / 
                                baseline[severity] * 100
                            ) if baseline[severity] > 0 else 100
                        })
        
        return anomalies
    
    def generate_alert(self, anomaly: Dict[str, Any]) -> Dict[str, Any]:
        """Generate alert for anomaly"""
        alert = {
            "type": "data_anomaly",
            "severity": anomaly["severity"],
            "asset_id": anomaly["asset_id"],
            "message": f"Anomaly detected: {anomaly['increase_percentage']:.1f}% increase in {anomaly['severity']} risk data",
            "details": anomaly
        }
        
        # Send alert (implementation depends on alerting system)
        logger.warning(f"Alert: {alert['message']}")
        
        return alert

# Custom Classifiers
class CustomClassifier:
    """Create custom data classifiers"""
    
    def __init__(self, client: SoverenClient):
        self.client = client
        self.classifiers = []
    
    def create_classifier(
        self,
        name: str,
        pattern: str,
        data_type: str,
        confidence: float = 0.9,
        description: str = ""
    ) -> Dict[str, Any]:
        """Create custom classifier"""
        classifier = {
            "name": name,
            "pattern": pattern,
            "dataType": data_type,
            "confidence": confidence,
            "description": description,
            "enabled": True
        }
        
        # API call to create classifier
        response = self.client.session.post(
            f"{self.client.config.base_url}/api/v1/classifiers",
            json=classifier,
            timeout=self.client.config.timeout
        )
        response.raise_for_status()
        
        result = response.json()
        self.classifiers.append(result)
        
        return result
    
    def create_regex_classifier(
        self,
        name: str,
        regex: str,
        data_type: str,
        examples: List[str] = None
    ) -> Dict[str, Any]:
        """Create regex-based classifier"""
        return self.create_classifier(
            name=name,
            pattern=regex,
            data_type=data_type,
            description=f"Regex classifier: {regex}"
        )
    
    def create_dictionary_classifier(
        self,
        name: str,
        terms: List[str],
        data_type: str
    ) -> Dict[str, Any]:
        """Create dictionary-based classifier"""
        pattern = "|".join(terms)
        return self.create_classifier(
            name=name,
            pattern=pattern,
            data_type=data_type,
            description=f"Dictionary classifier with {len(terms)} terms"
        )
    
    def test_classifier(
        self,
        classifier_id: str,
        test_data: List[str]
    ) -> Dict[str, Any]:
        """Test classifier against sample data"""
        response = self.client.session.post(
            f"{self.client.config.base_url}/api/v1/classifiers/{classifier_id}/test",
            json={"samples": test_data},
            timeout=self.client.config.timeout
        )
        response.raise_for_status()
        return response.json()

# Integration Manager
class SoverenIntegrationManager:
    """Manage Soveren integrations"""
    
    def __init__(self, client: SoverenClient):
        self.client = client
        self.integrations = {}
    
    def setup_slack_integration(
        self,
        webhook_url: str,
        channel: str,
        alert_levels: List[Severity]
    ) -> Dict[str, Any]:
        """Setup Slack integration"""
        integration = {
            "type": "slack",
            "config": {
                "webhookUrl": webhook_url,
                "channel": channel,
                "alertLevels": [level.value for level in alert_levels],
                "enabled": True
            }
        }
        
        response = self.client.session.post(
            f"{self.client.config.base_url}/api/v1/integrations",
            json=integration,
            timeout=self.client.config.timeout
        )
        response.raise_for_status()
        
        result = response.json()
        self.integrations["slack"] = result
        
        return result
    
    def setup_jira_integration(
        self,
        url: str,
        project_key: str,
        api_token: str,
        issue_type: str = "Security Alert"
    ) -> Dict[str, Any]:
        """Setup JIRA integration"""
        integration = {
            "type": "jira",
            "config": {
                "url": url,
                "projectKey": project_key,
                "apiToken": api_token,
                "issueType": issue_type,
                "enabled": True
            }
        }
        
        response = self.client.session.post(
            f"{self.client.config.base_url}/api/v1/integrations",
            json=integration,
            timeout=self.client.config.timeout
        )
        response.raise_for_status()
        
        result = response.json()
        self.integrations["jira"] = result
        
        return result
    
    def setup_webhook_integration(
        self,
        url: str,
        secret: str,
        events: List[str]
    ) -> Dict[str, Any]:
        """Setup generic webhook integration"""
        integration = {
            "type": "webhook",
            "config": {
                "url": url,
                "secret": secret,
                "events": events,
                "enabled": True
            }
        }
        
        response = self.client.session.post(
            f"{self.client.config.base_url}/api/v1/integrations",
            json=integration,
            timeout=self.client.config.timeout
        )
        response.raise_for_status()
        
        result = response.json()
        self.integrations["webhook"] = result
        
        return result

# Compliance Manager
class ComplianceManager:
    """Manage compliance with Soveren"""
    
    def __init__(self, client: SoverenClient):
        self.client = client
        self.frameworks = ["GDPR", "CCPA", "PCI_DSS", "SOC2", "HIPAA"]
    
    def get_compliance_dashboard(self) -> Dict[str, Any]:
        """Get compliance dashboard data"""
        dashboard = {}
        
        for framework in self.frameworks:
            try:
                status = self.client.get_compliance_status(framework)
                violations = self.client.get_violations(framework=framework)
                
                dashboard[framework] = {
                    "status": status,
                    "violations": violations,
                    "score": status.get("complianceScore", 0),
                    "critical_issues": len([v for v in violations if v["severity"] == "high"])
                }
            except Exception as e:
                logger.error(f"Failed to get compliance data for {framework}: {e}")
                dashboard[framework] = {"error": str(e)}
        
        return dashboard
    
    def generate_compliance_report(
        self,
        framework: str,
        format: str = "pdf"
    ) -> bytes:
        """Generate compliance report"""
        return self.client.generate_report(
            report_type="compliance",
            framework=framework,
            format=format
        )
    
    def create_gdpr_policy(self) -> Dict[str, Any]:
        """Create GDPR compliance policy"""
        gdpr_rules = [
            {
                "name": "detect_eu_personal_data",
                "description": "Detect EU personal data",
                "dataTypes": ["email", "phone", "address", "name"],
                "action": "alert",
                "severity": "high"
            },
            {
                "name": "cross_border_transfer",
                "description": "Detect cross-border data transfers",
                "condition": "data.location != 'EU'",
                "action": "block",
                "severity": "critical"
            },
            {
                "name": "data_retention",
                "description": "Check data retention limits",
                "retentionDays": 730,
                "action": "alert",
                "severity": "medium"
            }
        ]
        
        return self.client.create_policy(
            name="GDPR Compliance Policy",
            description="Policy for GDPR compliance",
            rules=gdpr_rules,
            enabled=True
        )
    
    def create_pci_dss_policy(self) -> Dict[str, Any]:
        """Create PCI DSS compliance policy"""
        pci_rules = [
            {
                "name": "detect_card_data",
                "description": "Detect credit card data",
                "dataTypes": ["credit_card", "cvv"],
                "action": "alert",
                "severity": "critical"
            },
            {
                "name": "card_data_encryption",
                "description": "Ensure card data is encrypted",
                "condition": "data.encrypted != true",
                "action": "block",
                "severity": "critical"
            },
            {
                "name": "pci_scope_boundary",
                "description": "Alert on card data outside PCI scope",
                "condition": "asset.tags not contains 'pci-scope'",
                "action": "alert",
                "severity": "high"
            }
        ]
        
        return self.client.create_policy(
            name="PCI DSS Compliance Policy",
            description="Policy for PCI DSS compliance",
            rules=pci_rules,
            enabled=True
        )

# Data Governance Workflow
class DataGovernanceWorkflow:
    """Complete data governance workflow"""
    
    def __init__(self, client: SoverenClient):
        self.client = client
        self.monitor = SoverenMonitor(client)
        self.compliance = ComplianceManager(client)
        self.integrations = SoverenIntegrationManager(client)
    
    async def run_daily_scan(self) -> Dict[str, Any]:
        """Run daily governance scan"""
        results = {
            "timestamp": datetime.now().isoformat(),
            "assets_scanned": 0,
            "sensitive_data_found": 0,
            "violations": [],
            "anomalies": [],
            "actions_taken": []
        }
        
        # Scan assets
        assets = self.client.get_assets()
        results["assets_scanned"] = len(assets)
        
        # Check for sensitive data
        for asset in assets:
            sensitive_data = self.client.get_sensitive_data(asset_id=asset["id"])
            results["sensitive_data_found"] += len(sensitive_data)
        
        # Check compliance
        compliance_dashboard = self.compliance.get_compliance_dashboard()
        for framework, data in compliance_dashboard.items():
            if "violations" in data:
                results["violations"].extend(data["violations"])
        
        # Check for anomalies
        anomalies = self.monitor.check_anomalies()
        results["anomalies"] = anomalies
        
        # Generate alerts for critical issues
        for anomaly in anomalies:
            if anomaly["severity"] == "high":
                alert = self.monitor.generate_alert(anomaly)
                results["actions_taken"].append({
                    "type": "alert",
                    "details": alert
                })
        
        # Generate report
        if len(results["violations"]) > 0 or len(results["anomalies"]) > 0:
            report = await self.generate_daily_report(results)
            results["report_generated"] = True
            results["report_path"] = report
        
        return results
    
    async def generate_daily_report(self, scan_results: Dict[str, Any]) -> str:
        """Generate daily governance report"""
        report_content = f"""
        # Data Governance Daily Report
        
        ## Scan Summary
        - Date: {scan_results['timestamp']}
        - Assets Scanned: {scan_results['assets_scanned']}
        - Sensitive Data Items: {scan_results['sensitive_data_found']}
        
        ## Compliance Violations
        - Total Violations: {len(scan_results['violations'])}
        - Critical: {len([v for v in scan_results['violations'] if v.get('severity') == 'high'])}
        - Medium: {len([v for v in scan_results['violations'] if v.get('severity') == 'medium'])}
        - Low: {len([v for v in scan_results['violations'] if v.get('severity') == 'low'])}
        
        ## Anomalies Detected
        - Total Anomalies: {len(scan_results['anomalies'])}
        
        ## Actions Taken
        - Alerts Generated: {len([a for a in scan_results['actions_taken'] if a['type'] == 'alert'])}
        """
        
        # Save report
        report_path = f"governance_report_{datetime.now().strftime('%Y%m%d')}.md"
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        return report_path

# Example Usage
def main():
    # Initialize client
    config = SoverenConfig(
        api_token="your-token",
        base_url="https://api.soveren.io"
    )
    
    client = SoverenClient(config)
    
    # Get discovered assets
    assets = client.get_assets()
    print(f"Found {len(assets)} assets")
    
    # Check for sensitive data
    for asset in assets[:5]:  # First 5 assets
        sensitive_data = client.get_sensitive_data(asset_id=asset["id"])
        print(f"Asset {asset['name']}: {len(sensitive_data)} sensitive data items")
    
    # Check compliance
    gdpr_status = client.get_compliance_status("GDPR")
    print(f"GDPR Compliance Score: {gdpr_status.get('complianceScore', 0)}%")
    
    # Create custom classifier
    classifier_manager = CustomClassifier(client)
    custom_classifier = classifier_manager.create_regex_classifier(
        name="employee_id",
        regex=r"EMP[0-9]{6}",
        data_type="employee_identifier"
    )
    print(f"Created classifier: {custom_classifier['name']}")
    
    # Setup monitoring
    monitor = SoverenMonitor(client)
    baseline = monitor.establish_baseline()
    print(f"Established baseline for {len(baseline)} assets")
    
    # Check for anomalies
    anomalies = monitor.check_anomalies()
    if anomalies:
        print(f"Found {len(anomalies)} anomalies")
        for anomaly in anomalies:
            print(f"  - {anomaly['asset_name']}: {anomaly['increase_percentage']:.1f}% increase")
    
    # Setup integrations
    integrations = SoverenIntegrationManager(client)
    
    # Setup Slack alerts
    slack_integration = integrations.setup_slack_integration(
        webhook_url="https://hooks.slack.com/services/...",
        channel="#security-alerts",
        alert_levels=[Severity.HIGH, Severity.MEDIUM]
    )
    print(f"Slack integration configured: {slack_integration['id']}")
    
    # Create compliance policies
    compliance_manager = ComplianceManager(client)
    
    gdpr_policy = compliance_manager.create_gdpr_policy()
    print(f"Created GDPR policy: {gdpr_policy['name']}")
    
    pci_policy = compliance_manager.create_pci_dss_policy()
    print(f"Created PCI DSS policy: {pci_policy['name']}")
    
    # Generate compliance report
    report = compliance_manager.generate_compliance_report("GDPR", format="pdf")
    with open("gdpr_compliance_report.pdf", "wb") as f:
        f.write(report)
    print("Generated GDPR compliance report")

if __name__ == "__main__":
    main()
```

## Soveren JavaScript/TypeScript Implementation

```typescript
// soveren-client.ts
import axios, { AxiosInstance } from 'axios';

interface SoverenConfig {
  apiToken: string;
  baseUrl?: string;
  timeout?: number;
}

interface Asset {
  id: string;
  name: string;
  type: string;
  namespace: string;
  labels: Record<string, string>;
  dataTypes: string[];
}

interface SensitiveData {
  id: string;
  assetId: string;
  dataType: string;
  severity: 'high' | 'medium' | 'low';
  location: string;
  count: number;
}

interface ComplianceStatus {
  framework: string;
  complianceScore: number;
  violations: number;
  lastChecked: string;
}

export class SoverenClient {
  private client: AxiosInstance;

  constructor(config: SoverenConfig) {
    this.client = axios.create({
      baseURL: config.baseUrl || 'https://api.soveren.io',
      timeout: config.timeout || 30000,
      headers: {
        'Authorization': `Bearer ${config.apiToken}`,
        'Content-Type': 'application/json'
      }
    });
  }

  // Data Discovery
  async getAssets(filters?: Record<string, any>): Promise<Asset[]> {
    const response = await this.client.get('/api/v1/assets', { params: filters });
    return response.data;
  }

  async getSensitiveData(
    assetId?: string,
    dataType?: string
  ): Promise<SensitiveData[]> {
    const params: any = {};
    if (assetId) params.assetId = assetId;
    if (dataType) params.dataType = dataType;

    const response = await this.client.get('/api/v1/sensitive-data', { params });
    return response.data;
  }

  async getDataFlows(assetId: string): Promise<any[]> {
    const response = await this.client.get(`/api/v1/assets/${assetId}/data-flows`);
    return response.data;
  }

  // Compliance
  async getComplianceStatus(framework: string): Promise<ComplianceStatus> {
    const response = await this.client.get(`/api/v1/compliance/${framework}`);
    return response.data;
  }

  async getViolations(framework?: string): Promise<any[]> {
    const params = framework ? { framework } : {};
    const response = await this.client.get('/api/v1/violations', { params });
    return response.data;
  }

  // Policies
  async createPolicy(policy: any): Promise<any> {
    const response = await this.client.post('/api/v1/policies', policy);
    return response.data;
  }

  async updatePolicy(policyId: string, updates: any): Promise<any> {
    const response = await this.client.patch(`/api/v1/policies/${policyId}`, updates);
    return response.data;
  }

  // Alerts
  async getAlerts(status?: string, severity?: string): Promise<any[]> {
    const params: any = {};
    if (status) params.status = status;
    if (severity) params.severity = severity;

    const response = await this.client.get('/api/v1/alerts', { params });
    return response.data;
  }

  async acknowledgeAlert(alertId: string, notes: string): Promise<any> {
    const response = await this.client.post(
      `/api/v1/alerts/${alertId}/acknowledge`,
      { notes }
    );
    return response.data;
  }
}

// Monitoring Service
export class SoverenMonitor {
  private client: SoverenClient;
  private baseline: Map<string, any>;

  constructor(client: SoverenClient) {
    this.client = client;
    this.baseline = new Map();
  }

  async establishBaseline(): Promise<void> {
    const assets = await this.client.getAssets();
    
    for (const asset of assets) {
      const sensitiveData = await this.client.getSensitiveData(asset.id);
      
      this.baseline.set(asset.id, {
        high: sensitiveData.filter(d => d.severity === 'high').length,
        medium: sensitiveData.filter(d => d.severity === 'medium').length,
        low: sensitiveData.filter(d => d.severity === 'low').length
      });
    }
  }

  async checkAnomalies(): Promise<any[]> {
    const anomalies = [];
    const assets = await this.client.getAssets();

    for (const asset of assets) {
      const current = await this.client.getSensitiveData(asset.id);
      const baseline = this.baseline.get(asset.id);

      if (!baseline) continue;

      const currentCounts = {
        high: current.filter(d => d.severity === 'high').length,
        medium: current.filter(d => d.severity === 'medium').length,
        low: current.filter(d => d.severity === 'low').length
      };

      for (const severity of ['high', 'medium', 'low']) {
        if (currentCounts[severity] > baseline[severity] * 1.5) {
          anomalies.push({
            assetId: asset.id,
            assetName: asset.name,
            severity,
            baselineCount: baseline[severity],
            currentCount: currentCounts[severity],
            increasePercentage: baseline[severity] > 0
              ? ((currentCounts[severity] - baseline[severity]) / baseline[severity] * 100)
              : 100
          });
        }
      }
    }

    return anomalies;
  }
}
```

## Best Practices

### FriendliAI Best Practices

1. **Optimization Configuration**
```python
# Optimal settings for production
config = {
    "quantization": "FP8",  # Balance speed and accuracy
    "batch_size": 32,
    "max_sequence_length": 4096,
    "iteration_batching": True,
    "cache_size": 1000
}
```

2. **Cost Management**
```python
# Implement scale-to-zero
endpoint_config = {
    "auto_scaling": True,
    "min_replicas": 0,  # Scale to zero
    "max_replicas": 10,
    "scale_down_delay": 300  # 5 minutes
}
```

### Soveren Best Practices

1. **Sensor Deployment**
```yaml
# Optimal sensor configuration
nodeSelector:
  node-role.kubernetes.io/monitoring: "true"
tolerations:
  - key: "monitoring"
    operator: "Equal"
    value: "true"
    effect: "NoSchedule"
```

2. **Data Classification**
```python
# Prioritize high-risk data types
priority_data_types = [
    "credit_card",
    "ssn",
    "passport",
    "bank_account",
    "api_key"
]
```

This completes the comprehensive documentation for FriendliAI and Soveren platforms with full Python and JavaScript/TypeScript implementations, configuration examples, and best practices.