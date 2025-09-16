"""
Embedding generation service for Clario
Handles text embedding creation for semantic search
"""

import openai
from typing import List, Dict, Any, Optional, Union
import asyncio
import tiktoken
import logging
from datetime import datetime
import hashlib
import json

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger("embeddings")


class EmbeddingService:
    """Service for generating text embeddings"""
    
    def __init__(self):
        self.openai_client = None
        self.encoding = None
        self.max_tokens = 8191  # OpenAI ada-002 limit
        
    async def initialize(self) -> None:
        """Initialize embedding service"""
        try:
            if settings.OPENAI_API_KEY:
                self.openai_client = openai.AsyncOpenAI(
                    api_key=settings.OPENAI_API_KEY
                )
                self.encoding = tiktoken.encoding_for_model("text-embedding-ada-002")
                logger.info("OpenAI embedding service initialized")
            else:
                logger.warning("No OpenAI API key provided, embeddings will be mocked")
                
        except Exception as e:
            logger.error(f"Failed to initialize embedding service: {e}")
            raise
    
    def _truncate_text(self, text: str, max_tokens: int = None) -> str:
        """Truncate text to fit within token limits"""
        if not self.encoding:
            return text[:4000]  # Fallback character limit
        
        max_tokens = max_tokens or self.max_tokens
        tokens = self.encoding.encode(text)
        
        if len(tokens) <= max_tokens:
            return text
        
        # Truncate tokens and decode back to text
        truncated_tokens = tokens[:max_tokens]
        return self.encoding.decode(truncated_tokens)
    
    async def generate_embedding(
        self,
        text: str,
        model: str = "text-embedding-ada-002"
    ) -> List[float]:
        """Generate embedding for single text"""
        try:
            if not self.openai_client:
                # Return mock embedding for development
                return self._generate_mock_embedding(text)
            
            # Truncate text if needed
            truncated_text = self._truncate_text(text)
            
            response = await self.openai_client.embeddings.create(
                model=model,
                input=truncated_text
            )
            
            embedding = response.data[0].embedding
            logger.debug(f"Generated embedding for text of length {len(text)}")
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            # Return mock embedding as fallback
            return self._generate_mock_embedding(text)
    
    async def generate_embeddings_batch(
        self,
        texts: List[str],
        model: str = "text-embedding-ada-002",
        batch_size: int = 100
    ) -> List[List[float]]:
        """Generate embeddings for multiple texts"""
        try:
            if not self.openai_client:
                return [self._generate_mock_embedding(text) for text in texts]
            
            embeddings = []
            
            # Process in batches
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                truncated_batch = [self._truncate_text(text) for text in batch]
                
                response = await self.openai_client.embeddings.create(
                    model=model,
                    input=truncated_batch
                )
                
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                
                # Small delay to respect rate limits
                if i + batch_size < len(texts):
                    await asyncio.sleep(0.1)
            
            logger.info(f"Generated {len(embeddings)} embeddings")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            return [self._generate_mock_embedding(text) for text in texts]
    
    def _generate_mock_embedding(self, text: str) -> List[float]:
        """Generate mock embedding for development/testing"""
        # Create deterministic but pseudo-random embedding based on text
        hash_obj = hashlib.md5(text.encode())
        seed = int(hash_obj.hexdigest()[:8], 16)
        
        import random
        random.seed(seed)
        
        # Generate 1536-dimensional embedding (OpenAI ada-002 size)
        embedding = [random.uniform(-1, 1) for _ in range(1536)]
        return embedding
    
    async def embed_meeting_content(self, meeting_data: Dict[str, Any]) -> Dict[str, List[float]]:
        """Generate embeddings for all meeting content"""
        try:
            # Combine different parts of meeting for comprehensive embedding
            content_parts = []
            
            # Title and description
            if meeting_data.get("title"):
                content_parts.append(f"Meeting: {meeting_data['title']}")
            
            if meeting_data.get("description"):
                content_parts.append(f"Description: {meeting_data['description']}")
            
            # Agenda content
            if meeting_data.get("agenda_text"):
                content_parts.append(f"Agenda: {meeting_data['agenda_text']}")
            
            # Summary if available
            if meeting_data.get("ai_summary"):
                content_parts.append(f"Summary: {meeting_data['ai_summary']}")
            
            # Create different embedding contexts
            embeddings = {}
            
            # Full content embedding
            full_content = " ".join(content_parts)
            embeddings["full_content"] = await self.generate_embedding(full_content)
            
            # Title embedding for quick matching
            if meeting_data.get("title"):
                embeddings["title"] = await self.generate_embedding(meeting_data["title"])
            
            # Agenda embedding for content-specific search
            if meeting_data.get("agenda_text"):
                embeddings["agenda"] = await self.generate_embedding(meeting_data["agenda_text"])
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to embed meeting content: {e}")
            return {}
    
    async def embed_decision_content(self, decision_data: Dict[str, Any]) -> Dict[str, List[float]]:
        """Generate embeddings for decision content"""
        try:
            content_parts = []
            
            # Decision title and description
            if decision_data.get("title"):
                content_parts.append(f"Decision: {decision_data['title']}")
            
            if decision_data.get("description"):
                content_parts.append(f"Description: {decision_data['description']}")
            
            # Reasoning
            if decision_data.get("reasoning"):
                content_parts.append(f"Reasoning: {decision_data['reasoning']}")
            
            # Options considered
            if decision_data.get("options_considered"):
                options_text = json.dumps(decision_data["options_considered"])
                content_parts.append(f"Options: {options_text}")
            
            embeddings = {}
            
            # Full decision embedding
            full_content = " ".join(content_parts)
            embeddings["full_content"] = await self.generate_embedding(full_content)
            
            # Reasoning-specific embedding
            if decision_data.get("reasoning"):
                embeddings["reasoning"] = await self.generate_embedding(decision_data["reasoning"])
            
            # Title embedding
            if decision_data.get("title"):
                embeddings["title"] = await self.generate_embedding(decision_data["title"])
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to embed decision content: {e}")
            return {}
    
    async def embed_user_profile(self, profile_data: Dict[str, Any]) -> List[float]:
        """Generate embedding for user decision profile"""
        try:
            # Create text representation of user profile
            profile_parts = []
            
            if profile_data.get("decision_style"):
                profile_parts.append(f"Decision style: {profile_data['decision_style']}")
            
            if profile_data.get("communication_style"):
                profile_parts.append(f"Communication style: {profile_data['communication_style']}")
            
            if profile_data.get("risk_tolerance"):
                profile_parts.append(f"Risk tolerance: {profile_data['risk_tolerance']}")
            
            if profile_data.get("primary_values"):
                values = ", ".join(profile_data["primary_values"])
                profile_parts.append(f"Values: {values}")
            
            # Scores
            scores = []
            for score_type in ["analytical_score", "intuitive_score", "collaborative_score"]:
                if profile_data.get(score_type):
                    scores.append(f"{score_type}: {profile_data[score_type]}")
            
            if scores:
                profile_parts.append(f"Scores: {', '.join(scores)}")
            
            profile_text = " ".join(profile_parts)
            return await self.generate_embedding(profile_text)
            
        except Exception as e:
            logger.error(f"Failed to embed user profile: {e}")
            return self._generate_mock_embedding("default_profile")
    
    async def embed_agenda_items(self, agenda_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate embeddings for agenda items"""
        try:
            embedded_items = []
            
            for item in agenda_items:
                content_parts = []
                
                if item.get("title"):
                    content_parts.append(f"Item: {item['title']}")
                
                if item.get("description"):
                    content_parts.append(f"Description: {item['description']}")
                
                item_text = " ".join(content_parts)
                embedding = await self.generate_embedding(item_text)
                
                embedded_items.append({
                    **item,
                    "embedding": embedding,
                    "embedded_text": item_text
                })
            
            return embedded_items
            
        except Exception as e:
            logger.error(f"Failed to embed agenda items: {e}")
            return agenda_items
    
    def calculate_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float]
    ) -> float:
        """Calculate cosine similarity between two embeddings"""
        try:
            import numpy as np
            
            # Convert to numpy arrays
            a = np.array(embedding1)
            b = np.array(embedding2)
            
            # Calculate cosine similarity
            similarity = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Failed to calculate similarity: {e}")
            return 0.0
    
    async def find_similar_content(
        self,
        query_embedding: List[float],
        candidate_embeddings: List[Dict[str, Any]],
        threshold: float = 0.7,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Find similar content based on embeddings"""
        try:
            similarities = []
            
            for candidate in candidate_embeddings:
                if "embedding" in candidate:
                    similarity = self.calculate_similarity(
                        query_embedding,
                        candidate["embedding"]
                    )
                    
                    if similarity >= threshold:
                        similarities.append({
                            **candidate,
                            "similarity_score": similarity
                        })
            
            # Sort by similarity score and limit results
            similarities.sort(key=lambda x: x["similarity_score"], reverse=True)
            return similarities[:limit]
            
        except Exception as e:
            logger.error(f"Failed to find similar content: {e}")
            return []


# Global service instance
embedding_service = None


async def get_embedding_service() -> EmbeddingService:
    """Get embedding service instance"""
    global embedding_service
    if not embedding_service:
        embedding_service = EmbeddingService()
        await embedding_service.initialize()
    return embedding_service
