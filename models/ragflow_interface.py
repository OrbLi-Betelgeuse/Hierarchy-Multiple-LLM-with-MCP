"""
RAGFlow Interface Module

Provides integration with RAGFlow for retrieval-augmented generation capabilities.
Enables document retrieval and context-aware responses.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import requests
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class RAGFlowResponse:
    """Standardized response format for RAGFlow interactions."""

    content: str
    retrieved_documents: List[Dict[str, Any]]
    confidence_score: float
    metadata: Optional[Dict[str, Any]] = None


class RAGFlowInterface:
    """Interface for RAGFlow integration."""

    def __init__(self, base_url: str = "http://localhost:9380", **kwargs):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update(
            {"Content-Type": "application/json", "Accept": "application/json"}
        )

    async def check_connection(self) -> bool:
        """Check if RAGFlow is accessible."""
        try:
            response = await asyncio.to_thread(
                self.session.get, f"{self.base_url}/health", timeout=10
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"RAGFlow connection check failed: {e}")
            return False

    async def create_knowledge_base(
        self, name: str, description: str = ""
    ) -> Dict[str, Any]:
        """Create a new knowledge base in RAGFlow."""
        try:
            payload = {"name": name, "description": description}

            response = await asyncio.to_thread(
                self.session.post,
                f"{self.base_url}/api/v1/knowledge-bases",
                json=payload,
                timeout=30,
            )

            if response.status_code == 201:
                return response.json()
            else:
                logger.error(f"Failed to create knowledge base: {response.text}")
                return {}

        except Exception as e:
            logger.error(f"Error creating knowledge base: {e}")
            return {}

    async def upload_documents(
        self, knowledge_base_id: str, documents: List[Dict[str, Any]]
    ) -> bool:
        """Upload documents to a knowledge base."""
        try:
            payload = {"documents": documents}

            response = await asyncio.to_thread(
                self.session.post,
                f"{self.base_url}/api/v1/knowledge-bases/{knowledge_base_id}/documents",
                json=payload,
                timeout=60,
            )

            if response.status_code == 200:
                logger.info(f"Successfully uploaded {len(documents)} documents")
                return True
            else:
                logger.error(f"Failed to upload documents: {response.text}")
                return False

        except Exception as e:
            logger.error(f"Error uploading documents: {e}")
            return False

    async def query_knowledge_base(
        self,
        knowledge_base_id: str,
        query: str,
        top_k: int = 5,
        similarity_threshold: float = 0.7,
    ) -> RAGFlowResponse:
        """Query a knowledge base for relevant documents."""
        try:
            payload = {
                "query": query,
                "top_k": top_k,
                "similarity_threshold": similarity_threshold,
            }

            response = await asyncio.to_thread(
                self.session.post,
                f"{self.base_url}/api/v1/knowledge-bases/{knowledge_base_id}/query",
                json=payload,
                timeout=30,
            )

            if response.status_code == 200:
                data = response.json()
                return RAGFlowResponse(
                    content=data.get("answer", ""),
                    retrieved_documents=data.get("documents", []),
                    confidence_score=data.get("confidence", 0.0),
                    metadata=data.get("metadata", {}),
                )
            else:
                logger.error(f"Query failed: {response.text}")
                return RAGFlowResponse(
                    content="", retrieved_documents=[], confidence_score=0.0
                )

        except Exception as e:
            logger.error(f"Error querying knowledge base: {e}")
            return RAGFlowResponse(
                content="", retrieved_documents=[], confidence_score=0.0
            )

    async def generate_rag_response(
        self,
        knowledge_base_id: str,
        query: str,
        context: str = "",
        max_tokens: int = 1000,
    ) -> RAGFlowResponse:
        """Generate a response using RAGFlow with retrieval and generation."""
        try:
            # First, retrieve relevant documents
            rag_response = await self.query_knowledge_base(knowledge_base_id, query)

            if not rag_response.retrieved_documents:
                logger.warning("No relevant documents found for query")
                return RAGFlowResponse(
                    content="I don't have enough information to answer this question.",
                    retrieved_documents=[],
                    confidence_score=0.0,
                )

            # Combine retrieved documents with context
            combined_context = (
                context
                + "\n\n"
                + "\n\n".join(
                    [doc.get("content", "") for doc in rag_response.retrieved_documents]
                )
            )

            # Generate response using the combined context
            payload = {
                "query": query,
                "context": combined_context,
                "max_tokens": max_tokens,
                "knowledge_base_id": knowledge_base_id,
            }

            response = await asyncio.to_thread(
                self.session.post,
                f"{self.base_url}/api/v1/generate",
                json=payload,
                timeout=60,
            )

            if response.status_code == 200:
                data = response.json()
                return RAGFlowResponse(
                    content=data.get("response", ""),
                    retrieved_documents=rag_response.retrieved_documents,
                    confidence_score=data.get(
                        "confidence", rag_response.confidence_score
                    ),
                    metadata={
                        "generation_time": data.get("generation_time"),
                        "tokens_used": data.get("tokens_used"),
                        **data.get("metadata", {}),
                    },
                )
            else:
                logger.error(f"Generation failed: {response.text}")
                return rag_response

        except Exception as e:
            logger.error(f"Error generating RAG response: {e}")
            return RAGFlowResponse(
                content="", retrieved_documents=[], confidence_score=0.0
            )

    async def list_knowledge_bases(self) -> List[Dict[str, Any]]:
        """List all available knowledge bases."""
        try:
            response = await asyncio.to_thread(
                self.session.get, f"{self.base_url}/api/v1/knowledge-bases", timeout=10
            )

            if response.status_code == 200:
                return response.json().get("knowledge_bases", [])
            else:
                logger.error(f"Failed to list knowledge bases: {response.text}")
                return []

        except Exception as e:
            logger.error(f"Error listing knowledge bases: {e}")
            return []

    async def delete_knowledge_base(self, knowledge_base_id: str) -> bool:
        """Delete a knowledge base."""
        try:
            response = await asyncio.to_thread(
                self.session.delete,
                f"{self.base_url}/api/v1/knowledge-bases/{knowledge_base_id}",
                timeout=30,
            )

            return response.status_code == 200

        except Exception as e:
            logger.error(f"Error deleting knowledge base: {e}")
            return False


def create_ragflow_interface(
    base_url: str = "http://localhost:9380", **kwargs
) -> RAGFlowInterface:
    """Factory function to create a RAGFlow interface."""
    return RAGFlowInterface(base_url=base_url, **kwargs)
