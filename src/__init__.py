"""
Production RAG System - src package
"""

__version__ = "1.0.0"
__author__ = "RAG Team"

from .pipeline import RAGPipeline, RAGResponse, ConversationManager, ConversationTurn

__all__ = ["RAGPipeline", "RAGResponse", "ConversationManager", "ConversationTurn"]
