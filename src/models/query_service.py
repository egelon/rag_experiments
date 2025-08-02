"""
QueryService Model - Handles all vector store query operations.
This class is responsible for searching and retrieving data from the vector store.
"""
from langchain_core.documents import Document
from typing import List, Optional, Tuple, Dict, Any
from .vector_store import VectorStore


class QueryService:
    """
    Service class that handles all query operations on the vector store.
    Separated from VectorStore to follow Single Responsibility Principle.
    """
    
    def __init__(self, vector_store: VectorStore):
        """
        Initialize QueryService with a VectorStore instance.
        
        Args:
            vector_store (VectorStore): The vector store to query against
        """
        self.vector_store = vector_store
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 4, 
        filter: Optional[Dict] = None
    ) -> List[Document]:
        """
        Perform similarity search on the vector store.
        
        Args:
            query: Search query string
            k: Number of results to return
            filter: Optional metadata filter
            
        Returns:
            List of most similar documents
        """
        try:
            vectorstore = self.vector_store.get_vectorstore()
            if not vectorstore:
                print("No vector store available for search.")
                return []
            
            results = vectorstore.similarity_search(query, k=k, filter=filter)
            print(f"Found {len(results)} similar documents for query: '{query[:50]}...'")
            return results
        except Exception as e:
            print(f"Error performing similarity search: {str(e)}")
            return []
    
    def similarity_search_with_score(
        self, 
        query: str, 
        k: int = 4,
        filter: Optional[Dict] = None
    ) -> List[Tuple[Document, float]]:
        """
        Perform similarity search with relevance scores.
        
        Args:
            query: Search query string
            k: Number of results to return
            filter: Optional metadata filter
            
        Returns:
            List of tuples (document, similarity_score)
        """
        try:
            vectorstore = self.vector_store.get_vectorstore()
            if not vectorstore:
                print("No vector store available for search.")
                return []
            
            results = vectorstore.similarity_search_with_score(query, k=k, filter=filter)
            print(f"Found {len(results)} similar documents with scores for query: '{query[:50]}...'")
            return results
        except Exception as e:
            print(f"Error performing similarity search with scores: {str(e)}")
            return []
