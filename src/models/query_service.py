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
    
    def get_retriever(self, search_kwargs: Optional[Dict] = None):
        """
        Get a retriever interface for the vector store.
        
        Args:
            search_kwargs: Optional search parameters (e.g., {'k': 4})
            
        Returns:
            VectorStoreRetriever object or None
        """
        try:
            vectorstore = self.vector_store.get_vectorstore()
            if not vectorstore:
                print("No vector store available to create retriever.")
                return None
            
            search_kwargs = search_kwargs or {'k': 4}
            retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
            print(f"Created retriever with search kwargs: {search_kwargs}")
            return retriever
        except Exception as e:
            print(f"Error creating retriever: {str(e)}")
            return None
    
    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Optional[Dict] = None
    ) -> List[Document]:
        """
        Perform Maximum Marginal Relevance search.
        This optimizes for both similarity to the query and diversity among results.
        
        Args:
            query: Search query string
            k: Number of results to return
            fetch_k: Number of documents to fetch before reranking
            lambda_mult: Balance between relevance (1) and diversity (0)
            filter: Optional metadata filter
            
        Returns:
            List of documents selected by MMR
        """
        try:
            vectorstore = self.vector_store.get_vectorstore()
            if not vectorstore:
                print("No vector store available for search.")
                return []
            
            results = vectorstore.max_marginal_relevance_search(
                query, 
                k=k, 
                fetch_k=fetch_k,
                lambda_mult=lambda_mult,
                filter=filter
            )
            print(f"Found {len(results)} documents using MMR for query: '{query[:50]}...'")
            return results
        except Exception as e:
            print(f"Error performing MMR search: {str(e)}")
            return []
    
    def search_by_vector(
        self,
        embedding: List[float],
        k: int = 4,
        filter: Optional[Dict] = None
    ) -> List[Document]:
        """
        Search using a pre-computed embedding vector.
        
        Args:
            embedding: The embedding vector to search with
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
            
            results = vectorstore.similarity_search_by_vector(
                embedding,
                k=k,
                filter=filter
            )
            print(f"Found {len(results)} similar documents for embedding vector")
            return results
        except Exception as e:
            print(f"Error performing vector search: {str(e)}")
            return []