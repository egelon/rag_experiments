"""
VectorStore Model - Handles FAISS vector database operations.
This class follows the Single Responsibility Principle by only managing
the vector store itself, not the querying logic.
"""
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from typing import List, Optional, Dict
from pathlib import Path


class VectorStore:
    """
    Model class that encapsulates FAISS vector store operations.
    Responsible only for storage, persistence, and basic CRUD operations.
    """
    
    def __init__(self, embedding_model: str = "text-embedding-ada-002"):
        """
        Initialize the VectorStore model.
        
        Args:
            embedding_model (str): OpenAI embedding model to use
        """
        self.embedding_model = embedding_model
        self._vectorstore = None
        self._embeddings = None
        
    def initialize_embeddings(self, api_key: str) -> bool:
        """
        Initialize the OpenAI embeddings with the provided API key.
        
        Args:
            api_key (str): OpenAI API key
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self._embeddings = OpenAIEmbeddings(
                model=self.embedding_model,
                api_key=api_key
            )
            return True
        except Exception as e:
            print(f"Error initializing embeddings: {str(e)}")
            return False
    
    def create_from_texts(self, texts: List[str], metadatas: Optional[List[dict]] = None) -> bool:
        """
        Create a new FAISS vector store from texts.
        
        Args:
            texts: List of texts to index
            metadatas: Optional metadata for each text
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self._embeddings:
                raise ValueError("Embeddings not initialized. Call initialize_embeddings first.")
            
            if not texts:
                print("No texts provided to create vector store")
                return False
            
            self._vectorstore = FAISS.from_texts(
                texts, 
                self._embeddings, 
                metadatas=metadatas
            )
            print(f"Successfully created FAISS vector store with {len(texts)} texts")
            return True
        except Exception as e:
            print(f"Error creating vector store: {str(e)}")
            return False
    
    def add_texts(self, texts: List[str], metadatas: Optional[List[dict]] = None) -> bool:
        """
        Add texts to existing vector store.
        
        Args:
            texts: List of text strings to add
            metadatas: Optional list of metadata dictionaries for each text
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self._vectorstore:
                print("No vector store exists. Create one first.")
                return False
            
            if not texts:
                print("No texts provided to add")
                return False
            
            self._vectorstore.add_texts(texts, metadatas=metadatas)
            print(f"Successfully added {len(texts)} texts to vector store")
            return True
        except Exception as e:
            print(f"Error adding texts: {str(e)}")
            return False
    
    def delete(self, ids: List[str]) -> bool:
        """
        Delete documents by IDs from the vector store.
        
        Args:
            ids: List of document IDs to delete
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self._vectorstore:
                print("No vector store exists.")
                return False
            
            if not ids:
                print("No IDs provided for deletion")
                return False
            
            self._vectorstore.delete(ids)
            print(f"Successfully deleted {len(ids)} documents")
            return True
        except Exception as e:
            print(f"Error deleting documents: {str(e)}")
            return False
    
    def save(self, index_path: str = "faiss_index") -> bool:
        """
        Save the FAISS index to disk.
        
        Args:
            index_path: Path to save the index
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self._vectorstore:
                print("No vector store exists. Create one first.")
                return False
            
            self._vectorstore.save_local(index_path)
            print(f"Successfully saved FAISS index to: {index_path}")
            return True
        except Exception as e:
            print(f"Error saving index: {str(e)}")
            return False
    
    def load(self, index_path: str = "faiss_index", api_key: str = None) -> bool:
        """
        Load a FAISS index from disk.
        
        Args:
            index_path: Path to load the index from
            api_key: OpenAI API key (required if embeddings not initialized)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self._embeddings and api_key:
                self.initialize_embeddings(api_key)
            
            if not self._embeddings:
                raise ValueError("Embeddings not initialized and no API key provided")
            
            if not Path(index_path).exists():
                print(f"Index path does not exist: {index_path}")
                return False
            
            self._vectorstore = FAISS.load_local(
                index_path, 
                self._embeddings, 
                allow_dangerous_deserialization=True
            )
            print(f"Successfully loaded FAISS index from: {index_path}")
            return True
        except Exception as e:
            print(f"Error loading index: {str(e)}")
            return False
    
    def get_stats(self) -> Dict:
        """
        Get statistics about the vector store.
        
        Returns:
            dict: Statistics about the vector store
        """
        try:
            if not self._vectorstore:
                return {"status": "No vector store loaded", "count": 0}
            
            index_count = self._vectorstore.index.ntotal if hasattr(self._vectorstore, 'index') else 0
            
            stats = {
                "status": "Vector store loaded",
                "embedding_model": self.embedding_model,
                "document_count": index_count,
                "embedding_dimension": self._vectorstore.index.d if hasattr(self._vectorstore, 'index') else 0
            }
            
            return stats
        except Exception as e:
            return {"status": f"Error getting stats: {str(e)}", "count": 0}
    
    def is_loaded(self) -> bool:
        """
        Check if a vector store is currently loaded.
        
        Returns:
            bool: True if vector store is loaded, False otherwise
        """
        return self._vectorstore is not None
    
    def get_vectorstore(self):
        """
        Get the underlying FAISS vectorstore object.
        Used by QueryService for search operations.
        
        Returns:
            FAISS vectorstore object or None
        """
        return self._vectorstore