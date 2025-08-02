from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from typing import List, Optional, Tuple, Any
import os
from pathlib import Path


class VectorStoreWrapper:
    def __init__(
        self, 
        embedding_model: str = "text-embedding-ada-002",
        api_key: Optional[str] = None
    ):
        """
        Initialize the VectorStoreWrapper.
        
        Args:
            embedding_model (str): OpenAI embedding model to use
            api_key (Optional[str]): OpenAI API key. If None, tries to get from environment
        """
        self.embedding_model = embedding_model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.vectorstore = None
        self.embeddings = None
        
        if not self.api_key:
            print("Warning: No OpenAI API key provided. Set it before creating embeddings.")
        else:
            self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """Initialize the OpenAI embeddings."""
        try:
            self.embeddings = OpenAIEmbeddings(
                model=self.embedding_model,
                api_key=self.api_key
            )
            print(f"Initialized embeddings with model: {self.embedding_model}")
        except Exception as e:
            print(f"Error initializing embeddings: {str(e)}")
            self.embeddings = None
    
    def create_from_chunks(self, chunks: List[str]) -> bool:
        """
        Create a new FAISS vector store from chunks.
        
        Args:
            chunks: List of chunks to index
        """
        try:
            if not self.embeddings:
                if not self.api_key:
                    raise ValueError("OpenAI API key required to create embeddings")
                self._initialize_embeddings()
            
            if not chunks:
                print("No chunks provided to create vector store")
                return False
            
            self.vectorstore = FAISS.from_texts(chunks, self.embeddings)
            print(f"Successfully created FAISS vector store with {len(chunks)} chunks")
            return True
        except Exception as e:
            print(f"Error creating vector store from chunks: {str(e)}")
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
            if not self.vectorstore:
                print("No vector store exists. Create one first.")
                return False
            
            if not texts:
                print("No texts provided to add")
                return False
            
            self.vectorstore.add_texts(texts, metadatas=metadatas)
            print(f"Successfully added {len(texts)} texts to vector store")
            return True
        except Exception as e:
            print(f"Error adding texts: {str(e)}")
            return False
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 4, 
        filter: Optional[dict] = None
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
            if not self.vectorstore:
                print("No vector store exists. Create one first.")
                return []
            
            results = self.vectorstore.similarity_search(query, k=k, filter=filter)
            print(f"Found {len(results)} similar documents for query: '{query[:50]}...'")
            return results
            
        except Exception as e:
            print(f"Error performing similarity search: {str(e)}")
            return []
    
    def similarity_search_with_score(
        self, 
        query: str, 
        k: int = 4,
        filter: Optional[dict] = None
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
            if not self.vectorstore:
                print("No vector store exists. Create one first.")
                return []
            
            results = self.vectorstore.similarity_search_with_score(query, k=k, filter=filter)
            print(f"Found {len(results)} similar documents with scores for query: '{query[:50]}...'")
            return results
            
        except Exception as e:
            print(f"Error performing similarity search with scores: {str(e)}")
            return []

    
    def save(self, index_path: str = "faiss_index") -> bool:
        """
        Save the FAISS index to disk.
        
        Args:
            index_path: Path to save the index
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.vectorstore:
                print("No vector store exists. Create one first.")
                return False
            
            self.vectorstore.save_local(index_path)
            print(f"Successfully saved FAISS index to: {index_path}")
            return True
            
        except Exception as e:
            print(f"Error saving index: {str(e)}")
            return False
    
    def load(self, index_path: str = "faiss_index") -> bool:
        """
        Load a FAISS index from disk.
        
        Args:
            index_path: Path to load the index from
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.embeddings:
                if not self.api_key:
                    raise ValueError("OpenAI API key required to load embeddings")
                self._initialize_embeddings()
            
            if not Path(index_path).exists():
                print(f"Index path does not exist: {index_path}")
                return False
            
            self.vectorstore = FAISS.load_local(
                index_path, 
                self.embeddings, 
                allow_dangerous_deserialization=True
            )
            print(f"Successfully loaded FAISS index from: {index_path}")
            return True
            
        except Exception as e:
            print(f"Error loading index: {str(e)}")
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
            if not self.vectorstore:
                print("No vector store exists. Create one first.")
                return False
            
            if not ids:
                print("No IDs provided for deletion")
                return False
            
            self.vectorstore.delete(ids)
            print(f"Successfully deleted {len(ids)} documents")
            return True
            
        except Exception as e:
            print(f"Error deleting documents: {str(e)}")
            return False
    
    def get_stats(self) -> dict:
        """
        Get statistics about the vector store.
        
        Returns:
            dict: Statistics about the vector store
        """
        try:
            if not self.vectorstore:
                return {"status": "No vector store loaded", "count": 0}
            
            # Get document count from FAISS index
            index_count = self.vectorstore.index.ntotal if hasattr(self.vectorstore, 'index') else 0
            
            stats = {
                "status": "Vector store loaded",
                "embedding_model": self.embedding_model,
                "document_count": index_count,
                "embedding_dimension": self.vectorstore.index.d if hasattr(self.vectorstore, 'index') else 0
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
        return self.vectorstore is not None
