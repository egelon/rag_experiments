"""
RAGController - Main controller for the RAG application.
Coordinates between the View (Streamlit UI) and Models (VectorStore, QueryService).
"""
from typing import List, Optional, Dict, Any
from pathlib import Path
from langchain_community.document_loaders import UnstructuredMarkdownLoader 
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai.chat_models import ChatOpenAI
from langchain_core.documents import Document

from models import VectorStore, QueryService
from config import Config


class RAGController:
    """
    Controller class that manages the business logic and coordinates
    between different components of the RAG application.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the RAG Controller with configuration.
        
        Args:
            config (Config): Application configuration
        """
        self.config = config
        self.vector_store = VectorStore(embedding_model=config.embedding_model)
        self.query_service = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            add_start_index=True,
        )
        self._chat_model = None
    
    def initialize(self) -> tuple[bool, str]:
        """
        Initialize the controller with the API key from config.
        
        Returns:
            tuple: (success, error_message)
        """
        # Validate configuration
        is_valid, error_msg = self.config.validate()
        if not is_valid:
            return False, error_msg
        
        # Initialize embeddings
        if not self.vector_store.initialize_embeddings(self.config.openai_api_key):
            return False, "Failed to initialize embeddings"
        
        # Create query service
        self.query_service = QueryService(self.vector_store)
        
        # Initialize chat model
        try:
            self._chat_model = ChatOpenAI(
                model=self.config.chat_model,
                temperature=self.config.temperature,
                api_key=self.config.openai_api_key
            )
        except Exception as e:
            return False, f"Failed to initialize chat model: {str(e)}"
        
        return True, ""
    
    def load_and_process_documents(self) -> tuple[bool, str]:
        """
        Load documents from processed files directory and create vector index.
        
        Returns:
            tuple: (success, error_message)
        """
        try:
            # Check if directory exists
            if not Path(self.config.processed_files_dir).exists():
                return False, f"Directory '{self.config.processed_files_dir}' not found"
            
            # Load documents
            loader = DirectoryLoader(
                self.config.processed_files_dir, 
                glob="**/*.md",
                loader_cls=UnstructuredMarkdownLoader
            )
            documents = loader.load()
            
            if not documents:
                return False, "No documents found to process"
            
            print(f"Loaded {len(documents)} documents")
            
            # Chunk documents
            chunks = self.text_splitter.split_documents(documents)
            print(f"Split into {len(chunks)} chunks")
            
            # Extract text and metadata
            texts = []
            metadatas = []
            for chunk in chunks:
                texts.append(chunk.page_content)
                metadatas.append(chunk.metadata)
            
            # Create vector store
            if not self.vector_store.create_from_texts(texts, metadatas):
                return False, "Failed to create vector store"
            
            return True, ""
            
        except Exception as e:
            return False, f"Error processing documents: {str(e)}"
    
    def save_index(self) -> tuple[bool, str]:
        """
        Save the vector index to disk.
        
        Returns:
            tuple: (success, error_message)
        """
        if not self.vector_store.is_loaded():
            return False, "No vector store to save"
        
        success = self.vector_store.save(self.config.faiss_index_path)
        if success:
            return True, ""
        else:
            return False, "Failed to save index"
    
    def load_index(self) -> tuple[bool, str]:
        """
        Load a vector index from disk.
        
        Returns:
            tuple: (success, error_message)
        """
        if not self.config.openai_api_key:
            return False, "API key required to load index"
        
        success = self.vector_store.load(
            self.config.faiss_index_path, 
            self.config.openai_api_key
        )
        
        if success:
            # Recreate query service with loaded vector store
            self.query_service = QueryService(self.vector_store)
            return True, ""
        else:
            return False, "Failed to load index"
    
    def search(
        self, 
        query: str, 
        k: Optional[int] = None,
        with_scores: bool = False,
        filter: Optional[Dict] = None
    ) -> List[Any]:
        """
        Perform a search query on the vector store.
        
        Args:
            query: Search query
            k: Number of results (uses config default if None)
            with_scores: Whether to return similarity scores
            filter: Optional metadata filter
            
        Returns:
            List of search results
        """
        if not self.query_service:
            print("Query service not initialized")
            return []
        
        k = k or self.config.default_search_k
        
        if with_scores:
            return self.query_service.similarity_search_with_score(query, k, filter)
        else:
            return self.query_service.similarity_search(query, k, filter)
    
    def search_mmr(
        self,
        query: str,
        k: Optional[int] = None,
        fetch_k: Optional[int] = None,
        lambda_mult: Optional[float] = None,
        filter: Optional[Dict] = None
    ) -> List[Document]:
        """
        Perform Maximum Marginal Relevance search.
        
        Args:
            query: Search query
            k: Number of results
            fetch_k: Number of documents to fetch before reranking
            lambda_mult: Balance between relevance and diversity
            filter: Optional metadata filter
            
        Returns:
            List of documents
        """
        if not self.query_service:
            print("Query service not initialized")
            return []
        
        k = k or self.config.default_search_k
        fetch_k = fetch_k or self.config.default_fetch_k
        lambda_mult = lambda_mult or self.config.default_lambda_mult
        
        return self.query_service.max_marginal_relevance_search(
            query, k, fetch_k, lambda_mult, filter
        )
    
    def get_retriever(self, search_kwargs: Optional[Dict] = None):
        """
        Get a retriever for use with LangChain chains.
        
        Args:
            search_kwargs: Optional search parameters
            
        Returns:
            Retriever object or None
        """
        if not self.query_service:
            print("Query service not initialized")
            return None
        
        search_kwargs = search_kwargs or {'k': self.config.default_search_k}
        return self.query_service.get_retriever(search_kwargs)
    
    def chat(self, message: str, context: Optional[List[Document]] = None) -> str:
        """
        Send a chat message to the language model.
        
        Args:
            message: User message
            context: Optional context documents
            
        Returns:
            Model response
        """
        if not self._chat_model:
            return "Chat model not initialized"
        
        try:
            if context:
                # Format context into the message
                context_text = "\n\n".join([doc.page_content for doc in context])
                full_message = f"Here is additional context:\n{context_text}\n\nThese are known facts, which you must use in order to answer the user's question.\n\nQuestion: {message}"
            else:
                full_message = message
            
            response = self._chat_model.invoke(full_message)
            return response.content
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def rag_query(self, query: str, k: Optional[int] = None) -> tuple[str, List[Document]]:
        """
        Perform a RAG query: search for context and generate response.
        
        Args:
            query: User query
            k: Number of context documents to retrieve
            
        Returns:
            tuple: (response, context_documents)
        """
        # Search for relevant context
        context = self.search(query, k)
        
        # Generate response with context
        response = self.chat(query, context)
        
        return response, context
    
    def get_stats(self) -> Dict:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary with statistics
        """
        return self.vector_store.get_stats()
    
    def update_config(self, **kwargs):
        """
        Update configuration parameters.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        # Reinitialize if API key changed
        if 'openai_api_key' in kwargs:
            self.initialize()