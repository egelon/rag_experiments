"""
DocumentProcessor - Refactored to use dependency injection and work with MVC architecture.
This class now focuses on document processing operations without being coupled to storage.
"""
from langchain_community.document_loaders import UnstructuredMarkdownLoader 
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
from typing import List, Optional, Tuple, Protocol
import os


class VectorStoreInterface(Protocol):
    """Interface for vector store operations."""
    def create_from_texts(self, texts: List[str], metadatas: Optional[List[dict]] = None) -> bool: ...
    def save(self, index_path: str) -> bool: ...
    def load(self, index_path: str, api_key: str) -> bool: ...
    def is_loaded(self) -> bool: ...


class DocumentProcessor:
    def __init__(
        self, 
        vector_store: Optional[VectorStoreInterface] = None,
        processed_files_dir: str = "processed_files",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Initialize the DocumentProcessor with dependency injection.
        
        Args:
            vector_store: Optional vector store instance (can be injected later)
            processed_files_dir (str): Directory containing processed markdown files
            chunk_size (int): Size of text chunks for splitting
            chunk_overlap (int): Overlap between chunks
        """
        self.vector_store = vector_store
        self.processed_files_dir = processed_files_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Initialize components
        self.documents = []
        self.chunks = []
        
        # Setup text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            add_start_index=True,
        )
    
    def set_vector_store(self, vector_store: VectorStoreInterface):
        """
        Inject the vector store dependency.
        
        Args:
            vector_store: Vector store instance to use
        """
        self.vector_store = vector_store
    
    def load_documents(self) -> List:
        """
        Load all markdown documents from the processed files directory.
        
        Returns:
            List of loaded documents
        """
        try:
            # Check if directory exists
            if not Path(self.processed_files_dir).exists():
                raise FileNotFoundError(f"Directory '{self.processed_files_dir}' not found")
            
            # Setup document loader
            loader = DirectoryLoader(
                self.processed_files_dir, 
                glob="**/*.md",
                loader_cls=UnstructuredMarkdownLoader
            )
            
            # Load documents
            self.documents = loader.load()
            
            print(f"Loaded {len(self.documents)} documents from {self.processed_files_dir}")
            
            if self.documents:
                total_chars = sum(len(doc.page_content) for doc in self.documents)
                print(f"Total characters across all documents: {total_chars}")
            
            return self.documents
            
        except Exception as e:
            print(f"Error loading documents: {str(e)}")
            return []
    
    def chunk_documents(self, documents: Optional[List] = None) -> List:
        """
        Split documents into chunks.
        
        Args:
            documents: List of documents to chunk. If None, uses self.documents
            
        Returns:
            List of document chunks
        """
        try:
            docs_to_chunk = documents if documents is not None else self.documents
            
            if not docs_to_chunk:
                print("No documents to chunk. Load documents first.")
                return []
            
            # Split documents into chunks
            self.chunks = self.text_splitter.split_documents(docs_to_chunk)
            
            print(f"Split {len(docs_to_chunk)} documents into {len(self.chunks)} chunks")
            
            return self.chunks
            
        except Exception as e:
            print(f"Error chunking documents: {str(e)}")
            return []
    
    def create_index(self, chunks: Optional[List] = None) -> Tuple[bool, str]:
        """
        Create a FAISS vector index from document chunks.
        
        Args:
            chunks: List of document chunks. If None, uses self.chunks
            
        Returns:
            Tuple[bool, str]: (success, error_message)
        """
        try:
            if not self.vector_store:
                return False, "No vector store configured. Call set_vector_store first."
            
            chunks_to_index = chunks if chunks is not None else self.chunks
            
            if not chunks_to_index:
                return False, "No chunks to index. Load and chunk documents first."
            
            # Extract text content and metadata from document chunks
            chunk_texts = []
            metadatas = []
            
            for chunk in chunks_to_index:
                if hasattr(chunk, 'page_content'):
                    chunk_texts.append(chunk.page_content)
                    metadatas.append(chunk.metadata if hasattr(chunk, 'metadata') else {})
                else:
                    chunk_texts.append(str(chunk))
                    metadatas.append({})
            
            # Create FAISS vectorstore from chunks
            success = self.vector_store.create_from_texts(chunk_texts, metadatas)
            
            if success:
                print(f"Successfully created FAISS index with {len(chunks_to_index)} chunks")
                return True, ""
            else:
                return False, "Failed to create vector store"
            
        except Exception as e:
            return False, f"Error creating index: {str(e)}"
    
    def process_all(self) -> Tuple[bool, str]:
        """
        Complete pipeline: load documents, chunk them, and create index.
        
        Returns:
            Tuple[bool, str]: (success, error_message)
        """
        print("Starting document processing pipeline...")
        
        # Load documents
        documents = self.load_documents()
        if not documents:
            return False, "Failed to load documents"
        
        # Chunk documents
        chunks = self.chunk_documents()
        if not chunks:
            return False, "Failed to chunk documents"
        
        # Create index
        success, error_msg = self.create_index()
        if not success:
            return False, error_msg
        
        print("Document processing pipeline completed!")
        return True, ""
    
    def save_index(self, index_path: str = "faiss_index") -> bool:
        """
        Save the FAISS index to disk.
        
        Args:
            index_path: Path to save the index
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.vector_store:
                print("No vector store configured.")
                return False
            
            return self.vector_store.save(index_path)
            
        except Exception as e:
            print(f"Error saving index: {str(e)}")
            return False
    
    def load_index(self, index_path: str = "faiss_index", api_key: str = None) -> bool:
        """
        Load a FAISS index from disk.
        
        Args:
            index_path: Path to load the index from
            api_key: OpenAI API key (required for loading)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.vector_store:
                print("No vector store configured.")
                return False
            
            if not api_key:
                print("API key required for loading index")
                return False
            
            return self.vector_store.load(index_path, api_key)
            
        except Exception as e:
            print(f"Error loading index: {str(e)}")
            return False


