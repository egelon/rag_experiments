from langchain_community.document_loaders import UnstructuredMarkdownLoader 
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pathlib import Path
from typing import List, Optional
import os
from VectorStoreWrapper import VectorStoreWrapper


class DocumentProcessor:
    def __init__(
        self, 
        processed_files_dir: str = "processed_files",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embedding_model: str = "text-embedding-ada-002"
    ):
        """
        Initialize the DocumentProcessor.
        
        Args:
            processed_files_dir (str): Directory containing processed markdown files
            chunk_size (int): Size of text chunks for splitting
            chunk_overlap (int): Overlap between chunks
            embedding_model (str): OpenAI embedding model to use
        """
        self.processed_files_dir = processed_files_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model
        
        # Initialize components
        self.documents = []
        self.chunks = []
        self.vectorstore = None
        
        # Setup text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            add_start_index=True,
        )
    
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
    
    def create_index(self, chunks: Optional[List] = None, api_key: Optional[str] = None) -> VectorStoreWrapper:
        """
        Create a FAISS vector index from document chunks using VectorStoreWrapper.
        
        Args:
            chunks: List of document chunks. If None, uses self.chunks
            api_key: OpenAI API key. If None, tries to get from environment
            
        Returns:
            VectorStoreWrapper instance
        """
        try:
            chunks_to_index = chunks if chunks is not None else self.chunks
            
            if not chunks_to_index:
                print("No chunks to index. Load and chunk documents first.")
                return None
            
            # Get API key
            openai_api_key = os.getenv("OPENAI_API_KEY") or api_key
            if not openai_api_key:
                raise ValueError("OpenAI API key not provided and not found in environment")
            
            # Create VectorStoreWrapper
            vector_wrapper = VectorStoreWrapper(
                embedding_model=self.embedding_model,
                api_key=openai_api_key
            )
            
            # Extract text content from document chunks
            chunk_texts = []
            for chunk in chunks_to_index:
                if hasattr(chunk, 'page_content'):
                    chunk_texts.append(chunk.page_content)
                else:
                    chunk_texts.append(str(chunk))
            
            # Create FAISS vectorstore from chunks
            try:
                success = vector_wrapper.create_from_chunks(chunk_texts)
            except Exception as e:
                print(f"Error creating vector store from chunks: {str(e)}")
                return None
            
            if success:
                self.vectorstore = vector_wrapper
                print(f"Successfully created FAISS index with {len(chunks_to_index)} chunks using VectorStoreWrapper")
                return vector_wrapper
            else:
                print("Failed to create vector store")
                return None
            
        except Exception as e:
            print(f"Error creating index: {str(e)}")
            return None
    
    def process_all(self, api_key: Optional[str] = None) -> VectorStoreWrapper:
        """
        Complete pipeline: load documents, chunk them, and create index.
        
        Args:
            api_key: OpenAI API key
            
        Returns:
            VectorStoreWrapper instance
        """
        print("Starting document processing pipeline...")
        
        # Load documents
        self.load_documents()
        
        # Chunk documents
        self.chunk_documents()
        
        # Create index
        self.create_index(api_key=api_key)
        
        print("Document processing pipeline completed!")
        
        return self.vectorstore
    
    def save_index(self, index_path: str = "faiss_index") -> bool:
        """
        Save the FAISS index to disk.
        
        Args:
            index_path: Path to save the index
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.vectorstore is None:
                print("No vectorstore to save. Create index first.")
                return False
            
            return self.vectorstore.save(index_path)
            
        except Exception as e:
            print(f"Error saving index: {str(e)}")
            return False
    
    def load_index(self, index_path: str = "faiss_index", api_key: Optional[str] = None) -> VectorStoreWrapper:
        """
        Load a FAISS index from disk.
        
        Args:
            index_path: Path to load the index from
            api_key: OpenAI API key
            
        Returns:
            VectorStoreWrapper instance
        """
        try:
            # Get API key
            openai_api_key = api_key or os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                raise ValueError("OpenAI API key not provided and not found in environment")
            
            #check if index path exists
            if not Path(index_path).exists():
                raise FileNotFoundError(f"Index path does not exist: {index_path}")

            if self.vectorstore is None:
                print("No vectorstore to load. Create index first.")
                return False
            
            return self.vectorstore.load(index_path)
            
        except Exception as e:
            print(f"Error loading index: {str(e)}")
            return None


# Example usage
if __name__ == "__main__":
    # Initialize processor
    processor = DocumentProcessor(
        processed_files_dir="processed_files",
        chunk_size=1000,
        chunk_overlap=200
    )
    
    # Process all documents (you'll need to provide your API key)
    # vectorstore = processor.process_all(api_key="your-openai-api-key")
    
    # Or step by step:
    documents = processor.load_documents()
    chunks = processor.chunk_documents(documents)
    vectorstore = processor.create_index(chunks, api_key="your-openai-api-key")
    isIndexSaved = processor.save_index()
    if isIndexSaved:
        print("Index saved successfully")
    else:
        print("Index not saved")
    isIndexLoaded = processor.load_index()
    if isIndexLoaded:
        print("Index loaded successfully")
    else:
        print("Index not loaded")