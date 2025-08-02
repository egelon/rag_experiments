# rag_experiments
A simple Retrieval-Augmented Generation (RAG) application that processes user queries based on a small dataset of documents.
The RAG application follows the Model-View-Controller (MVC) architectural pattern.

## Architecture Components

### 1. **Model Layer** (`src/models/`)
The Model layer handles data storage and retrieval operations.

- **VectorStore** (`vector_store.py`): 
  - Manages FAISS vector database operations
  - Handles storage, persistence, and CRUD operations
  - Single responsibility: vector store management only

- **QueryService** (`query_service.py`):
  - Handles all search and retrieval operations
  - Provides similarity search functionality

### 2. **Controller Layer** (`src/controllers/`)
The Controller layer manages business logic and coordinates between Model and View.

- **RAGController** (`rag_controller.py`):
  - Main controller for the application
  - Coordinates between UI and Models
  - Manages document processing pipeline
  - Handles chat interactions and RAG queries

### 3. **View Layer** (`src/app.py`)
The View layer handles user interface and user interactions.

- **Streamlit UI**:
  - Accepts user input (including API key)
  - Displays results and feedback

### 4. **Configuration** (`src/config/`)
Centralized configuration management.

- **Config** (`config.py`):
  - Stores all application settings
  - Manages API keys from UI
  - Provides validation

### 5. **Document Processing** (`utils/DocumentProcessor.py`)

- Accepts vector store through dependency injection

## Usage Example

```python
# 1. User provides API key in UI
# 2. UI updates Config with the key
# 3. User picks a chat model
# 4. UI updates COnfig with chat model
# 5. UI initializes Controller with Config
# 6. Controller initializes Models with API key
# 7. User uploads PDFs
# 8. User triggers document processing
# 9. PDFs are processed - their text is stripped and saved in *.md files in processed_files
# 10. User triggers index generation
# 11. Index is generated and stored in faiss_index
# 12. user posts a question, and picks how many chunks to fetch as additional context
# 13. Controller coordinates the process
# 14. Results displayed in UI
```

## App Setup

- Checkout the repository
- Make sure you have python 3.13 or higher, and it is set in PATH correctly
- Run setup.bat - this will create a python virtual environment and download required dependencies

## App Execution

Run start_app.bat after you've performed the setup - this will start the streamlit UI on http://localhost:8501/

## Used Tools
- Python
- Cursor as IDE
- pymupdf4llm for parsing raw text from PDFs
- LangChain for Document ingestion and Chunking
- FAISS for vector store
- streamlit for the UI
- LangChain's OpenAI wrappers for the OpenAI Embeddings for the vector store, and for the chat functionality

## Explanation

This application implements a MVC architecture to process and query PDF documents using OpenAI's language models. The system works by:

1. **Document Processing**: PDF files are parsed using PyMuPDF4LLM, converted to markdown, and then chunked using LangChain's text splitters with configurable chunk sizes (default: 1000 characters with 200 character overlap).

2. **Vector Storage**: Document chunks are embedded using OpenAI's `text-embedding-ada-002` model and stored in a FAISS vector database for efficient similarity search.

3. **Query Processing**: User queries are embedded and matched against stored document chunks using FAISS similarity search, with configurable retrieval count (default: 4 chunks).

4. **Response Generation**: Retrieved chunks provide context to OpenAI's chat models (default: GPT-4o) to generate informed responses.

**Key Tools & Technologies**:
- **Streamlit**: Web-based user interface
- **LangChain**: Document processing, chunking, and OpenAI integration
- **FAISS**: High-performance vector similarity search
- **OpenAI API**: Embeddings and chat completion
- **PyMuPDF4LLM**: PDF text extraction

**Key Assumptions**:
- Users have valid OpenAI API keys
- Documents are text-based PDFs suitable for extraction
- Single-user deployment model
- CPU-based FAISS is sufficient for dataset size
- Default chunking strategy works for most document types
- Network connectivity to OpenAI services is reliable, and API rate limits are never reached

## Scalability Considerations

While the current implementation is designed for small datasets, several improvements could be made:

### **Data Handling & Storage**
- **Database Migration**: Replace FAISS with cloud-native vector databases
- **Caching Layer**: Add caching for frequently accessed embeddings and query results (Redis perhaps?)

### **Performance Optimization**
- **GPU Acceleration**: Switch to GPU for faster similarity search on large datasets
- **Embedding Optimization**: Use local, open-source embedding models to reduce API costs
- **Chunk Strategy**: Implement a different chunking strategy for better context preservation (chunk at paragraphs, or sentence bounderies)
- **Connection Pooling**: Optimize OpenAI API usage with connection pooling and rate limiting

### **Architecture Enhancements**
- **Container Orchestration**: Use Docker iamge for reliable deployment

### **Users & Error handling**
- **User Management**: Add authentication and authorization for multi-user scenarios
- **Error Handling**: Robust error handling with retry mechanisms and circuit breakers
