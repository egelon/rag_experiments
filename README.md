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
  - Provides various search methods (similarity, MMR, etc.)

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

### 5. **Document Processing** (`src/DocumentProcessor.py`)

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

Checkout the repository
Make sure you have python 3.13 or higher, and it is set in PATH correctly
Run setup.bat - this will create a python virtual environment and download required dependencies

## App Execution

Run start_app.bat after you've performed the setup - this will start the streamlit UI on http://localhost:8501/