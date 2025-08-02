## Overview

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
# 3. UI initializes Controller with Config
# 4. Controller initializes Models with API key
# 5. User triggers document processing
# 6. Controller coordinates the process
# 7. Results displayed in UI
```
