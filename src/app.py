import streamlit as st
import os
import shutil
import time
from pathlib import Path

# Import MVC components
from config import Config
from controllers import RAGController
from models import VectorStore
from DocumentProcessor import DocumentProcessor

# Import the process_input_files function from utils
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'rag_experiments'))
from utils import process_input_files

st.title("RAG Experiments")

# Initialize session state for MVC components
if 'config' not in st.session_state:
    st.session_state.config = Config()
if 'controller' not in st.session_state:
    st.session_state.controller = None
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

# Sidebar configuration
st.sidebar.header("Configuration")

openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
st.sidebar.markdown("""
[Get your OpenAI API key](https://platform.openai.com/account/api-keys)
""")

# Model selection dropdown
model_options = {
    "GPT-4.1": "gpt-4.1",
    "GPT-4.1 Nano": "gpt-4.1-nano",
    "GPT-4o": "gpt-4o",
    "GPT-4o Mini": "gpt-4o-mini",
    "o4 Mini": "o4-mini"
}

# Initialize selected model in session state if not exists
if 'selected_model_key' not in st.session_state:
    st.session_state.selected_model_key = "GPT-4o Mini"  # Default
    # Initialize config with the default model
    st.session_state.config.update_chat_model(model_options[st.session_state.selected_model_key])

def on_model_change():
    """Callback function when model selection changes"""
    selected_key = st.session_state.model_selector
    st.session_state.selected_model_key = selected_key
    # Update configuration
    st.session_state.config.update_chat_model(model_options[selected_key])
    # Reinitialize chat model if controller exists
    if st.session_state.controller and st.session_state.initialized:
        st.session_state.controller.initialize()

# Get current index for the selectbox
current_index = list(model_options.keys()).index(st.session_state.selected_model_key)

selected_model = st.sidebar.selectbox(
    "Select OpenAI Model:",
    options=list(model_options.keys()),
    index=current_index,
    key="model_selector",
    on_change=on_model_change
)

# Update API key configuration and initialize controller
if openai_api_key:
    st.session_state.config.update_api_key(openai_api_key)
    
    # Initialize controller if API key is provided and not yet initialized
    if not st.session_state.initialized and openai_api_key.startswith("sk-"):
        st.session_state.controller = RAGController(st.session_state.config)
        success, error_msg = st.session_state.controller.initialize()
        if success:
            st.session_state.initialized = True
            # Try to load existing index
            load_success, _ = st.session_state.controller.load_index()
            if load_success:
                st.sidebar.success("✅ Loaded existing vector index")
        else:
            st.sidebar.error(f"❌ Initialization error: {error_msg}")

# File upload section
st.header("📁 Upload PDF Files")
st.markdown("Upload PDF files to the `input_files` directory for processing.")

# Create input_files directory if it doesn't exist
input_files_dir = Path("input_files")
input_files_dir.mkdir(exist_ok=True)

# File uploader
uploaded_files = st.file_uploader(
    "Choose PDF files",
    type=['pdf'],
    accept_multiple_files=True,
    help="Select one or more PDF files to upload"
)

if uploaded_files:
    st.write(f"**{len(uploaded_files)} file(s) selected:**")
    
    #for uploaded_file in uploaded_files:
    #    st.write(f"📄 {uploaded_file.name}")
    
    # Upload button
    if st.button("📤 Upload and process files"):
        uploaded_count = 0
        
        # Create containers for temporary messages
        upload_status_container = st.empty()
        processing_status_container = st.empty()
        
        for uploaded_file in uploaded_files:
            try:
                # Create the file path in input_files directory
                file_path = input_files_dir / uploaded_file.name
                
                # Save the uploaded file
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                uploaded_count += 1
                upload_status_container.success(f"✅ Successfully uploaded: {uploaded_file.name}")
                time.sleep(1)  # Show message for 1 seconds
                upload_status_container.empty()  # Hide the message
                
            except Exception as e:
                upload_status_container.error(f"❌ Error uploading {uploaded_file.name}: {str(e)}")
                time.sleep(1)  # Show error for 1 seconds
                upload_status_container.empty()  # Hide the message
        
        if uploaded_count > 0:
            upload_status_container.success(f"🎉 Successfully uploaded {uploaded_count} file(s) to input_files directory!")
            time.sleep(1)  # Show message for 1 seconds
            upload_status_container.empty()  # Hide the message
            
            # Process the uploaded files automatically
            processing_status_container.info("🔄 Processing uploaded files...")
            try:
                process_input_files()
                processing_status_container.success("✅ Files processed successfully! Markdown files created in processed_files directory.")
                time.sleep(4)  # Show success message for 4 seconds
                processing_status_container.empty()  # Hide the message
            except Exception as e:
                processing_status_container.error(f"❌ Error processing files: {str(e)}")
                time.sleep(4)  # Show error for 4 seconds
                processing_status_container.empty()  # Hide the message

# Display existing files in input_files directory
st.header("📂 Stored Files")

if input_files_dir.exists() and any(input_files_dir.iterdir()):
    existing_files = list(input_files_dir.glob("*.pdf"))
    if existing_files:
        st.write(f"**Found {len(existing_files)} PDF file(s):**")
        for file_path in existing_files:
            file_size = file_path.stat().st_size / 1024  # Size in KB
            st.write(f"📄 {file_path.name} ({file_size:.1f} KB)")
    else:
        st.info("No PDF files found in input_files directory.")
else:
    st.info("input_files directory is empty.")

# Add button to process existing files and create index
if st.button("🔄 Create Index", disabled=not st.session_state.initialized):
    if st.session_state.controller:
        with st.spinner("Processing documents and creating index..."):
            success, error_msg = st.session_state.controller.load_and_process_documents()
            if success:
                # Save the index
                save_success, save_error = st.session_state.controller.save_index()
                if save_success:
                    st.success("✅ Successfully processed documents and created index!")
                    # Show stats
                    stats = st.session_state.controller.get_stats()
                    st.info(f"📊 Index contains {stats.get('document_count', 0)} chunks")
                else:
                    st.error(f"❌ Error saving index: {save_error}")
            else:
                st.error(f"❌ Error: {error_msg}")

# Display current configuration and statistics
st.sidebar.markdown("### ⚙️ Current Settings")
st.sidebar.info(f"Active Model: {st.session_state.selected_model_key}")

if st.session_state.controller:
    stats = st.session_state.controller.get_stats()
    if stats.get('document_count', 0) > 0:
        st.sidebar.markdown("### 📊 Vector Store Stats")
        st.sidebar.info(f"Documents: {stats.get('document_count', 0)}\nEmbedding Model: {stats.get('embedding_model', 'N/A')}")

def generate_response(input_text):
    """Generate response using the RAG controller."""
    if not st.session_state.controller:
        st.error("Controller not initialized. Please provide API key.")
        return
    
    # Check if we have a vector store loaded
    stats = st.session_state.controller.get_stats()
    
    if stats.get('document_count', 0) > 0:
        # Use RAG query with context
        with st.spinner("Searching for relevant context..."):
            response, context = st.session_state.controller.rag_query(input_text)
            
        # Display the response
        st.info(response)
        
        # Show context documents in an expander
        if context:
            with st.expander(f"📚 Context ({len(context)} documents)"):
                for i, doc in enumerate(context):
                    st.markdown(f"**Document {i+1}:**")
                    st.text(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                    st.markdown("---")
    else:
        # No vector store, use regular chat
        response = st.session_state.controller.chat(input_text)
        st.info(response)

# Chat interface
st.header("💬 Chat with AI")
with st.form("my_form"):
    text = st.text_area(
        "Enter text:",
        "How many moons does Pluto have?",
    )
    col1, col2 = st.columns([1, 5])
    with col1:
        search_k = st.number_input("Results", min_value=1, max_value=10, value=4)
    with col2:
        use_mmr = st.checkbox("Use MMR (diverse results)", value=False)
    
    submitted = st.form_submit_button("Submit")
    if not openai_api_key.startswith("sk-"):
        st.warning("Please enter your OpenAI API key!", icon="⚠")
    if submitted and openai_api_key.startswith("sk-") and st.session_state.initialized:
        # Update search settings
        st.session_state.controller.update_config(default_search_k=search_k)
        generate_response(text)
