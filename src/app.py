import streamlit as st
import os
import shutil
import time
from pathlib import Path

# Import MVC components
from config import Config
from controllers import RAGController
from models import VectorStore


# Import the process_input_files function from utils
from utils.utils import process_input_files

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

# Check if API key is provided and valid before showing other config options
if not openai_api_key or not openai_api_key.startswith("sk-"):
    st.info("🔑 Please enter your OpenAI API key in the sidebar to access the application features.")
    st.stop()

# Model selection dropdown
model_options = {
    "GPT-4.1": "gpt-4.1",
    "GPT-4.1 Nano": "gpt-4.1-nano",
    "GPT-4o": "gpt-4o",
    "GPT-4o Mini": "gpt-4o-mini"
}

# Initialize selected model in session state if not exists
if 'selected_model_key' not in st.session_state:
    st.session_state.selected_model_key = "GPT-4o Mini"  # Default
    # Initialize config with the default model
    st.session_state.config.update_chat_model(model_options[st.session_state.selected_model_key])

# Initialize chunk configuration tracking
if 'last_chunk_size' not in st.session_state:
    st.session_state.last_chunk_size = st.session_state.config.chunk_size
if 'last_chunk_overlap' not in st.session_state:
    st.session_state.last_chunk_overlap = st.session_state.config.chunk_overlap
if 'config_changed' not in st.session_state:
    st.session_state.config_changed = False

# Initialize temperature tracking
if 'last_temperature' not in st.session_state:
    st.session_state.last_temperature = st.session_state.config.temperature

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

# Temperature configuration
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Model Settings")

def on_temperature_change():
    """Callback function when temperature changes"""
    new_temperature = st.session_state.temperature_input
    
    # Check if value actually changed
    if new_temperature != st.session_state.last_temperature:
        # Update config using the proper config method
        success, error_msg = st.session_state.config.update_temperature(new_temperature)
        
        if not success:
            st.sidebar.error(f"❌ Invalid temperature: {error_msg}")
            # Reset input to last valid value
            st.session_state.temperature_input = st.session_state.last_temperature
            return
        
        # Reinitialize chat model if controller exists
        if st.session_state.controller and st.session_state.initialized:
            try:
                success, error_msg = st.session_state.controller.initialize()
                if not success:
                    st.sidebar.error(f"❌ Model reinitialization error: {error_msg}")
                else:
                    st.sidebar.success("✅ Model updated with new temperature")
            except Exception as e:
                st.sidebar.error(f"❌ Error updating model: {str(e)}")
        
        # Update tracking value
        st.session_state.last_temperature = new_temperature

# Temperature input
temperature = st.sidebar.slider(
    "🌡️ Temperature:",
    min_value=0.0,
    max_value=2.0,
    value=st.session_state.config.temperature,
    step=0.1,
    help="Controls randomness in responses (0.0 = deterministic, 2.0 = very creative)",
    key="temperature_input",
    on_change=on_temperature_change
)

# Chunk configuration controls
st.sidebar.markdown("---")
st.sidebar.subheader("📄 Document Processing")

def on_chunk_config_change():
    """Callback function when chunk configuration changes"""
    # Check if values actually changed
    new_chunk_size = st.session_state.chunk_size_input
    new_chunk_overlap = st.session_state.chunk_overlap_input
    
    if (new_chunk_size != st.session_state.last_chunk_size or 
        new_chunk_overlap != st.session_state.last_chunk_overlap):
        
        # Update config using the proper config method
        success, error_msg = st.session_state.config.update_chunk_config(
            chunk_size=new_chunk_size,
            chunk_overlap=new_chunk_overlap
        )
        
        if not success:
            st.sidebar.error(f"❌ Invalid configuration: {error_msg}")
            # Reset inputs to last valid values
            st.session_state.chunk_size_input = st.session_state.last_chunk_size
            st.session_state.chunk_overlap_input = st.session_state.last_chunk_overlap
            return
        
        # Mark config as changed
        st.session_state.config_changed = True
        
        # Delete existing index if controller is initialized
        if st.session_state.controller and st.session_state.initialized:
            # Try to delete the index files
            try:
                if Path("faiss_index").exists():
                    shutil.rmtree("faiss_index")
                    st.sidebar.warning("⚠️ Index deleted due to configuration change. Please rebuild the index.")
                    
                # Reinitialize controller with new config
                st.session_state.controller = RAGController(st.session_state.config)
                success, error_msg = st.session_state.controller.initialize()
                if not success:
                    st.sidebar.error(f"❌ Reinitialization error: {error_msg}")
                    
            except Exception as e:
                st.sidebar.error(f"❌ Error deleting index: {str(e)}")
        
        # Update tracking values
        st.session_state.last_chunk_size = new_chunk_size
        st.session_state.last_chunk_overlap = new_chunk_overlap

# Chunk size input
chunk_size = st.sidebar.number_input(
    "Chunk Size (characters):",
    min_value=100,
    max_value=4000,
    value=st.session_state.config.chunk_size,
    step=100,
    help="Size of each text chunk for processing",
    key="chunk_size_input",
    on_change=on_chunk_config_change
)

# Chunk overlap input
chunk_overlap = st.sidebar.number_input(
    "Chunk Overlap (characters):",
    min_value=0,
    max_value=min(chunk_size - 1, 1000),
    value=st.session_state.config.chunk_overlap,
    step=50,
    help="Overlap between consecutive chunks",
    key="chunk_overlap_input",
    on_change=on_chunk_config_change
)

# Validation warning
if chunk_overlap >= chunk_size:
    st.sidebar.error("⚠️ Chunk overlap must be less than chunk size!")

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
create_index_disabled = not st.session_state.initialized
if st.session_state.config_changed:
    button_text = "🔄 Rebuild Index (Config Changed)"
else:
    button_text = "🔄 Create Index"

if st.button(button_text, disabled=create_index_disabled):
    if st.session_state.controller:
        with st.spinner("Processing documents and creating index..."):
            # Check if processed_files directory is empty and needs PDF processing first
            processed_files_path = Path(st.session_state.config.processed_files_dir)
            processed_files_exist = processed_files_path.exists() and any(processed_files_path.glob("*.md"))
            
            if not processed_files_exist:
                # Check if input_files directory has PDF files to process
                input_files_path = Path(st.session_state.config.input_files_dir)
                pdf_files_exist = input_files_path.exists() and any(input_files_path.glob("*.pdf"))
                
                if pdf_files_exist:
                    st.info("🔄 No processed files found. Converting PDF files to markdown first...")
                    try:
                        process_input_files()
                        st.success("✅ PDF files converted to markdown successfully!")
                    except Exception as e:
                        st.error(f"❌ Error processing PDF files: {str(e)}")
                        st.stop()
                else:
                    st.error("❌ No PDF files found in input_files directory. Please upload some PDF files first.")
                    st.stop()
            
            success, error_msg = st.session_state.controller.load_and_process_documents()
            if success:
                # Save the index
                save_success, save_error = st.session_state.controller.save_index()
                if save_success:
                    st.success("✅ Successfully processed documents and created index!")
                    # Reset config changed flag
                    st.session_state.config_changed = False
                    # Show stats
                    stats = st.session_state.controller.get_stats()
                    st.info(f"📊 Index contains {stats.get('document_count', 0)} chunks")
                else:
                    st.error(f"❌ Error saving index: {save_error}")
            else:
                st.error(f"❌ Error: {error_msg}")

# Display current configuration and statistics
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Current Settings")
st.sidebar.info(f"**Active Model:** {st.session_state.selected_model_key}")

# Show chunk configuration status
if st.session_state.config_changed:
    st.sidebar.warning(f"⚠️ **Config Changed - Rebuild Required**\nChunk Size: {st.session_state.config.chunk_size}\nChunk Overlap: {st.session_state.config.chunk_overlap}")
else:
    st.sidebar.success(f"**Document Processing Config:**\nChunk Size: {st.session_state.config.chunk_size}\nChunk Overlap: {st.session_state.config.chunk_overlap}")

if st.session_state.controller:
    stats = st.session_state.controller.get_stats()
    if stats.get('document_count', 0) > 0:
        st.sidebar.markdown("### 📊 Vector Store Stats")
        st.sidebar.info(f"Chunks: {stats.get('document_count', 0)}\nEmbedding Model: {stats.get('embedding_model', 'N/A')}\nEmbedding Dimensions: {stats.get('embedding_dimension', 'N/A')}")
    elif st.session_state.initialized:
        st.sidebar.info("No vector index found")

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

# Chat interface - only show if index exists and config hasn't changed
if st.session_state.controller and st.session_state.initialized:
    stats = st.session_state.controller.get_stats()
    has_index = stats.get('document_count', 0) > 0
    
    if not has_index or st.session_state.config_changed:
        if st.session_state.config_changed:
            st.info("⚠️ Document processing configuration has changed. Please rebuild the index before chatting.")
        else:
            st.info("📊 No vector index found. Please create an index first by clicking the '🔄 Create Index' button above.")
    else:
        st.header("💬 Chat with AI")
        with st.form("my_form"):
            text = st.text_area(
                "Enter text:",
                "How many moons does Pluto have?",
            )
            search_k = st.number_input("Results", min_value=1, max_value=10, value=4)
            
            submitted = st.form_submit_button("Submit")
            if submitted:
                # Update search settings
                st.session_state.controller.update_config(default_search_k=search_k)
                generate_response(text)
else:
    st.info("🔑 Please enter your OpenAI API key to enable chat functionality.")
