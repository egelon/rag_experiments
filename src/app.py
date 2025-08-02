import streamlit as st
import os
import shutil
from pathlib import Path
from langchain_openai.chat_models import ChatOpenAI

# Import the process_input_files function from utils
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'rag_experiments'))
from utils import process_input_files

st.title("RAG Experiments")

# Sidebar configuration
st.sidebar.header("Configuration")

openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
st.sidebar.markdown("""
[Get your OpenAI API key](https://platform.openai.com/account/api-keys)
""")

# Model selection dropdown
model_options = {
    "GPT-4o": "gpt-4o",
    "GPT-4.1": "gpt-4.1", 
    "GPT-4o Mini": "gpt-4o-mini",
    "GPT-4o Mini High": "gpt-4o-mini-high"
}

selected_model = st.sidebar.selectbox(
    "Select OpenAI Model:",
    options=list(model_options.keys()),
    index=0  # Default to GPT-4o
)

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
    
    for uploaded_file in uploaded_files:
        st.write(f"📄 {uploaded_file.name}")
    
    # Upload button
    if st.button("📤 Upload Files to input_files Directory"):
        uploaded_count = 0
        for uploaded_file in uploaded_files:
            try:
                # Create the file path in input_files directory
                file_path = input_files_dir / uploaded_file.name
                
                # Save the uploaded file
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                uploaded_count += 1
                st.success(f"✅ Successfully uploaded: {uploaded_file.name}")
                
            except Exception as e:
                st.error(f"❌ Error uploading {uploaded_file.name}: {str(e)}")
        
        if uploaded_count > 0:
            st.success(f"🎉 Successfully uploaded {uploaded_count} file(s) to input_files directory!")
            
            # Process the uploaded files automatically
            st.info("🔄 Processing uploaded files...")
            try:
                process_input_files()
                st.success("✅ Files processed successfully! Markdown files created in processed_files directory.")
            except Exception as e:
                st.error(f"❌ Error processing files: {str(e)}")

# Display existing files in input_files directory
st.header("📂 Current Files in input_files Directory")

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

def generate_response(input_text, model_name):
    model = ChatOpenAI(
        model=model_name,
        temperature=0.7, 
        api_key=openai_api_key
    )
    response = model.invoke(input_text)
    st.info(response.content)

# Chat interface
st.header("💬 Chat with AI")
with st.form("my_form"):
    text = st.text_area(
        "Enter text:",
        "How many moons does Pluto have?",
    )
    submitted = st.form_submit_button("Submit")
    if not openai_api_key.startswith("sk-"):
        st.warning("Please enter your OpenAI API key!", icon="⚠")
    if submitted and openai_api_key.startswith("sk-"):
        generate_response(text, model_options[selected_model])
