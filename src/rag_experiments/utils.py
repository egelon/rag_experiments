import pymupdf
import pymupdf4llm
import os
import glob
import pathlib
import re

def get_input_file_names():
    pdf_files = []
    input_dir = "input_files"
    
    # Sanity Check: Check if input_files directory exists
    if not os.path.exists(input_dir):
        print(f"ERROR: {input_dir} directory does not exist")
        return pdf_files
    
    # Sanity Check: Check if directory is empty
    try:
        if not os.listdir(input_dir):
            print(f"WARNING: {input_dir} directory is empty")
            return pdf_files
    except PermissionError:
        print(f"ERROR: Permission denied when trying to read {input_dir} directory")
        return pdf_files
    
    # Get all PDF files only from input_files directory
    pattern = os.path.join(input_dir, "*.pdf")
    pdf_paths = glob.glob(pattern)
    
    # Sanity Check: Check if any PDF files were found
    if not pdf_paths:
        print(f"WARNING: No PDF files found in {input_dir} directory")
        return pdf_files
    
    # Extract just the filenames (without path)
    for pdf_path in pdf_paths:
        filename = os.path.basename(pdf_path)
        pdf_files.append(filename)
    
    print(f"Found {len(pdf_files)} PDF files in {input_dir}")
    return pdf_files

def cleanup_text(md_text):
    """
    Clean up markdown text by removing multiple consecutive newlines.
    
    Args:
        md_text (str): The markdown text to clean
        
    Returns:
        str: The cleaned markdown text
    """
    return re.sub(r'\n{3,}', '\n', md_text)


def process_input_files():
    """
    Process PDF files from input_files directory and convert them to markdown format.
    Creates processed_files directory if it doesn't exist and saves markdown files there.
    """
    # Create processed_files directory in the repository root if it doesn't exist
    try:
        # Get the repository root using relative path from current file
        repo_root = pathlib.Path(__file__).parent
        processed_files_dir = repo_root.resolve() / "processed_files"
        processed_files_dir.mkdir(exist_ok=True)
        print(f"Using directory: {processed_files_dir.absolute()}")
    except OSError as e:
        print(f"Error creating processed_files directory: {e}")
        sys.exit(1)

    pdf_files = get_input_file_names()
    for filename in pdf_files:
        print(f"Processing {filename}")
        document = pymupdf.open(f"input_files/{filename}", filetype="pdf")   
        md_text = pymupdf4llm.to_markdown(document)
        
        # Sanity check: did we get text?
        if not md_text or md_text.strip() == "":
            print(f"Warning: No text content extracted from {filename}")
            continue
        
        try:
            # Clean up multiple consecutive newlines (empty rows)
            cleaned_md_text = cleanup_text(md_text)
            
            output_file = processed_files_dir / f"{filename}.md"
            output_file.write_bytes(cleaned_md_text.encode())
            print(f"Successfully wrote: {output_file}")
        except OSError as e:
            print(f"Error writing {filename}.md: {e}")
            continue