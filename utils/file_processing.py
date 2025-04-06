import os
import tempfile
import streamlit as st
from typing import List, Dict, Any, Tuple, Optional, Union

# Import optional dependencies - we'll handle cases where they're not available
try:
    import PyPDF2
    PYPDF2_AVAILABLE = True
except ImportError:
    PYPDF2_AVAILABLE = False

try:
    from langchain.document_loaders import PyPDFLoader, TextLoader, CSVLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

def save_uploaded_file(uploaded_file) -> str:
    """
    Save an uploaded file to a temporary directory and return the path
    
    Args:
        uploaded_file: The uploaded file from st.file_uploader
        
    Returns:
        str: Path to the saved file
    """
    # Create a temporary directory if it doesn't exist
    temp_dir = tempfile.mkdtemp()
    
    # Define path for saving the file
    file_path = os.path.join(temp_dir, uploaded_file.name)
    
    # Write the file
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    return file_path

def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract text from a PDF file
    
    Args:
        file_path: Path to the PDF file
        
    Returns:
        str: Extracted text
    """
    if not PYPDF2_AVAILABLE:
        raise ImportError("PyPDF2 is required for PDF processing. Please install it with 'pip install PyPDF2'.")
    
    text = ""
    with open(file_path, "rb") as f:
        pdf_reader = PyPDF2.PdfReader(f)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text() + "\n\n"
    
    return text

def extract_text_from_txt(file_path: str) -> str:
    """
    Extract text from a text file
    
    Args:
        file_path: Path to the text file
        
    Returns:
        str: Extracted text
    """
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    return text

def extract_text(file_path: str) -> str:
    """
    Extract text from a file based on its extension
    
    Args:
        file_path: Path to the file
        
    Returns:
        str: Extracted text
    """
    if file_path.lower().endswith(".pdf"):
        return extract_text_from_pdf(file_path)
    elif file_path.lower().endswith(".txt"):
        return extract_text_from_txt(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks of specified size
    
    Args:
        text: Text to split
        chunk_size: Size of each chunk in characters
        chunk_overlap: Overlap between chunks in characters
        
    Returns:
        List[str]: List of text chunks
    """
    chunks = []
    
    if not text:
        return chunks
    
    # Split text into sentences (simple approach)
    sentences = text.replace("\n", " ").split(". ")
    sentences = [s + "." for s in sentences if s]
    
    current_chunk = ""
    
    for sentence in sentences:
        # If adding the next sentence would exceed the chunk size, store the chunk and start a new one
        if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
            chunks.append(current_chunk)
            # Start new chunk with overlap
            words = current_chunk.split(" ")
            overlap_text = " ".join(words[-int(chunk_overlap/10):])  # Approximate word count for overlap
            current_chunk = overlap_text + " " + sentence
        else:
            current_chunk += " " + sentence
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def process_file_with_langchain(file_path: str) -> List[Dict[str, Any]]:
    """
    Process a file using LangChain document loaders and text splitters
    
    Args:
        file_path: Path to the file
        
    Returns:
        List[Dict[str, Any]]: List of document chunks with metadata
    """
    if not LANGCHAIN_AVAILABLE:
        raise ImportError("LangChain is required for advanced document processing. Please install with 'pip install langchain'.")
    
    # Select the appropriate loader based on file extension
    if file_path.lower().endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.lower().endswith(".txt"):
        loader = TextLoader(file_path)
    elif file_path.lower().endswith(".csv"):
        loader = CSVLoader(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    
    # Load documents
    documents = loader.load()
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    document_chunks = text_splitter.split_documents(documents)
    
    # Convert to dictionary format for easier handling
    result = []
    for chunk in document_chunks:
        result.append({
            "content": chunk.page_content,
            "metadata": chunk.metadata
        })
    
    return result

def get_file_stats(file_path: str) -> Dict[str, Any]:
    """
    Get statistics about a processed file
    
    Args:
        file_path: Path to the file
        
    Returns:
        Dict[str, Any]: Dictionary with file statistics
    """
    file_name = os.path.basename(file_path)
    file_size = os.path.getsize(file_path)
    
    # Extract text
    try:
        text = extract_text(file_path)
        word_count = len(text.split())
        char_count = len(text)
    except Exception as e:
        text = None
        word_count = 0
        char_count = 0
    
    return {
        "file_name": file_name,
        "file_path": file_path,
        "file_size_bytes": file_size,
        "file_size_kb": round(file_size / 1024, 2),
        "word_count": word_count,
        "char_count": char_count
    }

def display_file_preview(file_path: str, max_length: int = 500) -> None:
    """
    Display a preview of a file in Streamlit
    
    Args:
        file_path: Path to the file
        max_length: Maximum length of the preview text
    """
    try:
        text = extract_text(file_path)
        
        # Show preview with a reasonable length
        preview = text[:max_length] + ("..." if len(text) > max_length else "")
        
        st.subheader("File Preview")
        st.text(preview)
        
        # Show stats
        stats = get_file_stats(file_path)
        st.info(f"Total words: {stats['word_count']} | Characters: {stats['char_count']}")
        
    except Exception as e:
        st.error(f"Error previewing file: {str(e)}")
