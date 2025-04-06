import os
import time
from typing import List, Dict, Any, Tuple, Optional, Union
import streamlit as st

# Import optional dependencies - we'll handle cases where they're not available
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.llms import HuggingFaceHub, HuggingFacePipeline
    from langchain.vectorstores import Chroma
    from langchain.chains import RetrievalQA
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

def load_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    """
    Load a sentence transformer model for generating embeddings
    
    Args:
        model_name: Name of the model to load
        
    Returns:
        The loaded model or a function that can generate embeddings
    """
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        raise ImportError("Sentence Transformers is required for embeddings. Please install with 'pip install sentence-transformers'.")
    
    # Load the model
    return SentenceTransformer(model_name)

def generate_embeddings(model, texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a list of texts
    
    Args:
        model: The sentence transformer model
        texts: List of texts to embed
        
    Returns:
        List[List[float]]: List of embedding vectors
    """
    return model.encode(texts, convert_to_tensor=False).tolist()

def calculate_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """
    Calculate cosine similarity between two embeddings
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
        
    Returns:
        float: Cosine similarity score
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for similarity calculation. Please install with 'pip install torch'.")
    
    # Convert to tensors
    tensor1 = torch.tensor(embedding1)
    tensor2 = torch.tensor(embedding2)
    
    # Calculate cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(tensor1.unsqueeze(0), tensor2.unsqueeze(0))
    
    return cos_sim.item()

def create_langchain_embeddings(model_name: str = "all-MiniLM-L6-v2"):
    """
    Create a LangChain embeddings object
    
    Args:
        model_name: Name of the model to use
        
    Returns:
        HuggingFaceEmbeddings: LangChain embeddings object
    """
    if not LANGCHAIN_AVAILABLE:
        raise ImportError("LangChain is required. Please install with 'pip install langchain'.")
    
    return HuggingFaceEmbeddings(model_name=model_name)

def setup_vector_store(documents: List[Dict[str, Any]], embedding_model_name: str = "all-MiniLM-L6-v2", persist_directory: Optional[str] = None):
    """
    Set up a vector store with documents
    
    Args:
        documents: List of documents with content and metadata
        embedding_model_name: Name of the embedding model to use
        persist_directory: Directory to persist the vector store (optional)
        
    Returns:
        The vector store object
    """
    if not LANGCHAIN_AVAILABLE:
        raise ImportError("LangChain is required. Please install with 'pip install langchain'.")
    
    # Create embedding function
    embedding_function = create_langchain_embeddings(model_name=embedding_model_name)
    
    # Convert documents to LangChain format if needed
    from langchain.schema import Document
    langchain_docs = []
    
    for doc in documents:
        langchain_docs.append(
            Document(
                page_content=doc["content"],
                metadata=doc["metadata"]
            )
        )
    
    # Create vector store
    if persist_directory:
        vector_store = Chroma.from_documents(
            documents=langchain_docs,
            embedding=embedding_function,
            persist_directory=persist_directory
        )
        vector_store.persist()
    else:
        vector_store = Chroma.from_documents(
            documents=langchain_docs,
            embedding=embedding_function
        )
    
    return vector_store

def setup_rag_pipeline(vector_store, model_name: str = "google/flan-t5-base", temperature: float = 0.7, max_length: int = 512):
    """
    Set up a RAG pipeline using LangChain
    
    Args:
        vector_store: The vector store to use for retrieval
        model_name: Name of the LLM to use
        temperature: Temperature for generation
        max_length: Maximum output length
        
    Returns:
        The RAG pipeline
    """
    if not LANGCHAIN_AVAILABLE:
        raise ImportError("LangChain is required. Please install with 'pip install langchain'.")
    
    # Set up retriever
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    
    # Set up LLM
    llm = HuggingFaceHub(
        repo_id=model_name,
        model_kwargs={"temperature": temperature, "max_length": max_length}
    )
    
    # Create the QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    
    return qa_chain

def query_rag_system(qa_chain, query: str) -> Dict[str, Any]:
    """
    Query a RAG system and return the response with sources
    
    Args:
        qa_chain: The QA chain to use
        query: The query to process
        
    Returns:
        Dict[str, Any]: Response with answer and source documents
    """
    # Execute the query
    result = qa_chain({"query": query})
    
    # Extract answer and sources
    answer = result["result"]
    source_documents = result["source_documents"]
    
    # Format sources
    sources = []
    for doc in source_documents:
        sources.append({
            "content": doc.page_content,
            "metadata": doc.metadata
        })
    
    return {
        "answer": answer,
        "sources": sources
    }

def stream_text_generation(text: str, delay: float = 0.02) -> None:
    """
    Simulate streaming text generation in Streamlit
    
    Args:
        text: The text to stream
        delay: Delay between characters in seconds
    """
    placeholder = st.empty()
    full_response = ""
    
    for i in range(len(text) + 1):
        full_response = text[:i]
        placeholder.markdown(full_response + "▌" if i < len(text) else full_response)
        time.sleep(delay)

def load_transformer_pipeline(task: str, model_name: str, **kwargs):
    """
    Load a Hugging Face Transformers pipeline
    
    Args:
        task: The task for the pipeline (e.g., "text-generation", "summarization")
        model_name: The model to use
        **kwargs: Additional arguments for the pipeline
        
    Returns:
        The pipeline object
    """
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("Transformers is required. Please install with 'pip install transformers'.")
    
    return pipeline(task=task, model=model_name, **kwargs)

def tokenize_text(text: str, tokenizer_name: str = "gpt2") -> List[str]:
    """
    Tokenize text using a specified tokenizer
    
    Args:
        text: The text to tokenize
        tokenizer_name: Name of the tokenizer to use
        
    Returns:
        List[str]: List of tokens
    """
    if not TRANSFORMERS_AVAILABLE:
        raise ImportError("Transformers is required. Please install with 'pip install transformers'.")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Tokenize
    tokens = tokenizer.tokenize(text)
    
    return tokens

def count_tokens(text: str, tokenizer_name: str = "gpt2") -> int:
    """
    Count the number of tokens in a text
    
    Args:
        text: The text to analyze
        tokenizer_name: Name of the tokenizer to use
        
    Returns:
        int: Token count
    """
    tokens = tokenize_text(text, tokenizer_name)
    return len(tokens)

def display_model_info(model_name: str):
    """
    Display information about a model in Streamlit
    
    Args:
        model_name: Name of the model
    """
    model_info = {
        "gpt2": {
            "parameters": "124 million",
            "context_length": "1024 tokens",
            "training_data": "WebText dataset (8 million documents from the web)",
            "capabilities": "Text generation, reasonable coherence for short texts",
            "limitations": "Limited knowledge, weaker than newer models"
        },
        "google/flan-t5-base": {
            "parameters": "250 million",
            "context_length": "512 tokens",
            "training_data": "Mixture of tasks with instruction tuning",
            "capabilities": "Following instructions, summarization, question answering",
            "limitations": "Less creative than autoregressive models"
        },
        "facebook/bart-large-cnn": {
            "parameters": "400 million",
            "context_length": "1024 tokens",
            "training_data": "Fine-tuned on CNN Daily Mail for summarization",
            "capabilities": "Excellent summarization, good text generation",
            "limitations": "Specialized for news summarization"
        },
        "all-MiniLM-L6-v2": {
            "parameters": "22.7 million",
            "context_length": "256 tokens",
            "training_data": "Trained for generating sentence embeddings",
            "capabilities": "Semantic search, clustering, excellent embedding quality for size",
            "limitations": "Not designed for text generation"
        }
    }
    
    if model_name in model_info:
        info = model_info[model_name]
        
        st.subheader(f"Model: {model_name}")
        st.markdown(f"**Parameters:** {info['parameters']}")
        st.markdown(f"**Context Length:** {info['context_length']}")
        st.markdown(f"**Training Data:** {info['training_data']}")
        st.markdown(f"**Capabilities:** {info['capabilities']}")
        st.markdown(f"**Limitations:** {info['limitations']}")
    else:
        st.write(f"No detailed information available for {model_name}")
