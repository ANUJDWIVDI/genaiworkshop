import os
import json
import streamlit as st
from openai import OpenAI

def initialize_openai_client(user_id=None):
    """
    Initialize OpenAI client with API key
    
    Args:
        user_id: Optional user ID to use for retrieving API key from database
        
    Returns:
        OpenAI client if key is available, None otherwise
    """
    # Try to get API key from various sources
    api_key = None
    
    # First try environment variable
    api_key = os.environ.get("OPENAI_API_KEY")
    
    # Then try session state (for this session only)
    if not api_key and "openai_api_key" in st.session_state:
        api_key = st.session_state.openai_api_key
    
    # Finally try database if user_id is provided
    if not api_key and user_id:
        from utils.db_utils import get_api_key
        api_key = get_api_key(user_id, "openai")
    
    if api_key:
        try:
            return OpenAI(api_key=api_key)
        except Exception as e:
            st.error(f"Error initializing OpenAI client: {str(e)}")
            return None
    else:
        return None

def get_openai_models(client=None):
    """
    Get available OpenAI models
    
    Args:
        client: OpenAI client instance
        
    Returns:
        List of model IDs
    """
    if not client:
        return []
    
    try:
        response = client.models.list()
        return [model.id for model in response.data]
    except Exception as e:
        st.error(f"Error fetching OpenAI models: {str(e)}")
        return []

def summarize_text(text, client=None, model="gpt-4o", max_tokens=500):
    """
    Summarize text using OpenAI
    
    Args:
        text: Text to summarize
        client: OpenAI client instance
        model: Model to use for summarization
        max_tokens: Maximum tokens in the response
        
    Returns:
        Summary text or error message
    """
    # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
    # do not change this unless explicitly requested by the user
    
    if not client:
        return "OpenAI API key not configured. Please add your API key in the settings."
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes information concisely."},
                {"role": "user", "content": f"Please summarize the following text in a professional style. Focus on the key points and maintain context:\n\n{text}"}
            ],
            max_tokens=max_tokens,
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error during summarization: {str(e)}"

def analyze_sentiment(text, client=None, model="gpt-4o"):
    """
    Analyze sentiment of text using OpenAI
    
    Args:
        text: Text to analyze
        client: OpenAI client instance
        model: Model to use for analysis
        
    Returns:
        Dictionary with sentiment analysis or error message
    """
    # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
    # do not change this unless explicitly requested by the user
    
    if not client:
        return {"error": "OpenAI API key not configured. Please add your API key in the settings."}
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a sentiment analysis expert. "
                    + "Analyze the sentiment of the text and provide a rating "
                    + "from 1 to 5 stars and a confidence score between 0 and 1. "
                    + "Respond with JSON in this format: "
                    + "{'rating': number, 'confidence': number, 'sentiment': string, 'explanation': string}"
                },
                {"role": "user", "content": text},
            ],
            response_format={"type": "json_object"},
            temperature=0.3
        )
        result = json.loads(response.choices[0].message.content)
        return {
            "rating": max(1, min(5, round(result.get("rating", 3)))),
            "confidence": max(0, min(1, result.get("confidence", 0.5))),
            "sentiment": result.get("sentiment", "neutral"),
            "explanation": result.get("explanation", "No explanation provided")
        }
    except Exception as e:
        return {"error": f"Error during sentiment analysis: {str(e)}"}

def generate_rag_response(query, context, client=None, model="gpt-4o", max_tokens=1000):
    """
    Generate response using RAG (Retrieval-Augmented Generation) approach
    
    Args:
        query: User query
        context: Retrieved context passages
        client: OpenAI client instance
        model: Model to use for response generation
        max_tokens: Maximum tokens in the response
        
    Returns:
        Generated response or error message
    """
    # the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
    # do not change this unless explicitly requested by the user
    
    if not client:
        return "OpenAI API key not configured. Please add your API key in the settings."
    
    context_text = "\n\n".join([f"Context {i+1}:\n{passage}" for i, passage in enumerate(context)])
    
    system_prompt = """You are a helpful AI assistant that answers questions based on the provided context. 
Follow these rules:
1. Base your answer only on the provided context
2. If the context doesn't contain the answer, say "I don't have enough information to answer this question."
3. Do not make up information
4. Provide clear, concise, and accurate responses
5. Cite the specific context used in your answer (e.g., "According to Context 3...")"""
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context information:\n{context_text}\n\nQuestion: {query}"}
            ],
            max_tokens=max_tokens,
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating response: {str(e)}"

def check_openai_connection(client=None):
    """
    Check if OpenAI connection is working
    
    Args:
        client: OpenAI client instance
        
    Returns:
        Boolean indicating if connection is working, and message
    """
    if not client:
        return False, "OpenAI client not initialized. Please check your API key."
    
    try:
        # Try a simple completion as a connection test
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "This is a connection test."},
                {"role": "user", "content": "Respond with 'Connection successful' if you receive this message."}
            ],
            max_tokens=20,
            temperature=0
        )
        return True, "Connection successful"
    except Exception as e:
        return False, f"Connection failed: {str(e)}"

def display_openai_interface(user_id=None):
    """
    Display OpenAI interface in Streamlit
    
    Args:
        user_id: User ID for retrieving API key
    """
    st.title("OpenAI Integration")
    
    # Initialize client
    client = initialize_openai_client(user_id)
    
    # Check if API key is configured
    if not client:
        st.warning("OpenAI API key not configured. Please add your API key in the settings.")
        
        # Option to temporarily set API key for this session
        with st.expander("Temporarily set API key for this session"):
            temp_key = st.text_input("OpenAI API key", type="password", key="temp_openai_key")
            if st.button("Set temporary key") and temp_key:
                st.session_state.openai_api_key = temp_key
                st.success("API key set for this session. Please refresh this page.")
                st.rerun()
        
        return
    
    # Display connection status
    connection_ok, message = check_openai_connection(client)
    
    if connection_ok:
        st.success("✅ OpenAI API connection is working")
    else:
        st.error(f"❌ OpenAI API connection failed: {message}")
        return
    
    # Display available models
    models = get_openai_models(client)
    if models:
        with st.expander("Available Models"):
            st.write(models)
    
    # Create tabs for different OpenAI features
    tab1, tab2, tab3 = st.tabs(["Text Generation", "Summarization", "Sentiment Analysis"])
    
    with tab1:
        st.header("Text Generation")
        
        # Model selection
        model = st.selectbox(
            "Select model",
            options=["gpt-4o", "gpt-3.5-turbo"] + ([m for m in models if "gpt" in m.lower()] if models else []),
            index=0
        )
        
        # Input text
        prompt = st.text_area("Enter your prompt", height=150)
        
        # Parameters
        col1, col2, col3 = st.columns(3)
        with col1:
            temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
        with col2:
            max_tokens = st.slider("Max tokens", min_value=50, max_value=4000, value=1000, step=50)
        with col3:
            top_p = st.slider("Top P", min_value=0.1, max_value=1.0, value=1.0, step=0.1)
        
        # Generate button
        if st.button("Generate text", key="generate_text_btn"):
            if prompt:
                with st.spinner("Generating text..."):
                    try:
                        response = client.chat.completions.create(
                            model=model,
                            messages=[{"role": "user", "content": prompt}],
                            max_tokens=max_tokens,
                            temperature=temperature,
                            top_p=top_p
                        )
                        st.markdown("### Response")
                        st.write(response.choices[0].message.content)
                        
                        # Display token usage
                        st.info(f"Token usage: {response.usage.prompt_tokens} (prompt) + "
                                f"{response.usage.completion_tokens} (completion) = "
                                f"{response.usage.total_tokens} (total)")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
            else:
                st.warning("Please enter a prompt")
    
    with tab2:
        st.header("Text Summarization")
        
        # Input text
        text_to_summarize = st.text_area("Enter text to summarize", height=200, key="summarize_text")
        
        # Parameters
        col1, col2 = st.columns(2)
        with col1:
            summary_model = st.selectbox(
                "Select model",
                options=["gpt-4o", "gpt-3.5-turbo"],
                index=0,
                key="summary_model"
            )
        with col2:
            summary_length = st.slider("Summary length (tokens)", min_value=50, max_value=1000, value=200, step=50)
        
        # Generate button
        if st.button("Generate summary", key="summarize_btn"):
            if text_to_summarize:
                with st.spinner("Generating summary..."):
                    summary = summarize_text(text_to_summarize, client, summary_model, summary_length)
                    st.markdown("### Summary")
                    st.write(summary)
            else:
                st.warning("Please enter text to summarize")
    
    with tab3:
        st.header("Sentiment Analysis")
        
        # Input text
        text_to_analyze = st.text_area("Enter text to analyze", height=150, key="analyze_text")
        
        # Generate button
        if st.button("Analyze sentiment", key="analyze_btn"):
            if text_to_analyze:
                with st.spinner("Analyzing sentiment..."):
                    result = analyze_sentiment(text_to_analyze, client)
                    
                    if "error" in result:
                        st.error(result["error"])
                    else:
                        # Display results
                        st.markdown("### Sentiment Analysis Results")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Rating", f"{result['rating']}★")
                        with col2:
                            st.metric("Confidence", f"{result['confidence']:.2f}")
                        
                        st.write(f"**Sentiment**: {result['sentiment']}")
                        st.write(f"**Explanation**: {result['explanation']}")
            else:
                st.warning("Please enter text to analyze")