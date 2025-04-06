import streamlit as st
import pandas as pd
import numpy as np
import time
import os
from utils import file_processing, model_utils

def show():
    st.title("🤖 Mini Project - AI Chatbot with RAG")
    
    st.markdown("""
    In this mini-project, we'll build a complete chatbot application that uses Retrieval-Augmented 
    Generation (RAG) to answer questions based on your documents.
    
    The chatbot can:
    - Process PDF and text files
    - Extract and index content using vector embeddings
    - Retrieve relevant information based on your questions
    - Generate accurate, contextual responses
    
    Let's start building!
    """)
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    if "documents_processed" not in st.session_state:
        st.session_state.documents_processed = False
        
    if "document_list" not in st.session_state:
        st.session_state.document_list = []
    
    # Create tabs for different parts of the project
    tabs = st.tabs([
        "1. Upload Documents", 
        "2. Configure Chatbot", 
        "3. Chat Interface",
        "4. Architecture Details"
    ])
    
    # 1. Upload Documents tab
    with tabs[0]:
        st.header("Document Processing")
        
        st.markdown("""
        First, we need to upload and process the documents that our chatbot will use as its knowledge base.
        
        You can upload:
        - PDF files
        - Text (.txt) files
        
        The system will:
        - Extract text content
        - Split into manageable chunks
        - Create vector embeddings
        - Store in a vector database
        """)
        
        # File uploader for multiple files
        uploaded_files = st.file_uploader(
            "Upload your documents (PDF, TXT)", 
            type=["pdf", "txt"],
            accept_multiple_files=True
        )
        
        # Process uploaded files
        if uploaded_files:
            process_button = st.button("Process Documents")
            
            if process_button:
                with st.spinner("Processing documents..."):
                    # Clear previous documents if any
                    if st.session_state.documents_processed:
                        st.session_state.document_list = []
                    
                    # Process each file
                    for file in uploaded_files:
                        # In a real implementation, we would save and process the file here
                        # For this demo, we'll simulate the processing
                        file_name = file.name
                        file_size = file.size
                        file_type = file.type
                        
                        # Add to document list
                        st.session_state.document_list.append({
                            "name": file_name,
                            "size": file_size,
                            "type": file_type
                        })
                        
                        # Simulate processing time
                        time.sleep(1)
                    
                    # Mark as processed
                    st.session_state.documents_processed = True
                
                st.success(f"Successfully processed {len(uploaded_files)} document(s)!")
        
        # Display document list if documents are processed
        if st.session_state.documents_processed and st.session_state.document_list:
            st.subheader("Processed Documents")
            
            # Create a DataFrame for better display
            doc_df = pd.DataFrame(st.session_state.document_list)
            doc_df["size_kb"] = (doc_df["size"] / 1024).round(2)
            doc_df["status"] = "Processed"
            
            # Display the table
            st.table(doc_df[["name", "size_kb", "status"]])
            
            # Clear documents button
            if st.button("Clear All Documents"):
                st.session_state.document_list = []
                st.session_state.documents_processed = False
                st.experimental_rerun()
    
    # 2. Configure Chatbot tab
    with tabs[1]:
        st.header("Chatbot Configuration")
        
        st.markdown("""
        Now let's configure the retrieval and generation parameters for our chatbot.
        
        These settings affect:
        - How many documents are retrieved for each query
        - How the language model generates responses
        - The style and behavior of the chatbot
        """)
        
        # Check if documents have been processed
        if not st.session_state.documents_processed:
            st.warning("Please upload and process documents in the previous tab first!")
        else:
            st.subheader("Retrieval Settings")
            
            col1, col2 = st.columns(2)
            
            with col1:
                retrieval_method = st.selectbox(
                    "Retrieval Method",
                    ["Similarity Search", "MMR (Maximum Marginal Relevance)", "Hybrid Search"]
                )
                
                num_chunks = st.slider(
                    "Number of chunks to retrieve",
                    min_value=1,
                    max_value=10,
                    value=4,
                    help="How many document chunks to retrieve for each query"
                )
            
            with col2:
                if retrieval_method == "MMR (Maximum Marginal Relevance)":
                    diversity = st.slider(
                        "Diversity (Lambda)",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.5,
                        step=0.1,
                        help="0 = maximum diversity, 1 = maximum relevance"
                    )
                
                similarity_threshold = st.slider(
                    "Similarity Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.7,
                    step=0.05,
                    help="Minimum similarity score for retrieved chunks"
                )
            
            st.subheader("Generation Settings")
            
            col3, col4 = st.columns(2)
            
            with col3:
                model = st.selectbox(
                    "Language Model",
                    ["Flan-T5", "GPT-2", "Llama-2-7b"]
                )
                
                temperature = st.slider(
                    "Temperature",
                    min_value=0.1,
                    max_value=1.0,
                    value=0.7,
                    step=0.1,
                    help="Higher values make output more random, lower values more deterministic"
                )
            
            with col4:
                max_tokens = st.slider(
                    "Maximum Response Length",
                    min_value=50,
                    max_value=500,
                    value=256,
                    step=50,
                    help="Maximum number of tokens in the generated response"
                )
                
                stream_output = st.checkbox(
                    "Stream Output",
                    value=True,
                    help="Show response being generated word by word"
                )
            
            st.subheader("Chatbot Behavior")
            
            system_prompt = st.text_area(
                "System Prompt",
                """You are a helpful AI assistant that answers questions based on the provided document context. 
If the answer cannot be found in the documents, politely say you don't have that information. 
Always cite the specific document sources for your information.""",
                height=100,
                help="Instructions that define how the chatbot behaves"
            )
            
            col5, col6 = st.columns(2)
            
            with col5:
                show_sources = st.checkbox(
                    "Show Source Documents",
                    value=True,
                    help="Display the retrieved document chunks used to generate the response"
                )
            
            with col6:
                citation_style = st.selectbox(
                    "Citation Style",
                    ["Inline Citations", "Footnotes", "End of Response"]
                )
            
            # Save configuration button
            if st.button("Save Configuration"):
                # In a real implementation, we would save these settings to session state
                # For this demo, we'll simply acknowledge the action
                st.success("Configuration saved successfully!")
                
                # Store in session state
                st.session_state.chatbot_config = {
                    "retrieval_method": retrieval_method,
                    "num_chunks": num_chunks,
                    "similarity_threshold": similarity_threshold,
                    "model": model,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream_output": stream_output,
                    "system_prompt": system_prompt,
                    "show_sources": show_sources,
                    "citation_style": citation_style
                }
                
                if retrieval_method == "MMR (Maximum Marginal Relevance)":
                    st.session_state.chatbot_config["diversity"] = diversity
    
    # 3. Chat Interface tab
    with tabs[2]:
        st.header("Chat with Your Documents")
        
        # Check prerequisites
        if not st.session_state.documents_processed:
            st.warning("Please upload and process documents in the first tab!")
        elif "chatbot_config" not in st.session_state:
            st.warning("Please configure your chatbot in the previous tab!")
        else:
            st.markdown("""
            Now it's time to interact with your document-aware chatbot!
            
            Ask questions about the content of your uploaded documents, and the chatbot will:
            1. Retrieve relevant information from your documents
            2. Generate a response based on the retrieved context
            3. Provide citations to the source documents
            """)
            
            # Display chat messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
                    
                    # Display sources if this is an assistant message with sources
                    if message["role"] == "assistant" and "sources" in message:
                        with st.expander("View Sources"):
                            for i, source in enumerate(message["sources"]):
                                st.markdown(f"**Source {i+1}**: {source['document']}")
                                st.markdown(f"{source['content']}")
                                st.markdown("---")
            
            # Chat input
            if prompt := st.chat_input("Ask a question about your documents"):
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})
                
                # Display user message
                with st.chat_message("user"):
                    st.write(prompt)
                
                # Generate and display assistant response
                with st.chat_message("assistant"):
                    # Simulate retrieval process
                    with st.spinner("Retrieving relevant information..."):
                        # In a real implementation, this would query the vector database
                        time.sleep(1.5)
                    
                    # Get configuration
                    config = st.session_state.chatbot_config
                    
                    # Simulate response generation with streaming if enabled
                    if config["stream_output"]:
                        response_placeholder = st.empty()
                        full_response = ""
                        
                        # Generate a sample response for demonstration
                        sample_response = """Based on the documents you've provided, artificial intelligence (AI) and machine learning (ML) are transforming industries worldwide through automation, enhanced decision-making, and predictive capabilities.

Key applications include:
1. Predictive maintenance in manufacturing
2. Personalized recommendations in retail and entertainment
3. Fraud detection in financial services
4. Diagnostic assistance in healthcare

The documents highlight that successful AI implementation requires quality data, clear business objectives, and appropriate infrastructure. Companies that strategically integrate AI technologies are seeing significant competitive advantages in operational efficiency and customer experience."""
                        
                        # Simulate streaming
                        for i in range(len(sample_response) + 1):
                            time.sleep(0.01)  # Adjust speed
                            full_response = sample_response[:i]
                            response_placeholder.markdown(full_response + "▌" if i < len(sample_response) else full_response)
                        
                        response = full_response
                    else:
                        # Generate response without streaming
                        with st.spinner("Generating response..."):
                            time.sleep(2)
                            response = """Based on the documents you've provided, artificial intelligence (AI) and machine learning (ML) are transforming industries worldwide through automation, enhanced decision-making, and predictive capabilities.

Key applications include:
1. Predictive maintenance in manufacturing
2. Personalized recommendations in retail and entertainment
3. Fraud detection in financial services
4. Diagnostic assistance in healthcare

The documents highlight that successful AI implementation requires quality data, clear business objectives, and appropriate infrastructure. Companies that strategically integrate AI technologies are seeing significant competitive advantages in operational efficiency and customer experience."""
                            st.write(response)
                    
                    # Show sources if enabled
                    if config["show_sources"]:
                        # Generate sample sources
                        sources = [
                            {
                                "document": f"{st.session_state.document_list[0]['name']}, page 3",
                                "content": "Artificial intelligence and machine learning are revolutionizing industries worldwide. From predictive maintenance in manufacturing to personalized recommendations in retail, these technologies are creating new opportunities for efficiency and innovation."
                            },
                            {
                                "document": f"{st.session_state.document_list[0]['name']}, page 7",
                                "content": "Successful AI implementation requires three key elements: quality data for training models, clear business objectives aligned with organizational goals, and appropriate infrastructure to deploy and scale solutions."
                            }
                        ]
                        
                        with st.expander("View Sources"):
                            for i, source in enumerate(sources):
                                st.markdown(f"**Source {i+1}**: {source['document']}")
                                st.markdown(f"{source['content']}")
                                st.markdown("---")
                    
                    # Add assistant message to chat history
                    assistant_message = {
                        "role": "assistant",
                        "content": response
                    }
                    
                    if config["show_sources"]:
                        assistant_message["sources"] = sources
                    
                    st.session_state.messages.append(assistant_message)
    
    # 4. Architecture Details tab
    with tabs[3]:
        st.header("Chatbot Architecture")
        
        st.markdown("""
        This section explains how the chatbot works under the hood, so you can understand 
        the components and potentially customize it for your own projects.
        
        ### Overall Architecture
        
        Our chatbot follows a RAG (Retrieval-Augmented Generation) architecture with these main components:
        """)
        
        # Create columns for the components
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("1. Document Processing")
            st.markdown("""
            - **Document Loading**: Extract text from PDFs and text files
            - **Text Chunking**: Split documents into manageable pieces
            - **Embedding**: Convert text chunks to vector embeddings
            - **Indexing**: Store embeddings in a vector database (ChromaDB)
            """)
        
        with col2:
            st.subheader("2. Retrieval System")
            st.markdown("""
            - **Query Processing**: Embed user questions
            - **Semantic Search**: Find relevant chunks using vector similarity
            - **Context Assembly**: Combine retrieved chunks
            - **Source Tracking**: Maintain provenance information
            """)
        
        with col3:
            st.subheader("3. Generation System")
            st.markdown("""
            - **Prompt Construction**: Create effective prompts with context
            - **LLM Integration**: Send to language model
            - **Response Formatting**: Structure the generated text
            - **Citation Management**: Add source references
            """)
        
        # Architecture diagram (text-based for simplicity)
        st.markdown("### Architecture Diagram")
        
        st.code("""
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  Document Store  │     │   User Interface  │     │  Language Model  │
└────────┬─────────┘     └────────┬─────────┘     └────────┬─────────┘
         │                        │                        │
         ▼                        ▼                        ▼
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│ Document         │     │ Query            │     │ Prompt           │
│ Processing       │     │ Processing       │     │ Construction     │
└────────┬─────────┘     └────────┬─────────┘     └────────┬─────────┘
         │                        │                        │
         ▼                        ▼                        ▼
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│ Vector           │     │ Retrieval        │     │ Response         │
│ Database         │◄────┤ Engine           │────►│ Generation       │
└──────────────────┘     └──────────────────┘     └──────────────────┘
                                  │                        │
                                  └────────────┬───────────┘
                                               ▼
                                    ┌──────────────────┐
                                    │     Response     │
                                    │   with Sources   │
                                    └──────────────────┘
        """)
        
        st.markdown("---")
        
        st.subheader("Key Code Components")
        
        # Show some example code snippets
        with st.expander("Document Processor"):
            st.code("""
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

class DocumentProcessor:
    def __init__(self, embedding_model_name="all-MiniLM-L6-v2"):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': 'cpu'}
        )
        self.vector_store = None
    
    def load_document(self, file_path):
        """Load a document from file path"""
        if file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith('.txt'):
            loader = TextLoader(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
            
        return loader.load()
    
    def process_documents(self, documents):
        """Process documents into chunks and create embeddings"""
        # Split into chunks
        chunks = self.text_splitter.split_documents(documents)
        
        # Create or update vector store
        if self.vector_store is None:
            self.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embedding_model
            )
        else:
            self.vector_store.add_documents(chunks)
        
        return len(chunks)
""")
        
        with st.expander("RAG Chatbot"):
            st.code("""
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFaceHub

class RAGChatbot:
    def __init__(self, vector_store, model_name="google/flan-t5-base"):
        self.vector_store = vector_store
        self.llm = HuggingFaceHub(
            repo_id=model_name,
            model_kwargs={"temperature": 0.7, "max_length": 512}
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Create retriever
        self.retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        
        # Create chain
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory,
            return_source_documents=True
        )
    
    def ask(self, question):
        """Ask a question and get response with sources"""
        result = self.chain({"question": question})
        
        response = result["answer"]
        source_documents = result["source_documents"]
        
        # Format sources
        sources = []
        for doc in source_documents:
            sources.append({
                "document": doc.metadata.get("source", "Unknown"),
                "content": doc.page_content
            })
        
        return response, sources
""")
        
        st.markdown("---")
        
        st.subheader("Extending the Chatbot")
        
        st.markdown("""
        Here are some ways you could extend this chatbot for your own projects:
        
        1. **Multi-Modal Support**: Add image and table extraction from documents
        
        2. **Advanced Retrieval**: Implement hybrid search or query rewriting
        
        3. **UI Enhancements**: Add visualization of document relationships
        
        4. **Performance Optimization**: Implement caching and batch processing
        
        5. **Enterprise Features**:
           - User authentication and document permissions
           - Chat history management
           - Custom knowledge bases per user/team
        
        6. **Evaluation Framework**: Add tools to measure response quality and retrieval effectiveness
        
        The modular architecture makes it easy to enhance specific components without
        redesigning the entire system.
        """)
    
    st.markdown("---")
    st.info("This module demonstrated a complete RAG-powered chatbot. You can use this as a starting point for building your own AI applications with similar capabilities.")

