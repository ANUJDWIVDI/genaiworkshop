import streamlit as st
import pandas as pd
import numpy as np
import os
import time
from utils import file_processing, model_utils, openai_utils
from utils.api_key_utils import check_openai_api_key

def show():
    st.title("📄 RAG (Retrieval-Augmented Generation) Implementation")
    
    st.markdown("""
    This module introduces Retrieval-Augmented Generation (RAG), a technique that enhances LLM outputs 
    by retrieving relevant information from external sources before generating responses.
    
    RAG addresses a key limitation of LLMs: their knowledge is limited to what they learned during 
    training. By combining retrieval systems with generation capabilities, RAG enables models to 
    access and use up-to-date or domain-specific information.
    """)
    
    # Create tabs for different topics
    tabs = st.tabs([
        "RAG Fundamentals", 
        "Vector Databases", 
        "Retrieval Strategies", 
        "Building a RAG Pipeline",
        "RAG Evaluation"
    ])
    
    # RAG Fundamentals tab
    with tabs[0]:
        st.header("RAG Fundamentals")
        
        st.markdown("""
        ### What is RAG?
        
        Retrieval-Augmented Generation (RAG) is a hybrid approach that combines:
        
        1. **Retrieval**: Finding relevant information from a knowledge base
        2. **Generation**: Using that information to generate accurate, contextual responses
        
        ### Why Use RAG?
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Challenges with LLMs Alone")
            st.markdown("""
            - **Knowledge Cutoff**: LLMs only know facts up to their training cutoff
            - **Hallucinations**: Can generate plausible but incorrect information
            - **Domain Limitations**: May lack specialized knowledge
            - **Black Box**: Hard to trace source of information
            - **Costly Updates**: Retraining for new knowledge is expensive
            """)
        
        with col2:
            st.subheader("Benefits of RAG")
            st.markdown("""
            - **Current Information**: Can use up-to-date knowledge
            - **Factual Grounding**: Reduces hallucinations
            - **Domain Adaptation**: Easily incorporate specialized documents
            - **Transparency**: Can cite sources of information
            - **Cost Efficiency**: Update knowledge without retraining
            """)
        
        st.markdown("---")
        
        st.subheader("The RAG Architecture")
        
        st.markdown("""
        A typical RAG system consists of these components:
        
        1. **Document Processor**: Converts documents into a suitable format
           - Chunking documents into manageable pieces
           - Cleaning and normalizing text
           - Extracting metadata
        
        2. **Embedding Model**: Converts text into numerical vectors
           - Captures semantic meaning in high-dimensional space
           - Enables similarity search
        
        3. **Vector Database**: Stores and indexes document vectors
           - Efficiently performs similarity searches
           - Scales to large document collections
        
        4. **Retriever**: Finds relevant documents for a query
           - Semantic search
           - Keyword-based search
           - Hybrid approaches
        
        5. **Generator**: Creates a response using retrieved information
           - Incorporates context from retrieved documents
           - Synthesizes information to answer the query
           - Cites sources when appropriate
        """)
        
        st.image("https://raw.githubusercontent.com/langchain-ai/langchain/master/docs/static/img/rag_indexing.jpg", 
                caption="RAG Architecture - Indexing Phase (Source: LangChain)")
        
        st.image("https://raw.githubusercontent.com/langchain-ai/langchain/master/docs/static/img/rag.jpg", 
                caption="RAG Architecture - Query Phase (Source: LangChain)")
        
        st.markdown("---")
        
        st.subheader("RAG vs. Other Approaches")
        
        comparison = {
            "Approach": ["Standard LLM", "RAG (Retrieval-Augmented Generation)", "Fine-tuning", "Embedding-only Retrieval"],
            "Knowledge Source": ["Pre-training only", "Pre-training + External documents", "Pre-training + Training data", "External documents only"],
            "Query Processing": ["Direct prompt to model", "Retrieve + Prompt with context", "Direct prompt to specialized model", "Retrieve similar documents"],
            "Response Generation": ["Generated from model parameters", "Generated from model + retrieved context", "Generated from adapted model", "No generation, just retrieval"],
            "Strengths": [
                "Fast, no external dependencies",
                "Current information, verifiable sources",
                "Specialized domain expertise",
                "Simple, fast document lookup"
            ],
            "Limitations": [
                "Outdated knowledge, hallucinations",
                "Complexity, retrieval quality dependency",
                "Fixed knowledge, expensive to update",
                "No synthesis, requires exact matches"
            ]
        }
        
        comparison_df = pd.DataFrame(comparison)
        st.table(comparison_df)
    
    # Vector Databases tab
    with tabs[1]:
        st.header("Vector Databases")
        
        st.markdown("""
        Vector databases are specialized for storing and querying vector embeddings. They enable 
        efficient similarity searches in high-dimensional spaces, making them ideal for RAG systems.
        
        ### What are Embeddings?
        
        Embeddings are dense vector representations of text (or other data) that capture semantic meaning. 
        Similar concepts have similar vector representations, enabling similarity-based search.
        """)
        
        # Embeddings visualization (simplified 2D representation)
        st.subheader("Simplified Embedding Space Visualization")
        
        # Create some sample 2D embeddings for visualization
        np.random.seed(42)
        categories = ["AI", "Sports", "Finance", "Health", "Travel"]
        colors = ["#ff2b2b", "#2b7fff", "#2bff3e", "#ff2bef", "#ffbc2b"]
        
        points = []
        
        for i, category in enumerate(categories):
            # Generate 5 points per category in a cluster
            cluster_center = np.random.rand(2) * 8
            cluster_points = cluster_center + np.random.randn(5, 2) * 0.5
            
            for point in cluster_points:
                points.append({
                    "x": point[0],
                    "y": point[1],
                    "category": category,
                    "color": colors[i]
                })
        
        # Convert to DataFrame for easier plotting
        df = pd.DataFrame(points)
        
        # Plotting
        chart_data = pd.DataFrame({
            'x': df['x'],
            'y': df['y'],
            'category': df['category']
        })
        
        st.vega_lite_chart(chart_data, {
            'mark': {'type': 'circle', 'size': 100},
            'encoding': {
                'x': {'field': 'x', 'type': 'quantitative'},
                'y': {'field': 'y', 'type': 'quantitative'},
                'color': {'field': 'category', 'type': 'nominal'},
                'tooltip': [
                    {'field': 'category', 'type': 'nominal'}
                ]
            },
            'width': 500,
            'height': 400
        })
        
        st.caption("Simplified 2D visualization of embeddings. In reality, embeddings typically have hundreds or thousands of dimensions.")
        
        st.markdown("""
        In this simplified visualization, each point represents a document or chunk of text. 
        Points that are close together are semantically similar, while distant points are unrelated.
        
        When a query is embedded into the same space, we can find the most relevant documents 
        by identifying the closest points to the query embedding.
        """)
        
        st.markdown("---")
        
        st.subheader("Popular Vector Databases")
        
        vector_dbs = {
            "ChromaDB": {
                "description": "Open-source, lightweight embedding database that runs locally or in the cloud",
                "best_for": "Development, small to medium projects, getting started with RAG",
                "features": "In-memory or persistent storage, simple API, supports hybrid search",
                "limitations": "Less scalable for very large datasets"
            },
            "FAISS (Facebook AI Similarity Search)": {
                "description": "Library for efficient similarity search and clustering of dense vectors",
                "best_for": "High-performance vector search, large-scale applications",
                "features": "Highly optimized, supports GPU acceleration, various indexing methods",
                "limitations": "More complex API, not a full database solution"
            },
            "Pinecone": {
                "description": "Managed vector database service optimized for similarity search",
                "best_for": "Production deployments, serverless architecture",
                "features": "Fully managed, horizontal scaling, real-time updates",
                "limitations": "Paid service, less control over infrastructure"
            },
            "Weaviate": {
                "description": "Open-source vector search engine with GraphQL API",
                "best_for": "Projects needing both vector and traditional search capabilities",
                "features": "GraphQL interface, multi-modal, hybrid search",
                "limitations": "More complex setup, steeper learning curve"
            },
            "Milvus": {
                "description": "Open-source vector database designed for scalability",
                "best_for": "Enterprise applications, high throughput requirements",
                "features": "Horizontal scaling, cloud-native, support for complex queries",
                "limitations": "Requires more infrastructure knowledge to set up and maintain"
            }
        }
        
        selected_db = st.selectbox("Select a vector database to learn more", list(vector_dbs.keys()))
        
        db_info = vector_dbs[selected_db]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**{selected_db}**")
            st.markdown(db_info["description"])
            st.markdown("**Best For:**")
            st.markdown(db_info["best_for"])
        
        with col2:
            st.markdown("**Key Features:**")
            st.markdown(db_info["features"])
            st.markdown("**Limitations:**")
            st.markdown(db_info["limitations"])
        
        st.markdown("---")
        
        st.subheader("ChromaDB Example")
        
        st.markdown("""
        ChromaDB is one of the easiest vector databases to get started with. Here's an example of how to use it:
        """)
        
        st.code("""
import chromadb
from chromadb.utils import embedding_functions

# Initialize ChromaDB client
client = chromadb.Client()

# Create a collection with a specific embedding function
embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = client.create_collection(
    name="my_documents",
    embedding_function=embedding_function
)

# Add documents
documents = [
    "Artificial intelligence is transforming industries worldwide.",
    "Machine learning models require large amounts of quality data.",
    "Vector databases store embeddings for similarity search.",
    "Neural networks are inspired by the human brain."
]

collection.add(
    documents=documents,
    ids=["doc1", "doc2", "doc3", "doc4"]
)

# Query the collection
results = collection.query(
    query_texts=["How do AI systems learn?"],
    n_results=2
)

print("Query results:", results)
        """)
        
        st.markdown("""
        ### Vector Database Considerations
        
        When choosing a vector database for your RAG system, consider:
        
        1. **Scale**: How many documents/vectors will you store?
        2. **Performance**: What query latency is acceptable?
        3. **Deployment**: Local, cloud, or hybrid?
        4. **Maintenance**: Managed service or self-hosted?
        5. **Indexing**: What vector index types are supported?
        6. **Metadata Filtering**: Can you filter by metadata in addition to vector similarity?
        7. **Cost**: What are the operational and licensing costs?
        """)
    
    # Retrieval Strategies tab
    with tabs[2]:
        st.header("Retrieval Strategies")
        
        st.markdown("""
        The retrieval component of RAG is critical for finding the most relevant information. 
        Different retrieval strategies offer tradeoffs between precision, recall, and performance.
        
        ### Common Retrieval Methods
        """)
        
        retrieval_methods = {
            "Similarity Search": {
                "description": "Find documents with embeddings closest to the query embedding",
                "how_it_works": "Convert query to vector, calculate distance to all document vectors, return closest matches",
                "advantages": "Captures semantic meaning, works well for conceptual matching",
                "limitations": "May miss keyword matches, depends on embedding quality"
            },
            "Keyword Search": {
                "description": "Find documents containing specific words from the query",
                "how_it_works": "Index terms in documents, match query terms to document terms using methods like BM25",
                "advantages": "Works well for explicit term matches, well-established techniques",
                "limitations": "Misses semantic relationships, synonym problems"
            },
            "Hybrid Search": {
                "description": "Combine similarity and keyword search for better results",
                "how_it_works": "Run both methods and combine scores, or use one method to rerank results from the other",
                "advantages": "Balances semantic and lexical matching, typically better performance",
                "limitations": "More complex to implement, requires parameter tuning"
            },
            "Dense Passage Retrieval": {
                "description": "Use specialized models trained specifically for retrieval tasks",
                "how_it_works": "Train encoder models to bring relevant query-passage pairs closer in vector space",
                "advantages": "Better retrieval performance than general embeddings, task-specific",
                "limitations": "Requires training/fine-tuning, more compute intensive"
            },
            "Multi-Query Retrieval": {
                "description": "Generate multiple query variants to increase recall",
                "how_it_works": "Use LLM to rephrase original query into several alternatives, retrieve for each",
                "advantages": "Improves recall by exploring different query formulations",
                "limitations": "More expensive (multiple retrievals), requires result aggregation"
            }
        }
        
        method = st.selectbox("Select a retrieval method to explore", list(retrieval_methods.keys()))
        
        method_info = retrieval_methods[method]
        
        st.markdown(f"**{method}**")
        st.markdown(method_info["description"])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**How It Works:**")
            st.markdown(method_info["how_it_works"])
            st.markdown("**Advantages:**")
            st.markdown(method_info["advantages"])
        
        with col2:
            st.markdown("**Limitations:**")
            st.markdown(method_info["limitations"])
        
        st.markdown("---")
        
        st.subheader("Advanced Retrieval Techniques")
        
        st.markdown("""
        Beyond basic retrieval methods, these advanced techniques can improve RAG performance:
        
        ### 1. Chunking Strategies
        
        How you divide documents impacts retrieval quality:
        
        - **Fixed Size Chunks**: Simple but may break conceptual units
        - **Semantic Chunks**: Preserve meaning by splitting at logical boundaries
        - **Overlapping Chunks**: Ensure context isn't lost at chunk boundaries
        - **Hierarchical Chunks**: Store multiple granularities (paragraphs, sections, documents)
        
        ### 2. Re-ranking
        
        Apply a second evaluation to initial search results:
        
        - **Cross-Encoders**: Use more powerful models to score query-document pairs
        - **LLM Re-ranking**: Have the LLM itself evaluate relevance of retrieved passages
        - **Fusion Methods**: Combine multiple ranking signals (relevance, recency, authority)
        
        ### 3. Query Transformation
        
        Modify the query to improve retrieval:
        
        - **Query Expansion**: Add related terms to the query
        - **HyDE (Hypothetical Document Embeddings)**: Generate a hypothetical answer first, then retrieve
        - **Query Decomposition**: Break complex queries into simpler sub-queries
        
        ### 4. Contextual Compression
        
        Reduce retrieved context to the most relevant parts:
        
        - **Document Compression**: Extract only relevant sentences from retrieved documents
        - **Adaptive Retrieval**: Dynamically determine how much context to retrieve
        """)
        
        st.image("https://blog.langchain.dev/content/images/size/w1600/2023/11/query_transformations.svg", 
                caption="Query Transformation Techniques (Source: LangChain)")
    
    # Building a RAG Pipeline tab
    with tabs[3]:
        st.header("Building a RAG Pipeline")
        
        st.markdown("""
        Let's walk through the process of building a RAG pipeline using LangChain and ChromaDB.
        
        ### 1. Document Processing
        
        First, we need to process documents, chunk them appropriately, and prepare them for embedding.
        """)
        
        st.code("""
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load documents
def load_document(file_path):
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith('.txt'):
        loader = TextLoader(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
        
    documents = loader.load()
    return documents

# Split documents into chunks
def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    
    chunks = text_splitter.split_documents(documents)
    return chunks

# Process a single file
file_path = "path/to/document.pdf"
documents = load_document(file_path)
chunks = split_documents(documents)

print(f"Loaded {len(documents)} document(s), split into {len(chunks)} chunks")
        """)
        
        st.markdown("""
        ### 2. Creating Embeddings and Vector Store
        
        Next, we'll embed our document chunks and store them in ChromaDB.
        """)
        
        st.code("""
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

# Create vector store
def create_vector_store(chunks, embedding_model, persist_directory=None):
    if persist_directory:
        # Create persistent vector store
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=persist_directory
        )
        vector_store.persist()
    else:
        # Create in-memory vector store
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model
        )
    
    return vector_store

# Store our chunks
vector_store = create_vector_store(chunks, embedding_model, "chroma_db")
        """)
        
        st.markdown("""
        ### 3. Setting Up the Retriever
        
        Now we'll create a retriever that can find relevant documents for a query.
        """)
        
        st.code("""
# Create a retriever
retriever = vector_store.as_retriever(
    search_type="similarity",  # Options: similarity, mmr
    search_kwargs={"k": 4}     # Return top 4 most relevant chunks
)

# Test the retriever
query = "What are the main benefits of RAG systems?"
retrieved_docs = retriever.get_relevant_documents(query)

print(f"Retrieved {len(retrieved_docs)} documents for query: '{query}'")
for i, doc in enumerate(retrieved_docs):
    print(f"Document {i+1}:")
    print(f"Source: {doc.metadata.get('source', 'Unknown')}")
    print(f"Content: {doc.page_content[:200]}...")
    print("-" * 50)
        """)
        
        st.markdown("""
        ### 4. Creating the RAG Chain
        
        Finally, we'll connect everything to create our RAG pipeline using LangChain.
        """)
        
        st.code("""
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub

# Initialize LLM
llm = HuggingFaceHub(
    repo_id="google/flan-t5-base",  # You can replace with stronger models
    model_kwargs={"temperature": 0.5, "max_length": 512}
)

# Create the RAG chain
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # Options: stuff, map_reduce, refine
    retriever=retriever,
    return_source_documents=True,
    verbose=True
)

# Query the RAG system
response = rag_chain({"query": "What are the main benefits of RAG systems?"})

print("Generated Answer:")
print(response["result"])
print("\\nSource Documents:")
for i, doc in enumerate(response["source_documents"]):
    print(f"Document {i+1} from {doc.metadata.get('source', 'Unknown')}")
        """)
        
        st.markdown("""
        ### RAG Pipeline Using LangChain's LCEL (LangChain Expression Language)
        
        LangChain also provides a more modular approach using LCEL:
        """)
        
        st.code("""
from langchain.schema.runnable import RunnablePassthrough
from langchain.prompts import PromptTemplate

# Define the prompt template
prompt_template = """
You are an AI assistant providing accurate information based on the given context.
Answer the question based only on the provided context. If the context doesn't 
contain the answer, say "I don't have enough information to answer this question."

Context:
{context}

Question:
{question}

Answer:
"""

prompt = PromptTemplate.from_template(prompt_template)

# Define a function to format documents
def format_docs(docs):
    return "\\n\\n".join([d.page_content for d in docs])

# Create the RAG chain using LCEL
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
)

# Query the system
response = rag_chain.invoke("What are the main benefits of RAG systems?")
print(response)
        """)
        
        st.markdown("---")
        
        st.subheader("Interactive RAG Pipeline Demo")
        st.markdown("Upload a document and ask questions about it to see a RAG pipeline in action.")
        
        # Check if user has OpenAI API key
        user_id = None
        if "user" in st.session_state:
            user_id = st.session_state.user["id"]
        
        has_openai_key = check_openai_api_key(user_id)
        
        if not has_openai_key:
            st.warning("⚠️ OpenAI API key not configured. For the best experience with this demo, please add your OpenAI API key in the API Keys section.")
            st.info("Without an OpenAI API key, the demo will run with simulated responses.")
            
            if st.button("Configure OpenAI API Key"):
                st.session_state.current_page = "API Keys"
                st.rerun()
        
        # Model selection for those with API key
        model_option = "Simulated"
        if has_openai_key:
            model_options = ["OpenAI (GPT-4o)", "OpenAI (GPT-3.5 Turbo)", "Simulated"]
            model_option = st.selectbox("Select LLM for Generation", model_options)
        
        # File uploader
        uploaded_file = st.file_uploader("Upload a document (PDF or TXT)", type=["pdf", "txt"])
        
        if uploaded_file:
            # Save file and process
            with st.spinner("Processing document..."):
                # Save uploaded file
                file_path = file_processing.save_uploaded_file(uploaded_file)
                
                if "documents" not in st.session_state:
                    st.session_state.documents = {}
                
                # Process file if not already processed
                file_key = f"{uploaded_file.name}_{hash(file_path)}"
                
                if file_key not in st.session_state.documents:
                    try:
                        # Extract text
                        document_text = file_processing.extract_text(file_path)
                        
                        # Create chunks
                        chunks = file_processing.chunk_text(document_text)
                        
                        # Store for later use
                        st.session_state.documents[file_key] = {
                            "text": document_text,
                            "chunks": chunks,
                            "file_path": file_path,
                            "file_name": uploaded_file.name
                        }
                        
                        st.success(f"Document '{uploaded_file.name}' processed successfully!")
                    except Exception as e:
                        st.error(f"Error processing document: {str(e)}")
            
            if file_key in st.session_state.documents:
                document_data = st.session_state.documents[file_key]
                
                # Show document stats
                st.markdown("### Document Information")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Document Size", f"{len(document_data['text'])} chars")
                with col2:
                    st.metric("Chunks", f"{len(document_data['chunks'])}")
                with col3:
                    st.metric("Avg. Chunk Size", f"{len(document_data['text']) // max(1, len(document_data['chunks']))} chars")
                
                # Document preview
                with st.expander("Document Preview"):
                    st.markdown(document_data['text'][:1000] + "..." if len(document_data['text']) > 1000 else document_data['text'])
                
                # Ask questions section
                st.markdown("### Ask Questions About Your Document")
                query = st.text_input("Enter your question")
                
                if st.button("Submit Question") and query:
                    with st.spinner("Retrieving relevant information and generating answer..."):
                        chunks = document_data["chunks"]
                        
                        # Step 1: Find relevant chunks using embeddings
                        try:
                            # Load embedding model
                            embedding_model = model_utils.load_embedding_model()
                            
                            # Generate embeddings for chunks and query
                            chunk_embeddings = model_utils.generate_embeddings(embedding_model, chunks)
                            query_embedding = model_utils.generate_embeddings(embedding_model, [query])[0]
                            
                            # Calculate similarities
                            similarities = [model_utils.calculate_similarity(query_embedding, chunk_emb) for chunk_emb in chunk_embeddings]
                            
                            # Get top chunks
                            top_k = 3
                            top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:top_k]
                            top_chunks = [chunks[i] for i in top_indices]
                            top_similarities = [similarities[i] for i in top_indices]
                            
                            # Step 2: Generate answer based on retrieved chunks
                            if model_option == "Simulated":
                                # Generate a simulated answer
                                answer = "Based on the document, RAG systems offer several benefits including improved factual accuracy, the ability to use up-to-date information, reduction in hallucinations, and more transparent sourcing of information. Unlike regular LLMs, RAG can incorporate external knowledge without retraining the model."
                            else:
                                # Use OpenAI for generation
                                openai_client = openai_utils.initialize_openai_client(user_id)
                                
                                if openai_client:
                                    model = "gpt-4o" if "GPT-4o" in model_option else "gpt-3.5-turbo"
                                    answer = openai_utils.generate_rag_response(query, top_chunks, openai_client, model)
                                else:
                                    st.error("Failed to initialize OpenAI client. Falling back to simulated response.")
                                    answer = "Based on the document, RAG systems offer several benefits including improved factual accuracy, the ability to use up-to-date information, and reduction in hallucinations. (Note: This is a simulated response due to OpenAI API key issues)"
                            
                            # Show the answer
                            st.subheader("Answer")
                            st.write(answer)
                            
                            # Show source documents
                            st.subheader("Source Documents")
                            
                            with st.expander("Retrieved Chunks (Click to expand)"):
                                for i, (chunk, similarity) in enumerate(zip(top_chunks, top_similarities)):
                                    st.markdown(f"**Chunk {i+1}** (Similarity: {similarity:.4f})")
                                    st.markdown(f"```\n{chunk}\n```")
                                    st.markdown("---")
                        
                        except Exception as e:
                            st.error(f"Error processing query: {str(e)}")
                            st.info("This could be due to missing dependencies. Please make sure sentence-transformers is installed.")
            
            # Add explanation of how it works
            with st.expander("How this RAG Pipeline Works"):
                st.markdown("""
                1. **Document Processing**: The uploaded document is processed and split into chunks
                2. **Embedding Generation**: Each chunk is converted to a vector embedding
                3. **Query Processing**: Your question is also converted to a vector embedding
                4. **Retrieval**: The system finds chunks most similar to your question
                5. **Generation**: The retrieved chunks are used as context to generate an answer
                
                This is a simplified demonstration of RAG. In production systems, more sophisticated 
                techniques would be used for chunking, retrieval, and answer generation.
                """)
    
    # RAG Evaluation tab
    with tabs[4]:
        st.header("RAG Evaluation")
        
        st.markdown("""
        Evaluating RAG systems requires assessing both the retrieval and generation components, 
        as well as the overall pipeline.
        
        ### Key Evaluation Dimensions
        """)
        
        eval_dimensions = {
            "Dimension": [
                "Retrieval Relevance", 
                "Response Relevance", 
                "Faithfulness", 
                "Context Efficiency", 
                "Answer Completeness",
                "Groundedness"
            ],
            "Description": [
                "How relevant are the retrieved documents to the query?",
                "How relevant is the generated response to the query?",
                "Does the response accurately reflect the retrieved documents?",
                "Is the system retrieving the right amount of context?",
                "Does the response fully answer all aspects of the query?",
                "Can all claims in the response be verified from the retrieved context?"
            ],
            "Metrics": [
                "Precision, Recall, MRR, NDCG",
                "Human evaluation, relevance scoring",
                "Entailment scoring, hallucination detection",
                "Context utilization ratio, token efficiency",
                "Answer coverage scoring, human evaluation",
                "Citation precision, attribution accuracy"
            ]
        }
        
        eval_df = pd.DataFrame(eval_dimensions)
        st.table(eval_df)
        
        st.markdown("---")
        
        st.subheader("RAG Evaluation Methods")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Automated Evaluation")
            st.markdown("""
            - **RAGAS**: Framework specifically for RAG evaluation
              - Measures faithfulness, answer relevance, context relevance, context precision
            
            - **Retrieval Metrics**:
              - Precision@K and Recall@K
              - Mean Reciprocal Rank (MRR)
              - Normalized Discounted Cumulative Gain (NDCG)
            
            - **Generation Metrics**:
              - BLEU, ROUGE (against reference answers)
              - BERTScore for semantic similarity
              - SummaC for factual consistency
            
            - **End-to-End Evaluation**:
              - QA pairs with known answers
              - Factual accuracy checking
              - LLM-as-judge approaches
            """)
        
        with col2:
            st.markdown("### Human Evaluation")
            st.markdown("""
            - **Blind Comparison Tests**:
              - A/B testing different RAG configurations
              - Human judges unaware of system details
            
            - **Expert Review**:
              - Domain experts evaluate factual accuracy
              - Assess citation validity and context usage
            
            - **User Satisfaction Surveys**:
              - Measure end-user perception of quality
              - Track feedback after real-world interactions
            
            - **Annotation Campaigns**:
              - Structured rating of system outputs
              - Detailed error analysis and categorization
            """)
        
        st.markdown("---")
        
        st.subheader("Common RAG Issues and Solutions")
        
        problems = {
            "Problem": [
                "Irrelevant Retrievals",
                "Hallucinations Despite Retrieval",
                "Incomplete Answers",
                "Slow Response Time",
                "Contradictory Information",
                "Sensitivity to Query Phrasing"
            ],
            "Possible Causes": [
                "Poor chunking strategy, weak embeddings, query-document mismatch",
                "Prompt doesn't constrain the LLM, over-reliance on parametric knowledge",
                "Insufficient context retrieved, information spread across chunks",
                "Retrieving too many documents, inefficient embedding model",
                "Multiple conflicting sources retrieved, LLM synthesis issues",
                "Embedding model not robust to paraphrasing"
            ],
            "Potential Solutions": [
                "Improve chunking, use hybrid search, query reformulation",
                "Strengthen prompt constraints, implement fact-checking",
                "Retrieve more documents, use query decomposition",
                "Optimize retrieval count, use faster models, caching",
                "Source ranking, recency weighting, contradiction detection",
                "Query expansion, implement multi-query retrieval"
            ]
        }
        
        problems_df = pd.DataFrame(problems)
        st.table(problems_df)
        
        st.markdown("""
        ### RAG Tradeoffs and Optimizations
        
        When building and optimizing a RAG system, consider these tradeoffs:
        
        1. **Precision vs. Recall**: Retrieving more documents increases recall but may reduce precision
        
        2. **Speed vs. Quality**: More sophisticated retrieval methods often take longer
        
        3. **Freshness vs. Stability**: How often to update your knowledge base
        
        4. **Cost vs. Performance**: Better embedding models and more computation cost more
        
        5. **Generality vs. Specialization**: Domain-specific embeddings vs. general-purpose ones
        
        The best RAG system balances these considerations based on your specific use case requirements.
        """)
    
    st.markdown("---")
    st.info("This module covered the fundamentals of RAG. In the next module, we'll build a complete chatbot application using RAG techniques.")

