import os
import sys

# Fix 1: Force Python to use pysqlite3 for updated SQLite
try:
    import pysqlite3
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    raise RuntimeError("pysqlite3 is required but not installed. Install it using 'pip install pysqlite3-binary'.")

# Fix 2: Disable Streamlit's file watcher to avoid PyTorch conflicts
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

# Streamlit imports
import streamlit as st

# Set page configuration (must be the first Streamlit command)
st.set_page_config(
    page_title="GenAI & LLM Workshop",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Other imports
import pandas as pd
from utils.auth_utils import display_auth_page, init_auth_state, logout_user
from utils.db_utils import init_connection_pool
from utils.progress_utils import display_progress_dashboard, get_module_completion_status
from utils.api_key_utils import display_api_key_form, initialize_api_keys
from utils.admin_utils import display_admin_login, display_admin_dashboard, check_module_access, is_admin
from utils.messaging_utils import display_user_messaging, display_admin_messaging, get_unread_message_count
import importlib
import logging
import traceback
import warnings

# Your remaining code...
# Check if required packages are installed
def is_package_installed(package_name):
    try:
        importlib.import_module(package_name)
        return True
    except (ImportError, ModuleNotFoundError):
        return False

# Add workshop_modules directory to the Python path
sys.path.append(os.path.abspath("workshop_modules"))

# Import modules even if some dependencies are missing
try:
    from workshop_modules import python_basics
    modules_found = True
except ImportError:
    modules_found = False
    st.error("Could not find python_basics module. Please ensure workshop_modules directory exists.")

# Import other modules, but don't fail if they're missing dependencies
all_modules_available = False
genai_intro = None
advanced_llms = None
rag_pipeline = None
chatbot_app = None

if modules_found:
    # Import modules directory for modules path availability
    try:
        # Try to import modules - we'll handle any runtime errors later
        sys.path.append(os.path.abspath("modules"))
        from modules import genai_intro, advanced_llms, rag_pipeline, chatbot_app 
        all_modules_available = True 
    except (ImportError, ModuleNotFoundError):
        pass

# Check for missing dependencies but don't block the application
missing_packages = []
for package in ['transformers', 'sentence_transformers', 'torch', 'langchain', 'pypdf2', 'chromadb']:
    if not is_package_installed(package):
        missing_packages.append(package)

# Initialize the database connection
init_connection_pool()

# Initialize authentication state
init_auth_state()

# Initialize admin state if not exists
if 'admin_authenticated' not in st.session_state:
    st.session_state.admin_authenticated = False

# Initialize session state for navigation
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Home"

# Check authentication status
is_authenticated = st.session_state.is_authenticated
user = st.session_state.user
admin_logged_in = st.session_state.admin_authenticated

# If user is authenticated, initialize API keys
if is_authenticated and user:
    initialize_api_keys(user["id"])

# Sidebar navigation
st.sidebar.title("🚀 GenAI LLM Workshop")

# Show user info if authenticated
if is_authenticated and user:
    st.sidebar.success(f"Welcome, {user['username']}!")
    
    # Display total score
    if 'total_score' in st.session_state:
        st.sidebar.metric("Total Score", st.session_state.total_score)

st.sidebar.markdown("---")

# Navigation options
if not is_authenticated and not admin_logged_in:
    # Limited navigation for unauthenticated users
    pages = {
        "Home": "🏠 Home",
        "Auth": "🔐 Sign In / Sign Up"
    }
elif admin_logged_in:
    # Admin navigation
    pages = {
        "Home": "🏠 Home",
        "Admin Dashboard": "📊 Admin Dashboard",
        "Admin Messaging": "💬 Messages"
    }
else:
    # Full navigation for authenticated users
    pages = {
        "Home": "🏠 Home",
        "Learning Path": "🗺️ Learning Path",
        "Progress": "📊 My Progress",
        "API Keys": "🔑 API Keys",
        "Messages": "💬 Messages",
        "Python Basics": "✅ Python Basics",
        "GenAI Intro": "🧠 GenAI Intro",
        "Advanced LLM": "🔍 Advanced LLM",
        "RAG Pipeline": "📄 RAG Pipeline",
        "Chatbot App": "🤖 Chatbot Project"
    }

# Check for unread messages
unread_message_count = 0
if is_authenticated and user:
    unread_message_count = get_unread_message_count(user["id"])

# Create navigation menu in sidebar
for page_key, page_title in pages.items():
    # Add unread message indicator for message pages
    if page_key == "Messages" and unread_message_count > 0:
        title_with_count = f"{page_title} ({unread_message_count})"
    elif page_key == "Admin Messaging" and unread_message_count > 0:
        title_with_count = f"{page_title} ({unread_message_count})"
    else:
        title_with_count = page_title
        
    if st.sidebar.button(title_with_count, key=page_key, use_container_width=True):
        st.session_state.current_page = page_key
        st.rerun()

# Add logout button if authenticated
if is_authenticated or admin_logged_in:
    st.sidebar.markdown("---")
    if st.sidebar.button("Logout", use_container_width=True):
        # Also reset admin state
        st.session_state.admin_authenticated = False
        logout_user()

st.sidebar.markdown("---")
st.sidebar.info(
    "This app demonstrates practical applications of Generative AI and LLMs."
)

# Display dependency warning if needed
if missing_packages:
    st.sidebar.warning(f"Missing packages: {', '.join(missing_packages)}. Some features will be limited.")

# Main content based on navigation
if st.session_state.current_page == "Home":
    st.title("Comprehensive Generative AI & LLMs Workshop")
    
    # Main intro section
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("""
        <div style="background-color:#f0f2f6; padding:20px; border-radius:10px; margin-bottom:20px;">
            <h3>🚀 Build Production-Ready AI Skills</h3>
            <p>A comprehensive 15-hour workshop covering everything from Python basics to building 
            advanced AI applications with the latest Large Language Models and RAG techniques.</p>
            <p><strong>Duration:</strong> 15 Hours | <strong>Modules:</strong> 5 | <strong>Hands-on Projects:</strong> Multiple</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Show login prompt for unauthenticated users
        if not is_authenticated and not admin_logged_in:
            if st.button("Sign In / Sign Up", use_container_width=True, type="primary"):
                st.session_state.current_page = "Auth"
                st.rerun()
        elif is_authenticated and not admin_logged_in:
            if st.button("Continue Learning", use_container_width=True, type="primary"):
                st.session_state.current_page = "Learning Path"
                st.rerun()
    
    with col2:
        st.image("generated-icon.png", width=250)
    
    # Warning about missing packages
    if missing_packages:
        st.warning(f"**Workshop Dependencies Missing:** {', '.join(missing_packages)}. Some modules will have limited functionality until these are installed.")
    
    # Workshop highlights
    st.markdown("## Workshop Highlights")
    
    highlights_col1, highlights_col2 = st.columns(2)
    
    with highlights_col1:
        st.markdown("""
        ### What You'll Learn
        - 🔍 **Practical AI Skills**: From Python basics to advanced LLM concepts
        - 🧠 **Prompt Engineering**: Master techniques for effective AI interactions
        - 🔄 **RAG Systems**: Build powerful knowledge-based AI applications
        - 🤖 **LLM Fine-Tuning**: Customize models for specific use cases
        - 📊 **Vector Databases**: Work with ChromaDB, FAISS, and more
        """)
    
    with highlights_col2:
        st.markdown("""
        ### Technologies Covered
        - 🛠️ **Python Libraries**: pandas, numpy, requests, transformers
        - 🌐 **LLM Integration**: OpenAI, Hugging Face, Mistral, LLaMA
        - 🔗 **Frameworks**: LangChain, Haystack, Sentence Transformers
        - 🎨 **UI Tools**: Streamlit, Gradio, Flask, FastAPI
        - 📂 **Data Processing**: PyPDF2, Tika, document parsers
        """)
    
    # Module overview
    st.markdown("## Workshop Modules")
    
    # Define modules with enhanced descriptions
    modules = [
        {
            "name": "Python Essentials for AI", 
            "icon": "🐍", 
            "duration": "2 Hours",
            "desc": "Learn Python fundamentals essential for AI development, including API interactions, data manipulation, and core libraries. Includes hands-on projects using real APIs and datasets."
        },
        {
            "name": "Introduction to Generative AI", 
            "icon": "🧠", 
            "duration": "3 Hours",
            "desc": "Explore LLM fundamentals, prompt engineering techniques, and various model playgrounds. Compare different models like GPT, Mistral, and BLOOM while learning visualization techniques."
        },
        {
            "name": "Advanced LLM Concepts", 
            "icon": "🔬", 
            "duration": "3 Hours",
            "desc": "Master fine-tuning, PEFT (LoRA/Adapters), bias analysis, ethical considerations, and model optimization. Includes hands-on fine-tuning projects with custom datasets."
        },
        {
            "name": "Retrieval-Augmented Generation", 
            "icon": "🔍", 
            "duration": "3 Hours",
            "desc": "Build comprehensive RAG systems with vector databases, document embeddings, and retrieval orchestration. Create knowledge-based Q&A systems using Wikipedia and custom documents."
        },
        {
            "name": "Mini-Project: AI Chatbot", 
            "icon": "🤖", 
            "duration": "4 Hours",
            "desc": "Apply all concepts to build a complete PDF Q&A chatbot with Streamlit UI, document processing, vector storage, and LLM integration. Deploy your creation to Hugging Face Spaces."
        }
    ]
    
    # Display module cards with CSS grid
    st.markdown("""
    <style>
    .module-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
        gap: 20px;
        margin-top: 20px;
    }
    .module-item {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #4a86e8;
    }
    .module-title {
        font-size: 1.2rem;
        font-weight: bold;
        margin-bottom: 5px;
    }
    .module-duration {
        color: #666;
        font-size: 0.9rem;
        margin-bottom: 10px;
    }
    </style>
    
    <div class="module-grid">
    """, unsafe_allow_html=True)
    
    for module in modules:
        st.markdown(f"""
        <div class="module-item">
            <div class="module-title">{module['icon']} {module['name']}</div>
            <div class="module-duration">⏱️ {module['duration']}</div>
            <p>{module['desc']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Workshop benefits and requirements
    st.markdown("## Who Should Attend")
    
    benefit_col1, benefit_col2 = st.columns(2)
    
    with benefit_col1:
        st.markdown("""
        ### Perfect For
        - Data scientists looking to integrate LLMs into workflows
        - Developers wanting to build AI-powered applications
        - Researchers exploring cutting-edge AI techniques
        - Students seeking hands-on experience with GenAI
        - Professionals transitioning to AI-focused roles
        """)
    
    with benefit_col2:
        st.markdown("""
        ### Requirements
        - Basic Python programming knowledge
        - Familiarity with data structures and APIs
        - Curiosity about AI and machine learning
        - Computer with internet access
        - Free accounts on platforms like Hugging Face
        """)
        
    # Get started button for unauthenticated users
    if not is_authenticated and not admin_logged_in:
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Get Started with the Workshop", use_container_width=True, type="primary"):
                st.session_state.current_page = "Auth"
                st.rerun()

elif st.session_state.current_page == "Admin":
    # Check if we're already logged in as admin
    if st.session_state.get("admin_authenticated", False):
        st.session_state.current_page = "Admin Dashboard"
        st.rerun()
    else:
        # Show admin login form
        st.subheader("Admin Login")
        
        with st.form("admin_login_form"):
            admin_username = st.text_input("Admin Username")
            admin_password = st.text_input("Admin Password", type="password")
            
            st.info("Default admin credentials: username = 'admin', password = 'Admin123!'")
            
            submitted = st.form_submit_button("Login")
            
            if submitted:
                if is_admin(admin_username, admin_password):
                    st.session_state.admin_authenticated = True
                    st.success("✅ Admin login successful!")
                    st.session_state.current_page = "Admin Dashboard"
                    st.rerun()
                else:
                    st.error("❌ Invalid admin credentials.")

elif st.session_state.current_page == "Admin Dashboard":
    if not st.session_state.get("admin_authenticated", False):
        st.warning("⚠️ Please log in as admin to access this dashboard.")
        # Redirect directly to auth page
        st.session_state.current_page = "Auth"
        st.rerun()
    else:
        display_admin_dashboard()

elif st.session_state.current_page == "Learning Path":
    if not is_authenticated:
        st.warning("Please log in to view your learning path.")
        if st.button("Sign In / Sign Up"):
            st.session_state.current_page = "Auth"
            st.rerun()
    else:
        st.title("Your Learning Path")
        
        st.markdown("""
        <style>
        .module-card {
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            border-left: 5px solid #4a86e8;
        }
        .module-complete {
            border-left: 5px solid #4caf50;
        }
        .module-card h3 {
            margin-top: 0;
        }
        .module-card p {
            margin-bottom: 10px;
        }
        .module-status {
            font-weight: bold;
            margin-bottom: 10px;
        }
        .status-complete {
            color: #4caf50;
        }
        .status-incomplete {
            color: #f39c12;
        }
        </style>
        """, unsafe_allow_html=True)
        
        modules = [
            {
                "key": "Python Basics",
                "title": "Python Essentials for AI (2 Hours)",
                "description": "Master APIs, data handling, and essential libraries like requests, pandas, numpy, and transformers. Work with interactive coding environments and build real applications with weather and social data.",
                "icon": "🐍",
                "duration": "2 hours"
            },
            {
                "key": "GenAI Intro",
                "title": "Introduction to Generative AI & LLMs (3 Hours)",
                "description": "Explore various LLM playgrounds, prompt engineering techniques, and visualization tools. Compare outputs from different models at varying temperatures and learn how to craft effective prompts.",
                "icon": "🧠",
                "duration": "3 hours"
            },
            {
                "key": "Advanced LLM",
                "title": "Advanced LLM Concepts (3 Hours)",
                "description": "Dive into fine-tuning, parameter-efficient tuning (PEFT), bias analysis, ethical AI considerations, and model compression techniques. Use LoRA to customize models on your own datasets.",
                "icon": "🔬",
                "duration": "3 hours"
            },
            {
                "key": "RAG Pipeline",
                "title": "Retrieval-Augmented Generation (3 Hours)",
                "description": "Build systems with vector databases (ChromaDB, FAISS, Weaviate), embed documents with Sentence Transformers, and orchestrate RAG workflows with LangChain to create powerful knowledge-based applications.",
                "icon": "🔍",
                "duration": "3 hours"
            },
            {
                "key": "Chatbot App",
                "title": "Mini-Project: AI Chatbot with RAG (4 Hours)",
                "description": "Develop a PDF Q&A chatbot using Streamlit/Gradio for UI, document extraction with PyPDF2, vector storage with ChromaDB, and integration with LLMs through LangChain. Deploy your creation to Hugging Face Spaces.",
                "icon": "🤖",
                "duration": "4 hours"
            }
        ]
        
        for i, module in enumerate(modules):
            # Get completion status
            is_completed, score = get_module_completion_status(module["key"])
            
            # Check if user has access to this module
            has_access = check_module_access(st.session_state.user["id"], module["key"])
            
            # Determine status class
            status_class = "module-complete" if is_completed else ""
            status_text_class = "status-complete" if is_completed else "status-incomplete"
            if not has_access:
                status_text_class = "status-incomplete"
                status_text = "🔒 Access Restricted"
            else:
                status_text = "✓ Completed" if is_completed else "⋯ In Progress"
            
            # Display module card
            st.markdown(f"""
            <div class="module-card {status_class}">
                <h3>{module["icon"]} {module["title"]}</h3>
                <div class="module-status {status_text_class}">{status_text}</div>
                <p>{module["description"]}</p>
                <p><strong>Duration:</strong> {module["duration"]}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show tools and resources based on module
            if module["key"] == "Python Basics":
                with st.expander("Tools & Resources"):
                    st.markdown("""
                    **Key Tools & Resources:**
                    - **Interactive Coding:** Google Colab, Replit, Kaggle Kernels
                    - **APIs & Data Handling:** OpenWeatherMap, Twitter API, JSON tools
                    - **Essential Libraries:** requests, pandas, numpy, transformers
                    """)
            elif module["key"] == "GenAI Intro":
                with st.expander("Tools & Resources"):
                    st.markdown("""
                    **Key Tools & Resources:**
                    - **LLM Playgrounds:** Hugging Face Spaces, Ollama (LLaMA, Mistral)
                    - **Prompt Engineering:** ChatGPT Playground, Learn Prompting
                    - **Visualizations:** Excalidraw, Transformer animations
                    """)
            elif module["key"] == "Advanced LLM":
                with st.expander("Tools & Resources"):
                    st.markdown("""
                    **Key Tools & Resources:**
                    - **Fine-Tuning & PEFT:** Hugging Face PEFT, LoRA, Adapters
                    - **Bias Analysis:** AI Fairness 360 Toolkit
                    - **Model Compression:** GGML quantization, ONNX optimization
                    """)
            elif module["key"] == "RAG Pipeline":
                with st.expander("Tools & Resources"):
                    st.markdown("""
                    **Key Tools & Resources:**
                    - **Vector Databases:** ChromaDB, FAISS, Weaviate
                    - **RAG Orchestration:** LangChain, Sentence Transformers, Haystack
                    - **Visualization:** Mermaid.js flowcharts for diagramming
                    """)
            elif module["key"] == "Chatbot App":
                with st.expander("Tools & Resources"):
                    st.markdown("""
                    **Key Tools & Resources:**
                    - **UI/Backend:** Gradio, Streamlit, Flask, FastAPI
                    - **Hosting & Deployment:** Hugging Face Spaces, GitHub Pages
                    - **Data Extraction:** PyPDF2, Tika for document parsing
                    """)
            
            # Display missing package warning if relevant
            if module["key"] == "GenAI Intro" and 'transformers' in missing_packages:
                st.warning("⚠️ Required package 'transformers' is not installed for this module.")
            elif module["key"] == "Advanced LLM" and 'transformers' in missing_packages:
                st.warning("⚠️ Required package 'transformers' is not installed for this module.")
            elif module["key"] == "RAG Pipeline" and any(pkg in missing_packages for pkg in ['sentence_transformers', 'langchain', 'chromadb']):
                st.warning("⚠️ Required packages for this module are not installed.")
            elif module["key"] == "Chatbot App" and any(pkg in missing_packages for pkg in ['langchain', 'pypdf2', 'sentence_transformers']):
                st.warning("⚠️ Required packages for this module are not installed.")
            
            # Add button to navigate to module
            button_label = "Continue Learning" if not is_completed else "Review Module"
            if not has_access:
                button_label = "Access Restricted"
                
            if st.button(button_label, key=f"btn_{module['key']}", use_container_width=True, disabled=not has_access):
                if has_access:
                    st.session_state.current_page = module["key"]
                    st.rerun()
                else:
                    st.error("You don't have access to this module. Please contact the administrator.")
            
            # Add separator except for the last item
            if i < len(modules) - 1:
                st.markdown("---")

elif st.session_state.current_page == "Auth":
    display_auth_page()

elif st.session_state.current_page == "Progress":
    if not is_authenticated:
        st.warning("Please log in to view your progress.")
        if st.button("Sign In / Sign Up"):
            st.session_state.current_page = "Auth"
            st.rerun()
    else:
        display_progress_dashboard(st.session_state.user["id"])

elif st.session_state.current_page == "API Keys":
    if not is_authenticated:
        st.warning("Please log in to configure API keys.")
        if st.button("Sign In / Sign Up"):
            st.session_state.current_page = "Auth"
            st.rerun()
    else:
        display_api_key_form(st.session_state.user["id"])
        
elif st.session_state.current_page == "Messages":
    if not is_authenticated:
        st.warning("Please log in to access messaging.")
        if st.button("Sign In / Sign Up"):
            st.session_state.current_page = "Auth"
            st.rerun()
    else:
        display_user_messaging()
        
elif st.session_state.current_page == "Admin Messaging":
    if not st.session_state.get("admin_authenticated", False):
        st.warning("⚠️ Please log in as admin to access messaging.")
        # Redirect directly to auth page
        st.session_state.current_page = "Auth"
        st.rerun()
    else:
        display_admin_messaging()

elif st.session_state.current_page == "Python Basics":
    if not is_authenticated:
        st.warning("Please log in to access this module.")
        if st.button("Sign In / Sign Up"):
            st.session_state.current_page = "Auth"
            st.rerun()
    elif not check_module_access(st.session_state.user["id"], "Python Basics"):
        st.error("Access to this module is restricted by the administrator.")
        st.info("Please contact the workshop administrator for access.")
        if st.button("Return to Home"):
            st.session_state.current_page = "Home"
            st.rerun()
    else:
        python_basics.show()
    
elif st.session_state.current_page == "GenAI Intro":
    if not is_authenticated:
        st.warning("Please log in to access this module.")
        if st.button("Sign In / Sign Up"):
            st.session_state.current_page = "Auth"
            st.rerun()
    elif not check_module_access(st.session_state.user["id"], "GenAI Intro"):
        st.error("Access to this module is restricted by the administrator.")
        st.info("Please contact the workshop administrator for access.")
        if st.button("Return to Home"):
            st.session_state.current_page = "Home"
            st.rerun()
    else:
        # Check for missing dependencies but still proceed with module display
        if 'transformers' in missing_packages or 'sentence_transformers' in missing_packages or 'torch' in missing_packages:
            st.warning("⚠️ Some features of this module will be limited because required packages are missing.")
            st.info("The module will show available content, but interactive features may not work properly.")
        
        try:
            # Only attempt to show the module if it was successfully imported
            if genai_intro is not None:
                genai_intro.show()
            else:
                st.title("Introduction to Generative AI & LLMs")
                st.markdown("""
                ### This module requires additional packages to function fully
                
                This module covers:
                - Understanding Large Language Models (LLMs)
                - Prompt engineering techniques
                - Comparing different models and parameters
                - Hands-on exploration of model behaviors
                
                To access the full interactive content, please install the missing dependencies:
                """)
                st.code("pip install transformers sentence-transformers torch", language="bash")
        except Exception as e:
            st.error(f"Error loading this module: {str(e)}")
            st.info("This may be due to missing dependencies. Please check the sidebar for more information.")
    
elif st.session_state.current_page == "Advanced LLM":
    if not is_authenticated:
        st.warning("Please log in to access this module.")
        if st.button("Sign In / Sign Up"):
            st.session_state.current_page = "Auth"
            st.rerun()
    elif not check_module_access(st.session_state.user["id"], "Advanced LLM"):
        st.error("Access to this module is restricted by the administrator.")
        st.info("Please contact the workshop administrator for access.")
        if st.button("Return to Home"):
            st.session_state.current_page = "Home"
            st.rerun()
    else:
        # Check for missing dependencies but still proceed with module display
        if 'transformers' in missing_packages or 'torch' in missing_packages:
            st.warning("⚠️ Some features of this module will be limited because required packages are missing.")
            st.info("The module will show available content, but interactive features may not work properly.")
        
        try:
            # Only attempt to show the module if it was successfully imported
            if advanced_llms is not None:
                advanced_llms.show()
            else:
                st.title("Advanced LLM Concepts")
                st.markdown("""
                ### This module requires additional packages to function fully
                
                This module covers:
                - Fine-tuning language models for specific tasks
                - Parameter-efficient tuning using LoRA and adapters
                - Bias analysis and ethical considerations
                - Model optimization and compression techniques
                
                To access the full interactive content, please install the missing dependencies:
                """)
                st.code("pip install transformers torch", language="bash")
        except Exception as e:
            st.error(f"Error loading this module: {str(e)}")
            st.info("This may be due to missing dependencies. Please check the sidebar for more information.")
    
elif st.session_state.current_page == "RAG Pipeline":
    if not is_authenticated:
        st.warning("Please log in to access this module.")
        if st.button("Sign In / Sign Up"):
            st.session_state.current_page = "Auth"
            st.rerun()
    elif not check_module_access(st.session_state.user["id"], "RAG Pipeline"):
        st.error("Access to this module is restricted by the administrator.")
        st.info("Please contact the workshop administrator for access.")
        if st.button("Return to Home"):
            st.session_state.current_page = "Home"
            st.rerun()
    else:
        # Check for missing dependencies but still proceed with module display
        required_packages = ['sentence_transformers', 'langchain', 'chromadb']
        missing_required = [pkg for pkg in required_packages if pkg in missing_packages]
        
        if missing_required:
            st.warning(f"⚠️ Some features of this module will be limited because the following packages are missing: {', '.join(missing_required)}")
            st.info("The module will show available content, but interactive features may not work properly.")
        
        try:
            # Only attempt to show the module if it was successfully imported
            if rag_pipeline is not None:
                rag_pipeline.show()
            else:
                st.title("Retrieval-Augmented Generation (RAG)")
                st.markdown("""
                ### This module requires additional packages to function fully
                
                This module covers:
                - Building document retrieval systems
                - Working with vector databases like ChromaDB and FAISS
                - Creating embeddings with Sentence Transformers
                - Implementing complete RAG workflows with LangChain
                - Creating knowledge-based Q&A systems
                
                To access the full interactive content, please install the missing dependencies:
                """)
                st.code("pip install sentence-transformers langchain chromadb", language="bash")
        except Exception as e:
            st.error(f"Error loading this module: {str(e)}")
            st.info("This may be due to missing dependencies. Please check the sidebar for more information.")
    
elif st.session_state.current_page == "Chatbot App":
    if not is_authenticated:
        st.warning("Please log in to access this module.")
        if st.button("Sign In / Sign Up"):
            st.session_state.current_page = "Auth"
            st.rerun()
    elif not check_module_access(st.session_state.user["id"], "Chatbot App"):
        st.error("Access to this module is restricted by the administrator.")
        st.info("Please contact the workshop administrator for access.")
        if st.button("Return to Home"):
            st.session_state.current_page = "Home"
            st.rerun()
    else:
        # Check for missing dependencies but still proceed with module display
        required_packages = ['langchain', 'pypdf2', 'sentence_transformers', 'chromadb']
        missing_required = [pkg for pkg in required_packages if pkg in missing_packages]
        
        if missing_required:
            st.warning(f"⚠️ Some features of this module will be limited because the following packages are missing: {', '.join(missing_required)}")
            st.info("The module will show available content, but interactive features may not work properly.")
        
        try:
            # Only attempt to show the module if it was successfully imported
            if chatbot_app is not None:
                chatbot_app.show()
            else:
                st.title("Mini-Project: AI Chatbot with RAG")
                st.markdown("""
                ### This module requires additional packages to function fully
                
                This module covers:
                - Building a PDF Q&A system with Streamlit/Gradio
                - Parsing documents with PyPDF2
                - Creating and managing vector databases
                - Implementing retrieval-augmented generation 
                - Deployment to Hugging Face Spaces
                
                To access the full interactive content, please install the missing dependencies:
                """)
                st.code("pip install langchain pypdf2 sentence-transformers chromadb", language="bash")
        except Exception as e:
            st.error(f"Error loading this module: {str(e)}")
            st.info("This may be due to missing dependencies. Please check the sidebar for more information.")
