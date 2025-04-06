import streamlit as st
from utils.db_utils import save_api_key, get_api_key
import os

def display_api_key_form(user_id):
    """
    Display a form for entering and managing API keys.
    
    Args:
        user_id: ID of the current user
    """
    st.title("API Keys Configuration")
    
    st.markdown("""
    <div style="background-color:#f0f2f6; padding:20px; border-radius:10px; margin-bottom:20px;">
        <h3>Configure Your API Keys</h3>
        <p>Some features require API keys. Toggle the services you want to use.</p>
        <p>🔒 <b>Security Note</b>: Your API keys are securely stored and only accessible to you.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Set up tabs for different categories of API keys
    tab1, tab2, tab3 = st.tabs(["LLM Services", "Image Generation", "Optional Services"])
    
    with tab1:
        st.subheader("Language Model API Keys")
        
        # OpenAI API Key (Required for many features)
        api_key_section(
            user_id=user_id,
            service_name="openai",
            display_name="OpenAI API Key",
            env_var="OPENAI_API_KEY",
            description="""
            Required for GPT-4o, text generation, and other advanced features.
            Get one at [OpenAI Platform](https://platform.openai.com/account/api-keys).
            """,
            is_required=True
        )
        
        # Gemma API Key
        api_key_section(
            user_id=user_id,
            service_name="gemma",
            display_name="Gemma API Key",
            env_var="GEMMA_API_KEY",
            description="""
            Optional: Used for Gemma model access.
            Get one at [AI Studio](https://aistudio.google.com/).
            """,
            is_required=False
        )
    
    with tab2:
        st.subheader("Image Generation API Keys")
        
        # DALL-E/Image generation
        api_key_section(
            user_id=user_id,
            service_name="stability",
            display_name="Stability AI Key",
            env_var="STABILITY_API_KEY",
            description="""
            Optional: Used for Stable Diffusion image generation.
            Get one at [Stability AI](https://stability.ai/platform).
            """,
            is_required=False
        )
    
    with tab3:
        st.subheader("Other AI Services")
        
        # Hugging Face API key
        api_key_section(
            user_id=user_id,
            service_name="huggingface",
            display_name="Hugging Face API Key",
            env_var="HUGGINGFACE_API_KEY",
            description="""
            Optional: Used for accessing Hugging Face models.
            Get one at [Hugging Face](https://huggingface.co/settings/tokens).
            """,
            is_required=False
        )
        
        # Claude API key
        api_key_section(
            user_id=user_id,
            service_name="anthropic",
            display_name="Anthropic API Key (Claude)",
            env_var="ANTHROPIC_API_KEY",
            description="""
            Optional: Used for accessing Claude models.
            Get one at [Anthropic Console](https://console.anthropic.com/).
            """,
            is_required=False
        )
    
    # Help section
    st.markdown("---")
    with st.expander("Need Help with API Keys?"):
        st.markdown("""
        ### API Key Resources

        **OpenAI API**
        - [Getting Started Guide](https://platform.openai.com/docs/quickstart)
        - [API Reference](https://platform.openai.com/docs/api-reference)
        - [Pricing Information](https://openai.com/pricing)

        **Hugging Face**
        - [Getting Started with Inference API](https://huggingface.co/docs/api-inference/quicktour)
        - [Token Management](https://huggingface.co/docs/hub/security-tokens)

        **Troubleshooting**
        - Make sure you've created an account on the respective platforms
        - Verify that your API key has not expired
        - Check if you have sufficient credits in your account
        
        If you encounter persistent issues, contact your workshop instructor.
        """)

def api_key_section(user_id, service_name, display_name, env_var, description, is_required=False):
    """
    Display a section for a specific API key with toggle functionality.
    
    Args:
        user_id: ID of the current user
        service_name: Name of the service (e.g., "openai", "huggingface")
        display_name: Display name for the service (e.g., "OpenAI API Key")
        env_var: Environment variable name for the key
        description: Markdown description
        is_required: Whether this API key is required
    """
    # Create a container for the entire section
    key_container = st.container()
    
    with key_container:
        # Get existing API key if available
        try:
            api_key = get_api_key(user_id, service_name)
        except Exception as e:
            st.error(f"Error retrieving API key. Please try again later. Details: {str(e)}")
            api_key = None
        
        # Create header with status indicator
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"#### {display_name}")
        
        with col2:
            if api_key:
                st.success("Configured")
            elif is_required:
                st.error("Required")
            else:
                st.info("Optional")
        
        # Display description
        st.markdown(description)
        
        # Toggle for showing/hiding configuration
        config_key = f"show_{service_name}_config"
        if config_key not in st.session_state:
            st.session_state[config_key] = not api_key or is_required
        
        show_config = st.toggle(
            f"Configure {display_name}", 
            value=st.session_state[config_key],
            key=f"toggle_{service_name}"
        )
        
        st.session_state[config_key] = show_config
        
        # Show configuration form if toggled on
        if show_config:
            with st.form(f"{service_name}_api_key_form"):
                # Show masked version if key exists
                default_value = ""
                if api_key:
                    # Show masked version with last 4 chars visible
                    key_length = len(api_key)
                    if key_length > 4:
                        default_value = "•" * (key_length - 4) + api_key[-4:]
                    else:
                        default_value = "•" * key_length
                
                new_api_key = st.text_input(
                    f"Enter {display_name}",
                    type="password",
                    placeholder="sk-..." if service_name == "openai" else "",
                    help=f"Enter your {display_name} here"
                )
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    submit_button = st.form_submit_button(
                        f"Save {display_name}", 
                        type="primary" if is_required else "secondary",
                        use_container_width=True
                    )
                
                with col2:
                    if api_key:
                        clear_button = st.form_submit_button(
                            f"Clear {display_name}",
                            type="secondary",
                            use_container_width=True
                        )
                    else:
                        skip_button = st.form_submit_button(
                            "Skip for Now" if not is_required else "Continue with Warning",
                            type="secondary",
                            use_container_width=True,
                            disabled=is_required
                        )
                
                if submit_button and new_api_key:
                    try:
                        if save_api_key(user_id, service_name, new_api_key):
                            # Update environment variable for current session
                            os.environ[env_var] = new_api_key
                            st.success(f"{display_name} saved successfully!")
                            st.session_state[config_key] = False
                            st.rerun()
                        else:
                            st.error(f"Failed to save {display_name}.")
                    except Exception as e:
                        st.error(f"Error saving API key: {str(e)}")
                elif submit_button:
                    st.error("Please enter an API key.")
                
                if api_key and 'clear_button' in locals() and clear_button:
                    try:
                        # Use empty string to "clear" the key
                        if save_api_key(user_id, service_name, ""):
                            if env_var in os.environ:
                                del os.environ[env_var]
                            st.success(f"{display_name} cleared successfully!")
                            st.rerun()
                        else:
                            st.error(f"Failed to clear {display_name}.")
                    except Exception as e:
                        st.error(f"Error clearing API key: {str(e)}")
        
        # Add separator between sections
        st.markdown("---")

def initialize_api_keys(user_id):
    """
    Initialize API keys in the environment variables if they exist in the database.
    
    Args:
        user_id: ID of the current user
    """
    # Define mappings between service names and environment variables
    key_mappings = {
        "openai": "OPENAI_API_KEY",
        "huggingface": "HUGGINGFACE_API_KEY",
        "gemma": "GEMMA_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "stability": "STABILITY_API_KEY"
    }
    
    # Try to get and set each key
    for service_name, env_var in key_mappings.items():
        try:
            api_key = get_api_key(user_id, service_name)
            if api_key:
                os.environ[env_var] = api_key
        except Exception as e:
            # Log error but continue - this shouldn't break the application
            print(f"Error initializing {service_name} API key: {e}")

def check_api_key(user_id, service_name, env_var):
    """
    Check if a specific API key is configured.
    
    Args:
        user_id: ID of the current user
        service_name: Name of the service
        env_var: Environment variable name
        
    Returns:
        bool: Whether the API key is configured
    """
    # First check environment variable
    api_key = os.environ.get(env_var)
    
    # If not in environment, check database
    if not api_key:
        try:
            api_key = get_api_key(user_id, service_name)
            if api_key:
                os.environ[env_var] = api_key
        except Exception:
            # Return False on error
            return False
    
    return bool(api_key)

def check_openai_api_key(user_id):
    """
    Check if OpenAI API key is configured.
    
    Args:
        user_id: ID of the current user
        
    Returns:
        bool: Whether the OpenAI API key is configured
    """
    return check_api_key(user_id, "openai", "OPENAI_API_KEY")

def check_huggingface_api_key(user_id):
    """
    Check if Hugging Face API key is configured.
    
    Args:
        user_id: ID of the current user
        
    Returns:
        bool: Whether the Hugging Face API key is configured
    """
    return check_api_key(user_id, "huggingface", "HUGGINGFACE_API_KEY")

def check_gemma_api_key(user_id):
    """
    Check if Gemma API key is configured.
    
    Args:
        user_id: ID of the current user
        
    Returns:
        bool: Whether the Gemma API key is configured
    """
    return check_api_key(user_id, "gemma", "GEMMA_API_KEY")

def check_anthropic_api_key(user_id):
    """
    Check if Anthropic (Claude) API key is configured.
    
    Args:
        user_id: ID of the current user
        
    Returns:
        bool: Whether the Anthropic API key is configured
    """
    return check_api_key(user_id, "anthropic", "ANTHROPIC_API_KEY")