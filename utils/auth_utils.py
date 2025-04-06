import streamlit as st
from utils.db_utils import register_user, login_user, get_user_progress, get_total_score
import re

def validate_email(email: str) -> bool:
    """
    Validate email format.
    
    Args:
        email: Email to validate
        
    Returns:
        bool: Whether the email is valid
    """
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(email_pattern, email))

def validate_password(password: str) -> tuple[bool, str]:
    """
    Validate password strength.
    
    Args:
        password: Password to validate
        
    Returns:
        tuple[bool, str]: Validation result and message
    """
    if len(password) < 8:
        return False, "Password must be at least 8 characters long."
    
    # Check for at least one uppercase letter, one lowercase letter, and one digit
    if not (re.search(r'[A-Z]', password) and re.search(r'[a-z]', password) and re.search(r'[0-9]', password)):
        return False, "Password must contain at least one uppercase letter, one lowercase letter, and one digit."
    
    return True, "Password is valid."

def display_signup_form():
    """Display signup form and handle registration."""
    st.subheader("Sign Up")
    
    # Add access code verification
    from utils.access_code_utils import is_valid_access_code, mark_access_code_as_used, request_access

    # Show sign up options
    signup_option = st.radio(
        "Registration Method", 
        ["I have an access code", "Request access with email"]
    )
    
    if signup_option == "I have an access code":
        with st.form("signup_form"):
            access_code = st.text_input("Access Code", help="Enter the access code provided by the administrator")
            username = st.text_input("Username")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            
            submitted = st.form_submit_button("Sign Up")
            
            if submitted:
                # Validate inputs
                if not access_code or not username or not email or not password or not confirm_password:
                    st.error("Please fill in all fields including the access code.")
                    return
                
                # Verify access code first
                if not is_valid_access_code(access_code):
                    st.error("Invalid access code. Please check your code or request access below.")
                    return
                
                if not validate_email(email):
                    st.error("Please enter a valid email address.")
                    return
                
                is_valid_password, password_message = validate_password(password)
                if not is_valid_password:
                    st.error(password_message)
                    return
                
                if password != confirm_password:
                    st.error("Passwords do not match.")
                    return
                
                # Register the user
                success, message, user_id = register_user(username, email, password)
                
                if success and user_id is not None:
                    # Mark the access code as used
                    mark_access_code_as_used(access_code, user_id)
                    st.success(message)
                    st.info("Please log in with your new credentials.")
                    # Set session state to switch to login view
                    st.session_state.auth_view = "login"
                    st.rerun()
                else:
                    st.error(message)
    else:
        # Request access form
        with st.form("request_access_form"):
            email = st.text_input("Email", help="Your email address to receive access")
            reason = st.text_area("Reason for Access (Optional)", help="Why you want to access the workshop")
            
            submitted = st.form_submit_button("Request Access")
            
            if submitted:
                if not email:
                    st.error("Please enter your email address.")
                    return
                
                if not validate_email(email):
                    st.error("Please enter a valid email address.")
                    return
                
                # Submit access request
                success = request_access(email, reason)
                
                if success:
                    st.success("Your access request has been submitted. The administrator will review it soon.")
                    st.info("You will receive notification when approved.")
                else:
                    st.error("Failed to submit access request. You may already have a pending request.")
        
        # Add admin contact info
        st.info("For urgent access requirements, please contact the workshop administrator directly.")

def display_login_form():
    """Display login form and handle authentication."""
    st.subheader("Login")
    
    with st.form("login_form"):
        username_or_email = st.text_input("Username or Email")
        password = st.text_input("Password", type="password")
        
        submitted = st.form_submit_button("Login")
        
        if submitted:
            # Validate inputs
            if not username_or_email or not password:
                st.error("Please fill in all fields.")
                return
            
            # Check if it's an admin login
            from utils.admin_utils import is_admin
            if is_admin(username_or_email, password):
                st.session_state.admin_authenticated = True
                st.success("✅ Admin login successful!")
                st.session_state.current_page = "Admin Dashboard"
                st.rerun()
                return
            
            # If not admin, authenticate as regular user
            success, user_data, message = login_user(username_or_email, password)
            
            if success:
                # Store user data in session state
                st.session_state.user = user_data
                st.session_state.is_authenticated = True
                
                # Get user progress
                if user_data is not None and "id" in user_data:
                    user_progress = get_user_progress(user_data["id"])
                    st.session_state.user_progress = user_progress
                    
                    # Get total score
                    total_score = get_total_score(user_data["id"])
                    st.session_state.total_score = total_score
                else:
                    st.session_state.user_progress = []
                    st.session_state.total_score = 0
                
                st.success("Login successful!")
                st.rerun()
            else:
                st.error(message)

def display_auth_page():
    """Display authentication page with login and signup options."""
    # Initialize authentication view if not exists
    if 'auth_view' not in st.session_state:
        st.session_state.auth_view = "login"
    
    # Navigation tabs for login and signup
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Login", use_container_width=True, 
                    type="primary" if st.session_state.auth_view == "login" else "secondary"):
            st.session_state.auth_view = "login"
            st.rerun()
    
    with col2:
        if st.button("Sign Up", use_container_width=True,
                    type="primary" if st.session_state.auth_view == "signup" else "secondary"):
            st.session_state.auth_view = "signup"
            st.rerun()
    
    st.markdown("---")
    
    # Display the appropriate form based on the auth_view
    if st.session_state.auth_view == "login":
        display_login_form()
    else:
        display_signup_form()

def logout_user():
    """Log out the current user."""
    if 'user' in st.session_state:
        del st.session_state.user
    
    if 'is_authenticated' in st.session_state:
        del st.session_state.is_authenticated
    
    if 'admin_authenticated' in st.session_state:
        del st.session_state.admin_authenticated
    
    if 'user_progress' in st.session_state:
        del st.session_state.user_progress
    
    if 'total_score' in st.session_state:
        del st.session_state.total_score
    
    if 'current_page' in st.session_state:
        st.session_state.current_page = "Home"
    
    st.success("You have been logged out successfully.")
    st.rerun()

def init_auth_state():
    """Initialize authentication state."""
    if 'is_authenticated' not in st.session_state:
        st.session_state.is_authenticated = False
    
    if 'user' not in st.session_state:
        st.session_state.user = None
    
    if 'user_progress' not in st.session_state:
        st.session_state.user_progress = []
    
    if 'total_score' not in st.session_state:
        st.session_state.total_score = 0