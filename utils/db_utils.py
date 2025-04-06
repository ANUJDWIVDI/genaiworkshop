import os
import psycopg2
from psycopg2 import pool
from passlib.hash import bcrypt
import streamlit as st
from typing import Optional, Dict, List, Tuple, Any
from contextlib import contextmanager
import time

# Create a connection pool
connection_pool = None

def init_connection_pool():
    """Initialize the database connection pool for local PostgreSQL."""
    global connection_pool
    try:
        if connection_pool is None:
            # Local PostgreSQL connection configuration
            connection_pool = pool.ThreadedConnectionPool(
                minconn=1,
                maxconn=10,
                host="localhost",
                database="workshop_db",
                user="workshop_user",
                password="workshop_pass",
                port="5432"
            )
            
            # Test the connection
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT version();")
                    version = cur.fetchone()
                    print(f"Connected to PostgreSQL: {version[0]}")
            
            return True
    except Exception as e:
        st.error(f"Database connection error: {e}")
        return False

@contextmanager
def get_connection():
    """Get a connection from the pool and handle cleanup."""
    conn = None
    try:
        conn = connection_pool.getconn()
        yield conn
    finally:
        if conn:
            connection_pool.putconn(conn)

def register_user(username: str, email: str, password: str) -> Tuple[bool, str, Optional[int]]:
    """
    Register a new user in the database.
    
    Args:
        username: Username for the new user
        email: Email address for the new user
        password: Password for the new user
        
    Returns:
        Tuple[bool, str, Optional[int]]: Success status, message, and user ID if successful
    """
    try:
        # Hash the password
        hashed_password = bcrypt.hash(password)
        
        # Add user_password column if it doesn't exist
        with get_connection() as conn:
            with conn.cursor() as cur:
                # Check if user_password column exists
                cur.execute("""
                    SELECT column_name 
                    FROM information_schema.columns 
                    WHERE table_name='workshop_users' AND column_name='user_password';
                """)
                if not cur.fetchone():
                    # Add user_password column
                    cur.execute("""
                        ALTER TABLE workshop_users 
                        ADD COLUMN user_password VARCHAR(255) NOT NULL DEFAULT '';
                    """)
                    conn.commit()
        
        # Register the user
        with get_connection() as conn:
            with conn.cursor() as cur:
                # Check if username already exists
                cur.execute("SELECT id FROM workshop_users WHERE username = %s", (username,))
                if cur.fetchone():
                    return False, "Username already exists.", None
                
                # Check if email already exists
                cur.execute("SELECT id FROM workshop_users WHERE email = %s", (email,))
                if cur.fetchone():
                    return False, "Email already exists.", None
                
                # Insert the new user
                cur.execute(
                    "INSERT INTO workshop_users (username, email, user_password) VALUES (%s, %s, %s) RETURNING id",
                    (username, email, hashed_password)
                )
                user_id = cur.fetchone()[0]
                conn.commit()
                
                # Initialize progress for all modules
                module_names = [
                    "Python Basics",
                    "GenAI Intro",
                    "Advanced LLM",
                    "RAG Pipeline",
                    "Chatbot App"
                ]
                
                for module_name in module_names:
                    cur.execute(
                        "INSERT INTO workshop_progress (user_id, module_name, completed, score) VALUES (%s, %s, %s, %s)",
                        (user_id, module_name, False, 0)
                    )
                conn.commit()
                
                return True, f"User registered successfully with ID: {user_id}", user_id
    except Exception as e:
        return False, f"Error registering user: {e}", None

def login_user(username_or_email: str, password: str) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    """
    Authenticate a user.
    
    Args:
        username_or_email: Username or email for login
        password: Password for login
        
    Returns:
        Tuple[bool, Optional[Dict], str]: Success status, user data if successful, and message
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                # Get the user by username or email
                cur.execute(
                    "SELECT id, username, email, user_password FROM workshop_users WHERE username = %s OR email = %s",
                    (username_or_email, username_or_email)
                )
                user_data = cur.fetchone()
                
                if not user_data:
                    return False, None, "Invalid username or email."
                
                user_id, username, email, hashed_password = user_data
                
                # Verify password
                if not bcrypt.verify(password, hashed_password):
                    return False, None, "Invalid password."
                
                # Update last login time
                cur.execute(
                    "UPDATE workshop_users SET last_login = CURRENT_TIMESTAMP WHERE id = %s",
                    (user_id,)
                )
                conn.commit()
                
                # Return user data
                user = {
                    "id": user_id,
                    "username": username,
                    "email": email
                }
                
                return True, user, "Login successful."
    except Exception as e:
        return False, None, f"Error during login: {e}"

def get_user_progress(user_id: int) -> List[Dict[str, Any]]:
    """
    Get the progress of a user across all modules.
    
    Args:
        user_id: ID of the user
        
    Returns:
        List[Dict[str, Any]]: List of progress data for each module
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT module_name, completed, completion_date, score 
                    FROM workshop_progress 
                    WHERE user_id = %s 
                    ORDER BY id
                    """,
                    (user_id,)
                )
                progress_data = cur.fetchall()
                
                progress_list = []
                for module_name, completed, completion_date, score in progress_data:
                    progress_list.append({
                        "module_name": module_name,
                        "completed": completed,
                        "completion_date": completion_date,
                        "score": score
                    })
                
                return progress_list
    except Exception as e:
        st.error(f"Error retrieving user progress: {e}")
        return []

def update_module_progress(user_id: int, module_name: str, completed: bool, score: int) -> bool:
    """
    Update progress for a specific module.
    
    Args:
        user_id: ID of the user
        module_name: Name of the module
        completed: Whether the module is completed
        score: Score achieved in the module
        
    Returns:
        bool: Success status
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                # Check if record exists
                cur.execute(
                    "SELECT id FROM workshop_progress WHERE user_id = %s AND module_name = %s",
                    (user_id, module_name)
                )
                record = cur.fetchone()
                
                if record:
                    # Update existing record
                    cur.execute(
                        """
                        UPDATE workshop_progress 
                        SET completed = %s, 
                            completion_date = CASE WHEN %s = true THEN CURRENT_TIMESTAMP ELSE completion_date END, 
                            score = %s 
                        WHERE user_id = %s AND module_name = %s
                        """,
                        (completed, completed, score, user_id, module_name)
                    )
                else:
                    # Insert new record
                    cur.execute(
                        """
                        INSERT INTO workshop_progress (user_id, module_name, completed, completion_date, score) 
                        VALUES (%s, %s, %s, CASE WHEN %s = true THEN CURRENT_TIMESTAMP ELSE NULL END, %s)
                        """,
                        (user_id, module_name, completed, completed, score)
                    )
                conn.commit()
                return True
    except Exception as e:
        st.error(f"Error updating module progress: {e}")
        return False

def get_total_score(user_id: int) -> int:
    """
    Calculate the total score for a user across all modules.
    
    Args:
        user_id: ID of the user
        
    Returns:
        int: Total score
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT SUM(score) FROM workshop_progress WHERE user_id = %s",
                    (user_id,)
                )
                total_score = cur.fetchone()[0]
                return total_score if total_score else 0
    except Exception as e:
        st.error(f"Error calculating total score: {e}")
        return 0

def save_quiz_response(user_id: int, question_id: int, user_answer: str, is_correct: bool) -> bool:
    """
    Save a user's response to a quiz question.
    
    Args:
        user_id: ID of the user
        question_id: ID of the question
        user_answer: User's answer
        is_correct: Whether the answer is correct
        
    Returns:
        bool: Success status
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO workshop_quiz_responses (user_id, question_id, user_answer, is_correct) 
                    VALUES (%s, %s, %s, %s)
                    """,
                    (user_id, question_id, user_answer, is_correct)
                )
                conn.commit()
                return True
    except Exception as e:
        st.error(f"Error saving quiz response: {e}")
        return False

def get_quiz_questions(module_name: str) -> List[Dict[str, Any]]:
    """
    Get quiz questions for a specific module.
    
    Args:
        module_name: Name of the module
        
    Returns:
        List[Dict[str, Any]]: List of quiz questions
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, question_text, options, correct_answer, explanation FROM workshop_quiz_questions WHERE module_name = %s",
                    (module_name,)
                )
                questions_data = cur.fetchall()
                
                questions = []
                for id, question_text, options, correct_answer, explanation in questions_data:
                    questions.append({
                        "id": id,
                        "question_text": question_text,
                        "options": options,
                        "correct_answer": correct_answer,
                        "explanation": explanation
                    })
                
                return questions
    except Exception as e:
        st.error(f"Error retrieving quiz questions: {e}")
        return []

def save_api_key(user_id: int, service_name: str, api_key: str) -> bool:
    """
    Save an API key for a user.
    
    Args:
        user_id: ID of the user
        service_name: Name of the service (e.g., 'openai', 'huggingface')
        api_key: API key to save
        
    Returns:
        bool: Success status
    """
    try:
        # Create api_keys table if it doesn't exist
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS api_keys (
                        id SERIAL PRIMARY KEY,
                        user_id INTEGER REFERENCES workshop_users(id) ON DELETE CASCADE,
                        service_name VARCHAR(100) NOT NULL,
                        api_key VARCHAR(255) NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(user_id, service_name)
                    )
                """)
                conn.commit()
                
                # Check if API key already exists
                cur.execute(
                    "SELECT id FROM api_keys WHERE user_id = %s AND service_name = %s",
                    (user_id, service_name)
                )
                key_record = cur.fetchone()
                
                if key_record:
                    # Update existing API key
                    cur.execute(
                        "UPDATE api_keys SET api_key = %s WHERE user_id = %s AND service_name = %s",
                        (api_key, user_id, service_name)
                    )
                else:
                    # Insert new API key
                    cur.execute(
                        "INSERT INTO api_keys (user_id, service_name, api_key) VALUES (%s, %s, %s)",
                        (user_id, service_name, api_key)
                    )
                conn.commit()
                return True
    except Exception as e:
        st.error(f"Error saving API key: {e}")
        return False

def get_api_key(user_id: int, service_name: str) -> Optional[str]:
    """
    Get an API key for a user.
    
    Args:
        user_id: ID of the user
        service_name: Name of the service
        
    Returns:
        Optional[str]: API key if found, None otherwise
    """
    try:
        # First check if api_keys table exists
        with get_connection() as conn:
            with conn.cursor() as cur:
                # Check if table exists
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public'
                        AND table_name = 'api_keys'
                    )
                """)
                table_exists = cur.fetchone()[0]
                
                if not table_exists:
                    # Create the table if it doesn't exist
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS api_keys (
                            id SERIAL PRIMARY KEY,
                            user_id INTEGER REFERENCES workshop_users(id) ON DELETE CASCADE,
                            service_name VARCHAR(100) NOT NULL,
                            api_key VARCHAR(255) NOT NULL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            UNIQUE(user_id, service_name)
                        )
                    """)
                    conn.commit()
                    return None
                
                # Get the API key
                cur.execute(
                    "SELECT api_key FROM api_keys WHERE user_id = %s AND service_name = %s",
                    (user_id, service_name)
                )
                result = cur.fetchone()
                return result[0] if result else None
    except Exception as e:
        st.error(f"Error retrieving API key: {e}")
        return None

def save_document(user_id: int, document_name: str, document_content: str, document_type: str) -> Optional[int]:
    """
    Save a document uploaded by a user.
    
    Args:
        user_id: ID of the user
        document_name: Name of the document
        document_content: Content of the document
        document_type: Type of the document (e.g., 'pdf', 'txt')
        
    Returns:
        Optional[int]: Document ID if saved successfully, None otherwise
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO workshop_documents (user_id, document_name, document_content, document_type) 
                    VALUES (%s, %s, %s, %s)
                    RETURNING id
                    """,
                    (user_id, document_name, document_content, document_type)
                )
                document_id = cur.fetchone()[0]
                conn.commit()
                return document_id
    except Exception as e:
        st.error(f"Error saving document: {e}")
        return None

def get_user_documents(user_id: int) -> List[Dict[str, Any]]:
    """
    Get documents uploaded by a user.
    
    Args:
        user_id: ID of the user
        
    Returns:
        List[Dict[str, Any]]: List of documents
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, document_name, document_type, uploaded_at, embedding_status 
                    FROM workshop_documents 
                    WHERE user_id = %s 
                    ORDER BY uploaded_at DESC
                    """,
                    (user_id,)
                )
                documents_data = cur.fetchall()
                
                documents = []
                for id, document_name, document_type, uploaded_at, embedding_status in documents_data:
                    documents.append({
                        "id": id,
                        "document_name": document_name,
                        "document_type": document_type,
                        "uploaded_at": uploaded_at,
                        "embedding_status": embedding_status
                    })
                
                return documents
    except Exception as e:
        st.error(f"Error retrieving user documents: {e}")
        return []

# Initialize the connection pool when the module is imported
init_connection_pool()