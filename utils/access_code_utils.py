import streamlit as st
import random
import string
from datetime import datetime
from utils.db_utils import get_connection

def generate_access_code(length=8):
    """
    Generate a random access code.
    
    Args:
        length: Length of the code
        
    Returns:
        str: Generated code
    """
    characters = string.ascii_uppercase + string.digits
    return ''.join(random.choice(characters) for _ in range(length))

def create_new_access_code(admin_id, custom_code=None, description=""):
    """
    Create a new access code.
    
    Args:
        admin_id: ID of the admin creating the code
        custom_code: Optional custom access code (if None, generates a random code)
        description: Optional description for the access code
        
    Returns:
        tuple[bool, str, str]: Success status, message, and generated code if successful
    """
    try:
        code = custom_code if custom_code else generate_access_code()
        
        with get_connection() as conn:
            with conn.cursor() as cur:
                # Check if the code already exists
                if custom_code:
                    cur.execute("SELECT id FROM access_codes WHERE code = %s", (code,))
                    if cur.fetchone():
                        return False, "This access code already exists. Please use a different code.", ""
                
                # Debug
                print(f"Creating access code: {code}, admin: {admin_id}, description: {description}")
                
                # Insert the code
                cur.execute("""
                    INSERT INTO access_codes (code, created_by_admin_id, created_at, is_active, description)
                    VALUES (%s, %s, %s, %s, %s)
                    RETURNING id
                """, (code, admin_id, datetime.now(), True, description))
                
                # Commit the transaction
                conn.commit()
                
                result = cur.fetchone()
                
                if result:
                    return True, "Access code created successfully.", code
                else:
                    return False, "Failed to create access code.", ""
    except Exception as e:
        print(f"Error creating access code: {e}")
        return False, f"Error creating access code: {e}", ""

def is_valid_access_code(code):
    """
    Check if an access code is valid.
    
    Args:
        code: Access code to check
        
    Returns:
        bool: Whether the code is valid
    """
    if not code:
        print("Empty access code provided")
        return False
    
    # Debug
    print(f"Checking access code: {code}")
    
    # Convert to uppercase for case-insensitive matching
    code = code.upper()
    
    # Allow special default code 'SAMPLE'
    if code == 'SAMPLE':
        print("Using default SAMPLE code")
        return True
        
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                # Debug
                print(f"Querying database for code: {code}")
                
                cur.execute("""
                    SELECT id FROM access_codes
                    WHERE upper(code) = upper(%s) AND is_active = true AND used_by_user_id IS NULL
                """, (code,))
                result = cur.fetchone()
                
                valid = result is not None
                print(f"Access code valid: {valid}")
                return valid
    except Exception as e:
        print(f"Error checking access code: {e}")
        st.error(f"Error checking access code: {e}")
        return False

def mark_access_code_as_used(code, user_id):
    """
    Mark an access code as used.
    
    Args:
        code: Access code
        user_id: ID of the user using the code
        
    Returns:
        bool: Success status
    """
    # Debug
    print(f"Marking access code as used: {code}, user_id: {user_id}")
    
    # Convert to uppercase for case-insensitive matching
    code = code.upper()
    
    # Special handling for 'SAMPLE' code - we don't mark it as used
    if code == 'SAMPLE':
        print("Using SAMPLE code - not marking as used")
        return True
        
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE access_codes
                    SET used_by_user_id = %s, used_at = %s, is_active = false
                    WHERE upper(code) = upper(%s) AND is_active = true
                    RETURNING id
                """, (user_id, datetime.now(), code))
                
                # Make sure to commit
                conn.commit()
                
                result = cur.fetchone()
                success = result is not None
                print(f"Code marked as used: {success}")
                return success
    except Exception as e:
        print(f"Error marking access code as used: {e}")
        st.error(f"Error marking access code as used: {e}")
        return False

def deactivate_access_code(code_id):
    """
    Deactivate an access code.
    
    Args:
        code_id: ID of the code to deactivate
        
    Returns:
        bool: Success status
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE access_codes
                    SET is_active = false
                    WHERE id = %s
                    RETURNING id
                """, (code_id,))
                result = cur.fetchone()
                return result is not None
    except Exception as e:
        st.error(f"Error deactivating access code: {e}")
        return False

def get_all_access_codes():
    """
    Get all access codes.
    
    Returns:
        list: List of access codes with details
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        ac.id, 
                        ac.code, 
                        admin_user.username as created_by_name,
                        ac.created_at,
                        used_user.username as used_by_name,
                        ac.used_at,
                        ac.is_active,
                        ac.description
                    FROM access_codes ac
                    LEFT JOIN workshop_users admin_user ON ac.created_by_admin_id = admin_user.id
                    LEFT JOIN workshop_users used_user ON ac.used_by_user_id = used_user.id
                    ORDER BY ac.created_at DESC
                """)
                codes = cur.fetchall()
                
                result = []
                for code in codes:
                    result.append({
                        "id": code[0],
                        "code": code[1],
                        "created_by": code[2],
                        "created_at": code[3],
                        "used_by": code[4],
                        "used_at": code[5],
                        "is_active": code[6],
                        "description": code[7] if len(code) > 7 else ""
                    })
                return result
    except Exception as e:
        st.error(f"Error retrieving access codes: {e}")
        return []

def request_access(email, reason=""):
    """
    Submit an access request.
    
    Args:
        email: Email of the requester
        reason: Reason for the request
        
    Returns:
        bool: Success status
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                # Check if table exists
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public'
                        AND table_name = 'access_requests'
                    )
                """)
                table_exists = cur.fetchone()[0]
                
                # Create table if it doesn't exist
                if not table_exists:
                    cur.execute("""
                        CREATE TABLE access_requests (
                            id SERIAL PRIMARY KEY,
                            email VARCHAR(100) NOT NULL,
                            reason TEXT,
                            requested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            status VARCHAR(20) DEFAULT 'pending',
                            processed_by INTEGER REFERENCES workshop_users(id),
                            processed_at TIMESTAMP
                        )
                    """)
                    conn.commit()
                
                # Check if email already has a pending request
                cur.execute("""
                    SELECT id FROM access_requests
                    WHERE email = %s AND status = 'pending'
                """, (email,))
                existing_request = cur.fetchone()
                
                if existing_request:
                    return False  # Already has a pending request
                
                # Insert new request
                cur.execute("""
                    INSERT INTO access_requests (email, reason, requested_at, status)
                    VALUES (%s, %s, %s, %s)
                    RETURNING id
                """, (email, reason, datetime.now(), 'pending'))
                result = cur.fetchone()
                
                return result is not None
    except Exception as e:
        st.error(f"Error submitting access request: {e}")
        return False

def get_pending_access_requests():
    """
    Get all pending access requests.
    
    Returns:
        list: List of pending requests
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                # Check if table exists
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public'
                        AND table_name = 'access_requests'
                    )
                """)
                table_exists = cur.fetchone()[0]
                
                if not table_exists:
                    return []
                
                cur.execute("""
                    SELECT id, email, reason, requested_at
                    FROM access_requests
                    WHERE status = 'pending'
                    ORDER BY requested_at DESC
                """)
                requests = cur.fetchall()
                
                result = []
                for req in requests:
                    result.append({
                        "id": req[0],
                        "email": req[1],
                        "reason": req[2],
                        "requested_at": req[3]
                    })
                return result
    except Exception as e:
        st.error(f"Error retrieving access requests: {e}")
        return []

def process_access_request(request_id, admin_id, status, generate_code=False, custom_code=None, description=""):
    """
    Process an access request.
    
    Args:
        request_id: ID of the request
        admin_id: ID of the admin processing the request
        status: New status ('approved' or 'rejected')
        generate_code: Whether to generate an access code for approved requests
        custom_code: Optional custom access code to use
        description: Optional description for the access code
        
    Returns:
        tuple[bool, str, str]: Success status, message, and generated code if applicable
    """
    try:
        code = None
        
        with get_connection() as conn:
            with conn.cursor() as cur:
                # Update request status
                cur.execute("""
                    UPDATE access_requests
                    SET status = %s, processed_by = %s, processed_at = %s
                    WHERE id = %s
                    RETURNING email
                """, (status, admin_id, datetime.now(), request_id))
                result = cur.fetchone()
                
                if not result:
                    return False, "Request not found.", ""
                
                # If approved and code generation requested
                if status == 'approved' and generate_code:
                    if custom_code:
                        # Check if custom code exists
                        cur.execute("SELECT id FROM access_codes WHERE code = %s", (custom_code,))
                        if cur.fetchone():
                            return False, "This custom code already exists. Please use a different code.", ""
                        
                        # Create the custom code
                        cur.execute("""
                            INSERT INTO access_codes (code, created_by_admin_id, created_at, is_active, description)
                            VALUES (%s, %s, %s, %s, %s)
                            RETURNING id
                        """, (custom_code, admin_id, datetime.now(), True, description))
                        code_result = cur.fetchone()
                        
                        if code_result:
                            code = custom_code
                        else:
                            return False, "Failed to create custom access code.", ""
                    else:
                        # Use the 'SAMPLE' code as default
                        code = 'SAMPLE'  # Default code for approved users
                
                return True, f"Request {status}.", code or ""
    except Exception as e:
        return False, f"Error processing access request: {e}", ""

def display_access_code_management():
    """Display access code management interface for admin."""
    st.title("Access Code Management")
    
    # Check for admin authentication - either way of setting it should work
    is_authenticated = False
    if "admin_authenticated" in st.session_state:
        is_authenticated = st.session_state.admin_authenticated
    
    if not is_authenticated:
        st.warning("This section is only accessible to administrators.")
        return
    
    # Create new access code
    st.subheader("Generate New Access Code")
    
    # Add tabs for random code vs custom code
    tab1, tab2 = st.tabs(["Random Code", "Custom Code"])
    
    with tab1:
        if st.button("Generate Random Access Code", type="primary"):
            success, message, code = create_new_access_code(1)  # Using admin ID 1
            if success:
                st.success(f"{message} Code: {code}")
            else:
                st.error(message)
    
    with tab2:
        custom_code = st.text_input("Enter Custom Access Code", key="custom_code_input")
        description = st.text_area("Description (Optional)", key="code_description", 
                               placeholder="Add a description for this access code...")
        
        if st.button("Create Custom Access Code", type="primary"):
            if not custom_code:
                st.error("Please enter a custom access code.")
            else:
                success, message, code = create_new_access_code(1, custom_code, description)
                if success:
                    st.success(f"{message} Code: {code}")
                else:
                    st.error(message)
    
    # Display existing access codes
    st.subheader("Existing Access Codes")
    
    codes = get_all_access_codes()
    
    if not codes:
        st.info("No access codes found.")
    else:
        # Convert to appropriate format for display
        code_data = []
        for code in codes:
            status = "Active" if code["is_active"] else "Used" if code["used_by"] else "Inactive"
            created_at = code["created_at"].strftime("%Y-%m-%d %H:%M") if code["created_at"] else ""
            used_at = code["used_at"].strftime("%Y-%m-%d %H:%M") if code["used_at"] else "-"
            
            code_data.append({
                "ID": code["id"],
                "Code": code["code"],
                "Description": code["description"] or "-",
                "Created By": code["created_by"] or "System",
                "Created At": created_at,
                "Used By": code["used_by"] or "-",
                "Used At": used_at,
                "Status": status
            })
        
        # Display as table
        st.dataframe(code_data)
        
        # Option to deactivate codes
        st.subheader("Deactivate Access Code")
        
        active_codes = [c for c in codes if c["is_active"]]
        if active_codes:
            code_options = {c["id"]: f"{c['code']} (Created by: {c['created_by'] or 'System'})" for c in active_codes}
            selected_code_id = st.selectbox(
                "Select Code to Deactivate",
                options=list(code_options.keys()),
                format_func=lambda x: code_options[x]
            )
            
            if st.button("Deactivate Selected Code"):
                if deactivate_access_code(selected_code_id):
                    st.success("Access code deactivated successfully.")
                    st.rerun()
                else:
                    st.error("Failed to deactivate access code.")
        else:
            st.info("No active codes to deactivate.")
    
    # Access requests section
    st.markdown("---")
    st.subheader("Pending Access Requests")
    
    requests = get_pending_access_requests()
    
    if not requests:
        st.info("No pending access requests.")
    else:
        # Convert to appropriate format for display
        request_data = []
        for req in requests:
            requested_at = req["requested_at"].strftime("%Y-%m-%d %H:%M") if req["requested_at"] else ""
            
            request_data.append({
                "ID": req["id"],
                "Email": req["email"],
                "Reason": req["reason"] or "Not provided",
                "Requested At": requested_at
            })
        
        # Display as table
        st.dataframe(request_data)
        
        # Process requests
        st.subheader("Process Access Request")
        
        request_options = {r["id"]: f"{r['email']} ({r['requested_at'].strftime('%Y-%m-%d %H:%M')})" for r in requests}
        selected_request_id = st.selectbox(
            "Select Request to Process",
            options=list(request_options.keys()),
            format_func=lambda x: request_options[x]
        )
        
        # Add tabs for approval options
        tab1, tab2 = st.tabs(["Default Approval", "Custom Code Approval"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Approve with SAMPLE Code", type="primary", key="approve_sample"):
                    success, message, code = process_access_request(selected_request_id, 1, "approved", True)
                    if success:
                        st.success(f"{message} Access code: {code}")
                        st.rerun()
                    else:
                        st.error(message)
            
            with col2:
                if st.button("Reject Request", type="secondary", key="reject_default"):
                    success, message, _ = process_access_request(selected_request_id, 1, "rejected")
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)
                        
        with tab2:
            custom_code = st.text_input("Enter Custom Access Code", key="custom_approval_code")
            description = st.text_area("Description (Optional)", key="custom_approval_description", 
                               placeholder="Add a description for this access code...")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Approve with Custom Code", type="primary", key="approve_custom"):
                    if not custom_code:
                        st.error("Please enter a custom access code.")
                    else:
                        success, message, code = process_access_request(
                            selected_request_id, 1, "approved", True, 
                            custom_code=custom_code, description=description
                        )
                        if success:
                            st.success(f"{message} Access code: {code}")
                            st.rerun()
                        else:
                            st.error(message)
            
            with col2:
                if st.button("Reject Request", type="secondary", key="reject_custom"):
                    success, message, _ = process_access_request(selected_request_id, 1, "rejected")
                    if success:
                        st.success(message)
                        st.rerun()
                    else:
                        st.error(message)