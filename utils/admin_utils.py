import streamlit as st
import pandas as pd
from utils.db_utils import get_connection
import plotly.express as px

def is_admin(username, password):
    """
    Check if the provided credentials match admin credentials.
    
    Args:
        username: Admin username
        password: Admin password
        
    Returns:
        bool: Whether credentials are valid
    """
    # For security, we should check credentials against a database
    # But for this example, we'll use hardcoded credentials
    ADMIN_USERNAME = "admin"
    ADMIN_PASSWORD = "Admin123!"  # This should be properly hashed in production
    
    return username == ADMIN_USERNAME and password == ADMIN_PASSWORD

def display_admin_login():
    """Display admin login form."""
    st.subheader("Admin Login")
    
    with st.form("admin_login_form"):
        admin_username = st.text_input("Admin Username")
        admin_password = st.text_input("Admin Password", type="password")
        submitted = st.form_submit_button("Login")
        
        if submitted:
            if is_admin(admin_username, admin_password):
                st.session_state.admin_authenticated = True
                st.success("Admin login successful!")
                st.rerun()
            else:
                st.error("Invalid admin credentials.")

def get_all_users():
    """
    Get all users from the database.
    
    Returns:
        pd.DataFrame: DataFrame with user data
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT id, username, email, created_at, last_login 
                    FROM workshop_users 
                    ORDER BY created_at DESC
                """)
                users_data = cur.fetchall()
                
                if not users_data:
                    return pd.DataFrame()
                
                users_df = pd.DataFrame(users_data, columns=[
                    "id", "username", "email", "created_at", "last_login"
                ])
                return users_df
    except Exception as e:
        st.error(f"Error retrieving users: {e}")
        return pd.DataFrame()

def get_all_progress():
    """
    Get progress data for all users.
    
    Returns:
        pd.DataFrame: DataFrame with progress data
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        wp.id, 
                        wu.username, 
                        wp.module_name, 
                        wp.completed, 
                        wp.completion_date, 
                        wp.score 
                    FROM workshop_progress wp
                    JOIN workshop_users wu ON wp.user_id = wu.id
                    ORDER BY wu.username, wp.module_name
                """)
                progress_data = cur.fetchall()
                
                if not progress_data:
                    return pd.DataFrame()
                
                progress_df = pd.DataFrame(progress_data, columns=[
                    "id", "username", "module_name", "completed", "completion_date", "score"
                ])
                return progress_df
    except Exception as e:
        st.error(f"Error retrieving progress data: {e}")
        return pd.DataFrame()

def get_quiz_responses():
    """
    Get all quiz responses.
    
    Returns:
        pd.DataFrame: DataFrame with quiz response data
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        wqr.id,
                        wu.username,
                        wq.module_name,
                        wq.question_text,
                        wqr.user_answer,
                        wqr.is_correct,
                        wqr.response_time
                    FROM workshop_quiz_responses wqr
                    JOIN workshop_users wu ON wqr.user_id = wu.id
                    JOIN workshop_quiz_questions wq ON wqr.question_id = wq.id
                    ORDER BY wu.username, wqr.response_time DESC
                """)
                responses_data = cur.fetchall()
                
                if not responses_data:
                    return pd.DataFrame()
                
                responses_df = pd.DataFrame(responses_data, columns=[
                    "id", "username", "module_name", "question_text", 
                    "user_answer", "is_correct", "response_time"
                ])
                return responses_df
    except Exception as e:
        st.error(f"Error retrieving quiz responses: {e}")
        return pd.DataFrame()

def get_document_counts():
    """
    Get document counts by user.
    
    Returns:
        pd.DataFrame: DataFrame with document counts
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        wu.username,
                        COUNT(wd.id) as document_count,
                        MAX(wd.uploaded_at) as last_upload
                    FROM workshop_users wu
                    LEFT JOIN workshop_documents wd ON wu.id = wd.user_id
                    GROUP BY wu.username
                    ORDER BY document_count DESC
                """)
                doc_data = cur.fetchall()
                
                if not doc_data:
                    return pd.DataFrame()
                
                doc_df = pd.DataFrame(doc_data, columns=[
                    "username", "document_count", "last_upload"
                ])
                return doc_df
    except Exception as e:
        st.error(f"Error retrieving document counts: {e}")
        return pd.DataFrame()

def get_module_access_status():
    """
    Get module access status for all users.
    
    Returns:
        pd.DataFrame: DataFrame with module access status
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                # Check if the module_access table exists
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public'
                        AND table_name = 'module_access'
                    )
                """)
                table_exists = cur.fetchone()[0]
                
                if not table_exists:
                    # Create the table if it doesn't exist
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS module_access (
                            id SERIAL PRIMARY KEY,
                            user_id INTEGER REFERENCES workshop_users(id) ON DELETE CASCADE,
                            module_name VARCHAR(100) NOT NULL,
                            access_enabled BOOLEAN DEFAULT true,
                            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            UNIQUE(user_id, module_name)
                        )
                    """)
                    conn.commit()
                    
                    # Initialize module access for all users and modules
                    cur.execute("SELECT id FROM workshop_users")
                    users = cur.fetchall()
                    
                    module_names = [
                        "Python Basics",
                        "GenAI Intro",
                        "Advanced LLM",
                        "RAG Pipeline",
                        "Chatbot App"
                    ]
                    
                    for user_id in users:
                        for module_name in module_names:
                            cur.execute(
                                "INSERT INTO module_access (user_id, module_name, access_enabled) VALUES (%s, %s, %s)",
                                (user_id[0], module_name, True)
                            )
                    conn.commit()
                
                # Get module access data
                cur.execute("""
                    SELECT 
                        ma.id,
                        wu.username,
                        ma.module_name,
                        ma.access_enabled,
                        ma.last_updated
                    FROM module_access ma
                    JOIN workshop_users wu ON ma.user_id = wu.id
                    ORDER BY wu.username, ma.module_name
                """)
                access_data = cur.fetchall()
                
                if not access_data:
                    return pd.DataFrame()
                
                access_df = pd.DataFrame(access_data, columns=[
                    "id", "username", "module_name", "access_enabled", "last_updated"
                ])
                return access_df
    except Exception as e:
        st.error(f"Error retrieving module access data: {e}")
        return pd.DataFrame()

def update_module_access(user_id, module_name, access_enabled):
    """
    Update module access for a specific user and module.
    
    Args:
        user_id: ID of the user
        module_name: Name of the module
        access_enabled: Whether access is enabled
        
    Returns:
        bool: Success status
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                # Check if the module_access table exists
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public'
                        AND table_name = 'module_access'
                    )
                """)
                table_exists = cur.fetchone()[0]
                
                if not table_exists:
                    # Create the table if it doesn't exist
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS module_access (
                            id SERIAL PRIMARY KEY,
                            user_id INTEGER REFERENCES workshop_users(id) ON DELETE CASCADE,
                            module_name VARCHAR(100) NOT NULL,
                            access_enabled BOOLEAN DEFAULT true,
                            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            UNIQUE(user_id, module_name)
                        )
                    """)
                    conn.commit()
                
                # Update module access
                cur.execute("""
                    INSERT INTO module_access (user_id, module_name, access_enabled, last_updated)
                    VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
                    ON CONFLICT (user_id, module_name)
                    DO UPDATE SET 
                        access_enabled = %s,
                        last_updated = CURRENT_TIMESTAMP
                """, (user_id, module_name, access_enabled, access_enabled))
                conn.commit()
                return True
    except Exception as e:
        st.error(f"Error updating module access: {e}")
        return False

def check_module_access(user_id, module_name):
    """
    Check if a user has access to a specific module.
    
    Args:
        user_id: ID of the user
        module_name: Name of the module
        
    Returns:
        bool: Whether the user has access
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                # Check if the module_access table exists
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public'
                        AND table_name = 'module_access'
                    )
                """)
                table_exists = cur.fetchone()[0]
                
                if not table_exists:
                    # If the table doesn't exist, all users have access by default
                    return True
                
                # Check module access
                cur.execute("""
                    SELECT access_enabled
                    FROM module_access
                    WHERE user_id = %s AND module_name = %s
                """, (user_id, module_name))
                result = cur.fetchone()
                
                if result is None:
                    # If no record exists, grant access by default
                    return True
                
                return result[0]
    except Exception as e:
        st.error(f"Error checking module access: {e}")
        # Default to allowing access in case of errors
        return True

def get_module_completion_stats():
    """
    Get module completion statistics.
    
    Returns:
        pd.DataFrame: DataFrame with module statistics
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT 
                        module_name,
                        COUNT(*) as total_users,
                        SUM(CASE WHEN completed = true THEN 1 ELSE 0 END) as completed_count,
                        AVG(CASE WHEN score IS NOT NULL THEN score ELSE 0 END) as avg_score
                    FROM workshop_progress
                    GROUP BY module_name
                    ORDER BY module_name
                """)
                module_data = cur.fetchall()
                
                if not module_data:
                    return pd.DataFrame()
                
                module_df = pd.DataFrame(module_data, columns=[
                    "module_name", "total_users", "completed_count", "avg_score"
                ])
                module_df["completion_rate"] = (module_df["completed_count"] / module_df["total_users"]) * 100
                module_df["avg_score"] = module_df["avg_score"].round(1)
                module_df["completion_rate"] = module_df["completion_rate"].round(1)
                return module_df
    except Exception as e:
        st.error(f"Error retrieving module statistics: {e}")
        return pd.DataFrame()

def export_data_to_csv(table_name):
    """
    Export a database table to CSV.
    
    Args:
        table_name: Name of the table to export
        
    Returns:
        tuple[bool, str]: Success status and CSV data or error message
    """
    try:
        with get_connection() as conn:
            # Use pandas to read the SQL table directly
            df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
            
            if df.empty:
                return False, f"No data found in table {table_name}"
            
            # Convert to CSV string
            csv_data = df.to_csv(index=False)
            return True, csv_data
    except Exception as e:
        return False, f"Error exporting data: {e}"

def truncate_table(table_name):
    """
    Truncate (empty) a database table.
    
    Args:
        table_name: Name of the table to truncate
        
    Returns:
        bool: Success status
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                # Truncate the table
                cur.execute(f"TRUNCATE TABLE {table_name} CASCADE")
                conn.commit()
                return True
    except Exception as e:
        st.error(f"Error truncating table: {e}")
        return False

def display_admin_dashboard():
    """Display admin dashboard with all data visualizations."""
    st.title("📊 Admin Dashboard")
    
    st.markdown("""
    This dashboard provides an overview of all users, their progress, quiz responses, and system settings.
    Use the tabs below to view and manage different aspects of the workshop platform.
    """)
    
    # Create tabs for different data views
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "Users Overview", 
        "Module Progress", 
        "Quiz Responses", 
        "Documents", 
        "Module Statistics",
        "Module Access Control",
        "Registration & Access Codes",
        "Database Management"
    ])
    
    with tab1:
        st.header("Users Overview")
        
        # Get users data
        users_df = get_all_users()
        
        if users_df.empty:
            st.warning("No users found in the database.")
        else:
            # Display key metrics
            total_users = len(users_df)
            recent_users = len(users_df[users_df["created_at"] > pd.Timestamp.now() - pd.Timedelta(days=7)])
            active_users = len(users_df[users_df["last_login"] > pd.Timestamp.now() - pd.Timedelta(days=7)])
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Users", total_users)
            col2.metric("New Users (7 days)", recent_users)
            col3.metric("Active Users (7 days)", active_users)
            
            # Display users table
            st.subheader("All Users")
            
            # Convert timestamps to readable format
            users_df["created_at"] = users_df["created_at"].dt.strftime("%Y-%m-%d %H:%M")
            users_df["last_login"] = users_df["last_login"].dt.strftime("%Y-%m-%d %H:%M")
            
            st.dataframe(users_df, use_container_width=True)
    
    with tab2:
        st.header("Module Progress")
        
        # Get progress data
        progress_df = get_all_progress()
        
        if progress_df.empty:
            st.warning("No progress data found in the database.")
        else:
            # Format data
            progress_df["completion_date"] = pd.to_datetime(
                progress_df["completion_date"]
            ).dt.strftime("%Y-%m-%d %H:%M")
            
            # Filter options
            users = progress_df["username"].unique()
            modules = progress_df["module_name"].unique()
            
            col1, col2 = st.columns(2)
            with col1:
                selected_user = st.selectbox(
                    "Filter by User",
                    options=["All Users"] + list(users)
                )
            
            with col2:
                selected_module = st.selectbox(
                    "Filter by Module",
                    options=["All Modules"] + list(modules)
                )
            
            # Apply filters
            filtered_df = progress_df.copy()
            if selected_user != "All Users":
                filtered_df = filtered_df[filtered_df["username"] == selected_user]
            
            if selected_module != "All Modules":
                filtered_df = filtered_df[filtered_df["module_name"] == selected_module]
            
            # Display filtered data
            st.dataframe(filtered_df, use_container_width=True)
            
            # Show completion visualization
            if selected_user != "All Users":
                # For a single user, show their module progress
                user_progress = progress_df[progress_df["username"] == selected_user]
                
                fig = px.bar(
                    user_progress,
                    x="module_name",
                    y="score",
                    color="completed",
                    title=f"Module Scores for {selected_user}",
                    labels={"module_name": "Module", "score": "Score", "completed": "Completed"},
                    color_discrete_map={True: "green", False: "gray"}
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                # For all users, show completion by module
                completion_by_module = progress_df.groupby(["module_name", "completed"]).size().reset_index(name="count")
                completion_by_module = completion_by_module[completion_by_module["completed"] == True]
                
                fig = px.bar(
                    completion_by_module,
                    x="module_name",
                    y="count",
                    title="Completed Modules Count",
                    labels={"module_name": "Module", "count": "Number of Users Completed"}
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("Quiz Responses")
        
        # Get quiz response data
        responses_df = get_quiz_responses()
        
        if responses_df.empty:
            st.warning("No quiz responses found in the database.")
        else:
            # Format data
            responses_df["response_time"] = pd.to_datetime(
                responses_df["response_time"]
            ).dt.strftime("%Y-%m-%d %H:%M")
            
            # Calculate metrics
            total_responses = len(responses_df)
            correct_responses = len(responses_df[responses_df["is_correct"] == True])
            correct_pct = (correct_responses / total_responses) * 100 if total_responses > 0 else 0
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Responses", total_responses)
            col2.metric("Correct Responses", correct_responses)
            col3.metric("Correct Percentage", f"{correct_pct:.1f}%")
            
            # Filter options
            users = responses_df["username"].unique()
            modules = responses_df["module_name"].unique()
            
            col1, col2 = st.columns(2)
            with col1:
                selected_user = st.selectbox(
                    "Filter by User",
                    options=["All Users"] + list(users),
                    key="quiz_user_filter"
                )
            
            with col2:
                selected_module = st.selectbox(
                    "Filter by Module",
                    options=["All Modules"] + list(modules),
                    key="quiz_module_filter"
                )
            
            # Apply filters
            filtered_df = responses_df.copy()
            if selected_user != "All Users":
                filtered_df = filtered_df[filtered_df["username"] == selected_user]
            
            if selected_module != "All Modules":
                filtered_df = filtered_df[filtered_df["module_name"] == selected_module]
            
            # Display filtered data
            st.dataframe(filtered_df, use_container_width=True)
            
            # Show visualization of correct vs incorrect answers
            correct_by_module = responses_df.groupby(["module_name", "is_correct"]).size().reset_index(name="count")
            
            fig = px.bar(
                correct_by_module,
                x="module_name",
                y="count",
                color="is_correct",
                barmode="group",
                title="Correct vs Incorrect Responses by Module",
                labels={"module_name": "Module", "count": "Number of Responses", "is_correct": "Correct Answer"},
                color_discrete_map={True: "green", False: "red"}
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header("Documents")
        
        # Get document count data
        doc_df = get_document_counts()
        
        if doc_df.empty:
            st.warning("No document data found in the database.")
        else:
            # Format data
            doc_df["last_upload"] = pd.to_datetime(
                doc_df["last_upload"]
            ).dt.strftime("%Y-%m-%d %H:%M")
            
            # Display document counts
            st.dataframe(doc_df, use_container_width=True)
            
            # Show visualization
            fig = px.bar(
                doc_df,
                x="username",
                y="document_count",
                title="Documents Uploaded by User",
                labels={"username": "User", "document_count": "Number of Documents"}
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.header("Module Statistics")
        
        # Get module statistics
        module_df = get_module_completion_stats()
        
        if module_df.empty:
            st.warning("No module statistics found in the database.")
        else:
            # Display statistics
            st.dataframe(module_df, use_container_width=True)
            
            # Show visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                fig1 = px.bar(
                    module_df,
                    x="module_name",
                    y="completion_rate",
                    title="Module Completion Rate (%)",
                    labels={"module_name": "Module", "completion_rate": "Completion Rate (%)"}
                )
                
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                fig2 = px.bar(
                    module_df,
                    x="module_name",
                    y="avg_score",
                    title="Average Score by Module",
                    labels={"module_name": "Module", "avg_score": "Average Score"}
                )
                
                st.plotly_chart(fig2, use_container_width=True)
            
            # Completion funnel
            module_df_sorted = module_df.sort_values(by="completion_rate", ascending=False)
            
            fig3 = px.funnel(
                module_df_sorted,
                x="completed_count",
                y="module_name",
                title="Module Completion Funnel",
                labels={"completed_count": "Number of Users", "module_name": "Module"}
            )
            
            st.plotly_chart(fig3, use_container_width=True)
    
    with tab6:
        st.header("Module Access Control")
        st.markdown("""
        Use this panel to control which modules each user can access.
        Toggle the switches to enable or disable access to specific modules.
        """)
        
        # Get module access data
        access_df = get_module_access_status()
        
        if access_df.empty:
            st.warning("No users or modules found in the database.")
        else:
            # Format data
            access_df["last_updated"] = pd.to_datetime(
                access_df["last_updated"]
            ).dt.strftime("%Y-%m-%d %H:%M")
            
            # Get users and modules
            users = sorted(access_df["username"].unique())
            modules = [
                "Python Basics",
                "GenAI Intro",
                "Advanced LLM",
                "RAG Pipeline",
                "Chatbot App"
            ]
            
            # Create a user selection dropdown
            selected_user = st.selectbox(
                "Select a user to manage module access",
                options=users
            )
            
            if selected_user:
                # Get user ID
                with get_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute("SELECT id FROM workshop_users WHERE username = %s", (selected_user,))
                        user_id = cur.fetchone()[0]
                
                st.subheader(f"Module Access Settings for {selected_user}")
                
                # User's current access settings
                user_access = access_df[access_df["username"] == selected_user]
                
                # For each module, create a toggle
                for module in modules:
                    module_row = user_access[user_access["module_name"] == module]
                    current_access = True  # Default to True
                    
                    if not module_row.empty:
                        current_access = module_row["access_enabled"].iloc[0]
                    
                    # Create a toggle for the module
                    new_access = st.checkbox(
                        f"Enable access to {module}",
                        value=current_access,
                        key=f"access_{selected_user}_{module}"
                    )
                    
                    # If the toggle changed, update the database
                    if new_access != current_access:
                        success = update_module_access(user_id, module, new_access)
                        if success:
                            st.success(f"Updated {module} access for {selected_user} to {'enabled' if new_access else 'disabled'}")
                        else:
                            st.error(f"Failed to update {module} access for {selected_user}")
                
                # Display current access status
                st.subheader("Current Access Status")
                user_access = access_df[access_df["username"] == selected_user]
                
                # Add a button to view all module access settings
                if st.button("View All Module Access Settings"):
                    st.dataframe(access_df, use_container_width=True)
    
    with tab7:
        st.header("Registration & Access Codes")
        
        # Create tabs for different views
        subtab1, subtab2 = st.tabs(["Access Code Management", "Registration Requests"])
        
        # Import access code functions at the top level
        from utils.access_code_utils import display_access_code_management, get_pending_access_requests, process_access_request
        
        with subtab1:
            display_access_code_management()
            
        with subtab2:
            # Display pending access requests
            st.subheader("Pending Access Requests")
            
            # Check for admin authentication - either way of setting it should work
            is_authenticated = False
            if "admin_authenticated" in st.session_state:
                is_authenticated = st.session_state.admin_authenticated
            
            if not is_authenticated:
                st.warning("This section is only accessible to administrators.")
                return
                
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
                
                if requests:
                    request_options = {r["id"]: f"{r['email']} ({r['requested_at'].strftime('%Y-%m-%d %H:%M')})" for r in requests}
                    selected_request_id = st.selectbox(
                        "Select Request to Process",
                        options=list(request_options.keys()),
                        format_func=lambda x: request_options[x]
                    )
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("Approve and Generate Code", type="primary"):
                            admin_id = 1  # Default admin ID
                            success, message, code = process_access_request(selected_request_id, admin_id, "approved", True)
                            if success:
                                st.success(f"{message} Access code: {code}")
                                st.rerun()
                            else:
                                st.error(message)
                    
                    with col2:
                        if st.button("Reject Request", type="secondary"):
                            admin_id = 1  # Default admin ID
                            success, message, _ = process_access_request(selected_request_id, admin_id, "rejected")
                            if success:
                                st.success(message)
                                st.rerun()
                            else:
                                st.error(message)
    
    with tab8:
        st.header("Database Management")
        
        st.markdown("""
        This section provides tools for managing the database. Please be careful with these operations
        as they may affect data integrity.
        """)
        
        # Database tables overview
        st.subheader("Database Tables")
        
        try:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    # Get list of tables
                    cur.execute("""
                        SELECT 
                            table_name,
                            (SELECT COUNT(*) FROM information_schema.columns WHERE table_name = t.table_name) as column_count,
                            pg_total_relation_size(quote_ident(table_name)) as table_size
                        FROM (
                            SELECT table_name FROM information_schema.tables 
                            WHERE table_schema = 'public'
                            ORDER BY table_name
                        ) t
                    """)
                    tables = cur.fetchall()
                    
                    if not tables:
                        st.info("No tables found in the database.")
                    else:
                        tables_df = pd.DataFrame(tables, columns=["table_name", "column_count", "size_bytes"])
                        
                        # Format size
                        tables_df["size"] = tables_df["size_bytes"].apply(lambda x: f"{x/1024:.1f} KB")
                        tables_df = tables_df.drop(columns=["size_bytes"])
                        
                        # Display tables
                        st.dataframe(tables_df, use_container_width=True)
                        
                        # Table row counts
                        st.subheader("Table Row Counts")
                        
                        row_counts = []
                        for table in tables_df["table_name"]:
                            try:
                                cur.execute(f"SELECT COUNT(*) FROM {table}")
                                count = cur.fetchone()[0]
                                row_counts.append({"table": table, "row_count": count})
                            except Exception:
                                row_counts.append({"table": table, "row_count": "Error"})
                        
                        row_counts_df = pd.DataFrame(row_counts)
                        st.dataframe(row_counts_df, use_container_width=True)
                        
                        # Table operations
                        st.subheader("Table Operations")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # View table structure
                            selected_table = st.selectbox(
                                "Select a table to view its structure",
                                options=tables_df["table_name"]
                            )
                            
                            if selected_table and st.button("View Table Structure", use_container_width=True):
                                try:
                                    cur.execute(f"""
                                        SELECT 
                                            column_name, 
                                            data_type, 
                                            is_nullable, 
                                            column_default
                                        FROM information_schema.columns
                                        WHERE table_name = %s
                                        ORDER BY ordinal_position
                                    """, (selected_table,))
                                    columns = cur.fetchall()
                                    
                                    columns_df = pd.DataFrame(columns, columns=[
                                        "column_name", "data_type", "is_nullable", "default_value"
                                    ])
                                    
                                    st.dataframe(columns_df, use_container_width=True)
                                except Exception as e:
                                    st.error(f"Error viewing table structure: {e}")
                        
                        with col2:
                            # View table data
                            data_table = st.selectbox(
                                "Select a table to view data",
                                options=tables_df["table_name"],
                                key="view_data_table"
                            )
                            
                            limit = st.slider("Number of rows to view", 5, 100, 20)
                            
                            if data_table and st.button("View Table Data", use_container_width=True):
                                try:
                                    cur.execute(f"SELECT * FROM {data_table} LIMIT {limit}")
                                    data = cur.fetchall()
                                    
                                    # Get column names
                                    cur.execute(f"""
                                        SELECT column_name
                                        FROM information_schema.columns
                                        WHERE table_name = %s
                                        ORDER BY ordinal_position
                                    """, (data_table,))
                                    columns = [col[0] for col in cur.fetchall()]
                                    
                                    data_df = pd.DataFrame(data, columns=columns)
                                    st.dataframe(data_df, use_container_width=True)
                                except Exception as e:
                                    st.error(f"Error viewing table data: {e}")
                        
                        # Data Export
                        st.subheader("Export Data as CSV")
                        export_table = st.selectbox(
                            "Select a table to export",
                            options=tables_df["table_name"],
                            key="export_table"
                        )
                        
                        if export_table and st.button("Export Table to CSV", type="primary", use_container_width=True):
                            success, csv_data = export_data_to_csv(export_table)
                            
                            if success:
                                # Create download button for CSV
                                st.download_button(
                                    label=f"Download {export_table} CSV",
                                    data=csv_data,
                                    file_name=f"{export_table}.csv",
                                    mime="text/csv",
                                    use_container_width=True,
                                )
                            else:
                                st.error(csv_data)  # Display error message
                        
                        # Database Cleanup
                        st.subheader("Database Cleanup")
                        st.warning("⚠️ **CAUTION**: The following operations will permanently delete data. Use with extreme care.")
                        
                        cleanup_table = st.selectbox(
                            "Select a table to clean up",
                            options=tables_df["table_name"],
                            key="cleanup_table"
                        )
                        
                        if cleanup_table:
                            # Confirm with a checkbox
                            confirm = st.checkbox(f"I understand that clearing the {cleanup_table} table will permanently delete all its data")
                            
                            if confirm and st.button("Clear Table Data", type="secondary", use_container_width=True):
                                if truncate_table(cleanup_table):
                                    st.success(f"Table {cleanup_table} has been cleared successfully.")
                                    st.rerun()
                                else:
                                    st.error(f"Failed to clear table {cleanup_table}.")
                                    
        except Exception as e:
            st.error(f"Error accessing database: {e}")