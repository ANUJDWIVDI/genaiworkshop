import streamlit as st
from utils.db_utils import get_user_progress, update_module_progress, get_total_score
import plotly.express as px
import pandas as pd

def display_progress_dashboard(user_id):
    """
    Display a dashboard with user progress information.
    
    Args:
        user_id: ID of the current user
    """
    st.header("Your Progress")
    
    # Get user progress data
    progress_data = get_user_progress(user_id)
    
    if not progress_data:
        st.warning("No progress data available yet. Start exploring the modules!")
        return
    
    # Get total score
    total_score = get_total_score(user_id)
    
    # Calculate completion percentage
    completed_modules = sum(1 for p in progress_data if p["completed"])
    total_modules = len(progress_data)
    completion_percentage = (completed_modules / total_modules) * 100 if total_modules > 0 else 0
    
    # Display summary metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Modules Completed",
            value=f"{completed_modules}/{total_modules}",
            delta=f"{completion_percentage:.1f}%"
        )
    
    with col2:
        st.metric(
            label="Total Score",
            value=total_score
        )
    
    with col3:
        max_possible_score = total_modules * 100
        st.metric(
            label="Overall Grade",
            value=f"{(total_score / max_possible_score) * 100:.1f}%" if max_possible_score > 0 else "0%"
        )
    
    st.markdown("---")
    
    # Create a DataFrame for visualization
    df = pd.DataFrame(progress_data)
    df["score"] = df["score"].fillna(0)
    
    # Bar chart for module scores
    fig = px.bar(
        df,
        x="module_name",
        y="score",
        color="completed",
        labels={"module_name": "Module", "score": "Score", "completed": "Completed"},
        title="Module Scores",
        color_discrete_map={True: "green", False: "gray"}
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Detailed progress table
    st.subheader("Module Details")
    
    for module in progress_data:
        module_name = module["module_name"]
        completed = module["completed"]
        score = module["score"]
        completion_date = module["completion_date"]
        
        # Create a status indicator
        if completed:
            status = "✅ Completed"
            date_str = f" on {completion_date.strftime('%Y-%m-%d')}" if completion_date else ""
            score_str = f" with score: {score}/100"
        else:
            status = "⏳ In Progress"
            date_str = ""
            score_str = ""
        
        st.markdown(f"**{module_name}**: {status}{date_str}{score_str}")

def mark_module_completed(user_id, module_name, score=None):
    """
    Mark a module as completed with an optional score.
    
    Args:
        user_id: ID of the current user
        module_name: Name of the module
        score: Score achieved in the module (default: None)
        
    Returns:
        bool: Success status
    """
    # If score is not provided, use a default value
    if score is None:
        # Get current progress to check current score
        progress_data = get_user_progress(user_id)
        current_score = 0
        
        for module in progress_data:
            if module["module_name"] == module_name:
                current_score = module["score"] or 0
                break
        
        # Use current score or default to 0
        score = current_score
    
    # Update the module progress
    success = update_module_progress(user_id, module_name, True, score)
    
    if success:
        # Update session state
        if 'user_progress' in st.session_state:
            for i, module in enumerate(st.session_state.user_progress):
                if module["module_name"] == module_name:
                    st.session_state.user_progress[i]["completed"] = True
                    st.session_state.user_progress[i]["score"] = score
                    break
        
        # Update total score in session state
        st.session_state.total_score = get_total_score(user_id)
    
    return success

def update_module_score(user_id, module_name, score):
    """
    Update the score for a module.
    
    Args:
        user_id: ID of the current user
        module_name: Name of the module
        score: Score achieved in the module
        
    Returns:
        bool: Success status
    """
    # Get current progress to check completion status
    progress_data = get_user_progress(user_id)
    is_completed = False
    
    for module in progress_data:
        if module["module_name"] == module_name:
            is_completed = module["completed"]
            break
    
    # Update the module progress
    success = update_module_progress(user_id, module_name, is_completed, score)
    
    if success:
        # Update session state
        if 'user_progress' in st.session_state:
            for i, module in enumerate(st.session_state.user_progress):
                if module["module_name"] == module_name:
                    st.session_state.user_progress[i]["score"] = score
                    break
        
        # Update total score in session state
        st.session_state.total_score = get_total_score(user_id)
    
    return success

def get_module_completion_status(module_name):
    """
    Get the completion status of a module for the current user.
    
    Args:
        module_name: Name of the module
        
    Returns:
        tuple: (is_completed, score)
    """
    if 'user_progress' not in st.session_state:
        return False, 0
    
    for module in st.session_state.user_progress:
        if module["module_name"] == module_name:
            return module["completed"], module["score"] or 0
    
    return False, 0