import streamlit as st
import pandas as pd
from datetime import datetime
from utils.db_utils import get_connection

def save_message(sender_id, message_text, receiver_id=None, is_to_admin=False, is_from_admin=False):
    """
    Save a message to the database.
    
    Args:
        sender_id: ID of the message sender
        message_text: Content of the message
        receiver_id: ID of the message receiver (optional)
        is_to_admin: Whether message is sent to admin
        is_from_admin: Whether message is from admin
        
    Returns:
        bool: Success status
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO workshop_messages 
                    (sender_id, receiver_id, is_to_admin, is_from_admin, message_text, sent_at) 
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id
                """, (sender_id, receiver_id, is_to_admin, is_from_admin, message_text, datetime.now()))
                result = cur.fetchone()
                return result is not None
    except Exception as e:
        st.error(f"Error saving message: {e}")
        return False

def get_user_messages(user_id):
    """
    Get messages for a user (both sent and received).
    
    Args:
        user_id: ID of the user
        
    Returns:
        pd.DataFrame: DataFrame with message data
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                # Get messages where user is sender or receiver
                cur.execute("""
                    SELECT 
                        m.id,
                        m.message_text,
                        m.sent_at,
                        m.read_at,
                        m.is_to_admin,
                        m.is_from_admin,
                        sender.username as sender_name,
                        CASE 
                            WHEN m.receiver_id IS NOT NULL THEN receiver.username
                            ELSE 'Admin'
                        END as receiver_name,
                        CASE 
                            WHEN m.sender_id = %s THEN true
                            ELSE false
                        END as is_sent_by_user
                    FROM workshop_messages m
                    JOIN workshop_users sender ON m.sender_id = sender.id
                    LEFT JOIN workshop_users receiver ON m.receiver_id = receiver.id
                    WHERE (m.sender_id = %s OR m.receiver_id = %s)
                    ORDER BY m.sent_at DESC
                """, (user_id, user_id, user_id))
                messages = cur.fetchall()
                
                if not messages:
                    return pd.DataFrame()
                
                messages_df = pd.DataFrame(messages, columns=[
                    "id", "message_text", "sent_at", "read_at", "is_to_admin", 
                    "is_from_admin", "sender_name", "receiver_name", "is_sent_by_user"
                ])
                
                # Format timestamps
                if 'sent_at' in messages_df:
                    messages_df['sent_at'] = pd.to_datetime(messages_df['sent_at'])
                if 'read_at' in messages_df:
                    messages_df['read_at'] = pd.to_datetime(messages_df['read_at'])
                    
                return messages_df
    except Exception as e:
        st.error(f"Error retrieving messages: {e}")
        return pd.DataFrame()

def get_admin_messages():
    """
    Get all messages for admin (messages to/from admin).
    
    Returns:
        pd.DataFrame: DataFrame with message data
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                # Get messages where admin is involved
                cur.execute("""
                    SELECT 
                        m.id,
                        m.message_text,
                        m.sent_at,
                        m.read_at,
                        m.is_to_admin,
                        m.is_from_admin,
                        sender.username as sender_name,
                        sender.id as sender_id,
                        CASE 
                            WHEN m.receiver_id IS NOT NULL THEN receiver.username
                            ELSE 'Admin'
                        END as receiver_name,
                        CASE 
                            WHEN m.receiver_id IS NOT NULL THEN receiver.id
                            ELSE NULL
                        END as receiver_id
                    FROM workshop_messages m
                    JOIN workshop_users sender ON m.sender_id = sender.id
                    LEFT JOIN workshop_users receiver ON m.receiver_id = receiver.id
                    WHERE m.is_to_admin = true OR m.is_from_admin = true
                    ORDER BY m.sent_at DESC
                """)
                messages = cur.fetchall()
                
                if not messages:
                    return pd.DataFrame()
                
                messages_df = pd.DataFrame(messages, columns=[
                    "id", "message_text", "sent_at", "read_at", "is_to_admin", 
                    "is_from_admin", "sender_name", "sender_id", "receiver_name", "receiver_id"
                ])
                
                # Format timestamps
                if 'sent_at' in messages_df:
                    messages_df['sent_at'] = pd.to_datetime(messages_df['sent_at'])
                if 'read_at' in messages_df:
                    messages_df['read_at'] = pd.to_datetime(messages_df['read_at'])
                    
                return messages_df
    except Exception as e:
        st.error(f"Error retrieving admin messages: {e}")
        return pd.DataFrame()

def mark_message_as_read(message_id):
    """
    Mark a message as read.
    
    Args:
        message_id: ID of the message
        
    Returns:
        bool: Success status
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    UPDATE workshop_messages
                    SET read_at = %s
                    WHERE id = %s AND read_at IS NULL
                """, (datetime.now(), message_id))
                conn.commit()
                return True
    except Exception as e:
        st.error(f"Error marking message as read: {e}")
        return False

def get_unread_message_count(user_id):
    """
    Get count of unread messages for a user.
    
    Args:
        user_id: ID of the user
        
    Returns:
        int: Count of unread messages
    """
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT COUNT(*)
                    FROM workshop_messages
                    WHERE receiver_id = %s AND read_at IS NULL
                """, (user_id,))
                count = cur.fetchone()[0]
                
                # Also check for admin messages if user is admin
                if st.session_state.get("is_admin", False):
                    cur.execute("""
                        SELECT COUNT(*)
                        FROM workshop_messages
                        WHERE is_to_admin = true AND read_at IS NULL
                    """)
                    admin_count = cur.fetchone()[0]
                    count += admin_count
                    
                return count
    except Exception as e:
        st.error(f"Error getting unread message count: {e}")
        return 0

def display_user_messaging():
    """
    Display messaging interface for regular users.
    """
    st.title("Messages")
    
    if not st.session_state.is_authenticated:
        st.warning("Please log in to access messaging.")
        if st.button("Sign In / Sign Up"):
            st.session_state.current_page = "Auth"
            st.rerun()
        return
    
    user_id = st.session_state.user["id"]
    username = st.session_state.user["username"]
    
    # Get user messages
    messages_df = get_user_messages(user_id)
    
    # Compose new message
    st.subheader("Send Message to Admin")
    
    message_text = st.text_area("Message", height=100, 
                               placeholder="Type your message to the admin here...")
    
    if st.button("Send Message", use_container_width=True):
        if not message_text.strip():
            st.error("Please enter a message before sending.")
        else:
            success = save_message(
                sender_id=user_id,
                message_text=message_text,
                is_to_admin=True
            )
            
            if success:
                st.success("Message sent successfully!")
                # Clear the input field by rerunning
                st.rerun()
            else:
                st.error("Failed to send message. Please try again.")
    
    # Display existing messages
    st.subheader("Your Messages")
    
    if messages_df.empty:
        st.info("You don't have any messages yet.")
    else:
        # Create tabs for grouping messages
        all_tab, sent_tab, received_tab = st.tabs(["All Messages", "Sent Messages", "Received Messages"])
        
        with all_tab:
            _display_messages(messages_df)
        
        with sent_tab:
            sent_df = messages_df[messages_df['is_sent_by_user']]
            if sent_df.empty:
                st.info("You haven't sent any messages yet.")
            else:
                _display_messages(sent_df)
        
        with received_tab:
            received_df = messages_df[~messages_df['is_sent_by_user']]
            if received_df.empty:
                st.info("You haven't received any messages yet.")
            else:
                _display_messages(received_df)

def display_admin_messaging():
    """
    Display messaging interface for admin.
    """
    st.title("Admin Messaging")
    
    if not st.session_state.get("is_admin", False):
        st.warning("This section is only accessible to administrators.")
        return
    
    # Get all messages
    messages_df = get_admin_messages()
    
    # Get list of users for sending messages
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, username FROM workshop_users ORDER BY username")
            users = cur.fetchall()
    
    # Compose new message
    st.subheader("Send Message to User")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        user_options = {user[0]: user[1] for user in users}
        selected_user_id = st.selectbox(
            "Select User", 
            options=list(user_options.keys()),
            format_func=lambda x: user_options[x]
        )
    
    with col2:
        st.write("")
        st.write("")
        refresh = st.button("Refresh", use_container_width=True)
        if refresh:
            st.rerun()
    
    message_text = st.text_area("Message", height=100, 
                               placeholder="Type your message to the user here...")
    
    if st.button("Send Message", use_container_width=True):
        if not message_text.strip():
            st.error("Please enter a message before sending.")
        else:
            # Get admin ID (using a default ID of 1 for demo purposes)
            admin_id = 1
            
            success = save_message(
                sender_id=admin_id,
                message_text=message_text,
                receiver_id=selected_user_id,
                is_from_admin=True
            )
            
            if success:
                st.success(f"Message sent successfully to {user_options[selected_user_id]}!")
                # Clear the input field by rerunning
                st.rerun()
            else:
                st.error("Failed to send message. Please try again.")
    
    # Display existing messages
    st.subheader("All Messages")
    
    if messages_df.empty:
        st.info("There are no messages yet.")
    else:
        # Create tabs for all messages and by user
        all_tab, by_user_tab = st.tabs(["All Messages", "By User"])
        
        with all_tab:
            _display_admin_messages(messages_df)
        
        with by_user_tab:
            if 'sender_name' in messages_df.columns:
                # Get unique users who have sent or received messages
                unique_senders = pd.concat([
                    messages_df[['sender_id', 'sender_name']].rename(
                        columns={'sender_id': 'user_id', 'sender_name': 'username'}
                    ),
                    messages_df[['receiver_id', 'receiver_name']].rename(
                        columns={'receiver_id': 'user_id', 'receiver_name': 'username'}
                    )
                ])
                unique_users = unique_senders.dropna().drop_duplicates('user_id')
                
                if not unique_users.empty:
                    selected_username = st.selectbox(
                        "Select User",
                        options=unique_users['username'].unique()
                    )
                    
                    user_id = unique_users[unique_users['username'] == selected_username]['user_id'].iloc[0]
                    user_messages = messages_df[
                        (messages_df['sender_id'] == user_id) | 
                        (messages_df['receiver_id'] == user_id)
                    ]
                    
                    if user_messages.empty:
                        st.info(f"No messages with {selected_username}.")
                    else:
                        _display_admin_messages(user_messages)
                else:
                    st.info("No user messages found.")
            else:
                st.info("No messages by user found.")

def _display_messages(messages_df):
    """
    Helper function to display messages in a consistent format.
    
    Args:
        messages_df: DataFrame containing messages
    """
    for _, row in messages_df.iterrows():
        message_container = st.container(border=True)
        
        with message_container:
            col1, col2 = st.columns([5, 1])
            
            with col1:
                if row['is_sent_by_user']:
                    st.write(f"**To:** {row['receiver_name']}")
                else:
                    st.write(f"**From:** {row['sender_name']}")
            
            with col2:
                st.write(f"**{row['sent_at'].strftime('%Y-%m-%d %H:%M')}**")
            
            st.markdown(f"{row['message_text']}")
            
            # Show read status
            if row['read_at'] is not None and not row['is_sent_by_user']:
                st.caption(f"Read at {row['read_at'].strftime('%Y-%m-%d %H:%M')}")
            elif not row['is_sent_by_user'] and row['read_at'] is None:
                # Mark as read when viewed
                mark_message_as_read(row['id'])
                st.caption("Just read")

def _display_admin_messages(messages_df):
    """
    Helper function to display admin messages in a consistent format.
    
    Args:
        messages_df: DataFrame containing messages
    """
    for _, row in messages_df.iterrows():
        message_container = st.container(border=True)
        
        with message_container:
            col1, col2 = st.columns([5, 1])
            
            with col1:
                if row['is_from_admin']:
                    st.write(f"**To:** {row['receiver_name']}")
                else:
                    st.write(f"**From:** {row['sender_name']}")
            
            with col2:
                st.write(f"**{row['sent_at'].strftime('%Y-%m-%d %H:%M')}**")
            
            st.markdown(f"{row['message_text']}")
            
            # Show read status
            if row['read_at'] is not None:
                st.caption(f"Read at {row['read_at'].strftime('%Y-%m-%d %H:%M')}")
            elif row['is_to_admin'] and row['read_at'] is None:
                # Mark as read when viewed by admin
                mark_message_as_read(row['id'])
                st.caption("Just read")