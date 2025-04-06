import streamlit as st
import random
from utils.db_utils import get_quiz_questions, save_quiz_response
from utils.progress_utils import update_module_score

def display_quiz(user_id, module_name):
    """
    Display a quiz for a module and handle user responses.
    
    Args:
        user_id: ID of the current user
        module_name: Name of the module
    """
    st.header(f"Quiz: {module_name}")
    
    # Get quiz questions for the module
    questions = get_quiz_questions(module_name)
    
    if not questions:
        st.warning("No quiz questions available for this module yet.")
        return
    
    # Initialize quiz state in session
    if 'quiz_state' not in st.session_state:
        st.session_state.quiz_state = {
            "current_question": 0,
            "correct_answers": 0,
            "total_questions": len(questions),
            "answered": [False] * len(questions),
            "user_answers": [""] * len(questions),
            "is_correct": [False] * len(questions),
            "completed": False
        }
    
    # Get current quiz state
    quiz_state = st.session_state.quiz_state
    
    # Display quiz progress
    progress_text = f"Question {quiz_state['current_question'] + 1} of {quiz_state['total_questions']}"
    st.progress(quiz_state['current_question'] / quiz_state['total_questions'])
    st.markdown(f"**{progress_text}**")
    
    # Display current question
    current_q = questions[quiz_state['current_question']]
    st.subheader(current_q["question_text"])
    
    # Create radio buttons for options
    options = current_q["options"]
    
    # Check if the current question has been answered
    if not quiz_state["answered"][quiz_state["current_question"]]:
        # Display options as radio buttons
        user_answer = st.radio(
            "Select your answer:",
            options=list(options.keys()),
            format_func=lambda x: f"{x}. {options[x]}"
        )
        
        # Submit button
        if st.button("Submit Answer"):
            # Record the user's answer
            quiz_state["user_answers"][quiz_state["current_question"]] = user_answer
            quiz_state["answered"][quiz_state["current_question"]] = True
            
            # Check if the answer is correct
            correct_answer = current_q["correct_answer"]
            is_correct = user_answer == correct_answer
            quiz_state["is_correct"][quiz_state["current_question"]] = is_correct
            
            if is_correct:
                quiz_state["correct_answers"] += 1
            
            # Save the response in the database
            save_quiz_response(user_id, current_q["id"], user_answer, is_correct)
            
            # Update the session state to show the result
            st.rerun()
    else:
        # Show the user's answer and whether it was correct
        user_answer = quiz_state["user_answers"][quiz_state["current_question"]]
        is_correct = quiz_state["is_correct"][quiz_state["current_question"]]
        
        # Display the options with the correct answer highlighted
        for option_key, option_text in options.items():
            if option_key == current_q["correct_answer"]:
                st.success(f"{option_key}. {option_text} ✅")
            elif option_key == user_answer and not is_correct:
                st.error(f"{option_key}. {option_text} ❌")
            else:
                st.write(f"{option_key}. {option_text}")
        
        # Display explanation
        if "explanation" in current_q and current_q["explanation"]:
            st.info(f"**Explanation:** {current_q['explanation']}")
        
        # Navigation buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if quiz_state["current_question"] > 0:
                if st.button("Previous Question"):
                    quiz_state["current_question"] -= 1
                    st.rerun()
        
        with col2:
            if quiz_state["current_question"] < quiz_state["total_questions"] - 1:
                if st.button("Next Question"):
                    quiz_state["current_question"] += 1
                    st.rerun()
            elif not quiz_state["completed"] and all(quiz_state["answered"]):
                if st.button("Finish Quiz"):
                    quiz_state["completed"] = True
                    st.rerun()
    
    # If all questions have been answered and quiz is marked as completed, show the results
    if quiz_state["completed"]:
        show_quiz_results(user_id, module_name, quiz_state)

def show_quiz_results(user_id, module_name, quiz_state):
    """
    Display quiz results and update user score.
    
    Args:
        user_id: ID of the current user
        module_name: Name of the module
        quiz_state: Current quiz state
    """
    st.header("Quiz Results")
    
    # Calculate score
    correct_answers = quiz_state["correct_answers"]
    total_questions = quiz_state["total_questions"]
    score = int((correct_answers / total_questions) * 100) if total_questions > 0 else 0
    
    # Display score
    st.markdown(f"### Your Score: {score}%")
    st.markdown(f"**Correct Answers:** {correct_answers} out of {total_questions}")
    
    # Update module score
    update_module_score(user_id, module_name, score)
    
    # Display feedback based on score
    if score >= 90:
        st.success("🎉 Excellent! You've mastered this topic!")
    elif score >= 70:
        st.success("👍 Good job! You have a solid understanding of the material.")
    elif score >= 50:
        st.warning("📚 You're on the right track, but might want to review some concepts.")
    else:
        st.error("💡 It looks like you need more practice with this topic.")
    
    # Option to retake the quiz
    if st.button("Retake Quiz"):
        # Reset quiz state
        st.session_state.quiz_state = {
            "current_question": 0,
            "correct_answers": 0,
            "total_questions": total_questions,
            "answered": [False] * total_questions,
            "user_answers": [""] * total_questions,
            "is_correct": [False] * total_questions,
            "completed": False
        }
        st.rerun()
    
    # Option to continue to next module
    st.button("Continue Learning", on_click=lambda: setattr(st.session_state, 'quiz_state', None))

def create_sample_quiz_questions(module_name, questions_data):
    """
    Create sample quiz questions for a module.
    
    Args:
        module_name: Name of the module
        questions_data: List of question data dictionaries
    """
    # This function can be used to add sample quiz questions to the database
    # For testing purposes
    pass