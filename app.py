"""
Streamlit Web UI for Human-in-the-Loop Email Assistant
Powered by LangGraph, LangChain, and Ollama llama3.2.
"""

import os
import uuid
import streamlit as st
from langgraph.types import Command

import agent
from agent import EmailState, build_agent, is_valid_email, is_valid_name, is_valid_request

# Page configuration
st.set_page_config(
    page_title="Human-in-the-Loop Email Assistant",
    page_icon="📧",
    layout="wide"
)

# Initialize cached LangGraph agent with MemorySaver
@st.cache_resource
def get_agent():
    return build_agent()


app_agent = get_agent()

# Session State initialization
if "status" not in st.session_state:
    st.session_state.status = "input"  # 'input', 'review', 'sent'
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "current_draft" not in st.session_state:
    st.session_state.current_draft = ""
if "agent_result" not in st.session_state:
    st.session_state.agent_result = ""
if "sender" not in st.session_state:
    st.session_state.sender = ""
if "to" not in st.session_state:
    st.session_state.to = ""
if "to_name" not in st.session_state:
    st.session_state.to_name = ""
if "request_text" not in st.session_state:
    st.session_state.request_text = ""

# Sidebar
with st.sidebar:
    st.header("⚙️ Agent Settings")
    st.markdown("**LLM:** Ollama `llama3.2`")
    st.markdown("**Framework:** LangGraph `StateGraph`")
    st.markdown("**Safety:** `langgraph.types.interrupt`")
    st.markdown("**Checkpointer:** `MemorySaver`")
    st.divider()

    st.subheader("📬 Live Outbox (`outbox.txt`)")
    if os.path.exists("outbox.txt"):
        with open("outbox.txt", "r", encoding="utf-8") as f:
            outbox_content = f.read()
        if outbox_content.strip():
            st.text_area("Recent Sent Emails", outbox_content, height=250, disabled=True)
            if st.button("Clear Outbox"):
                os.remove("outbox.txt")
                st.rerun()
        else:
            st.info("Outbox is currently empty.")
    else:
        st.info("No emails sent yet (`outbox.txt` will appear after first approved email).")

    st.divider()
    st.markdown("### 💡 How Human-in-the-Loop Works")
    st.caption(
        "1. **Draft Node**: The LLM creates an initial email draft.\n"
        "2. **Pause (`interrupt`)**: Graph halts execution before saving or sending.\n"
        "3. **Human Review**: You can request changes or approve.\n"
        "4. **Resume (`Command`)**: The graph resumes statefully from memory."
    )

# Main Title & Header
st.title("📧 Human-in-the-Loop Email Assistant")
st.markdown("Draft professional emails using **LangGraph** with a mandatory human approval pause before sending.")

# Stage 1: Input Form
if st.session_state.status == "input":
    st.subheader("1. Enter Email Details")
    with st.form("email_input_form"):
        col1, col2 = st.columns(2)
        with col1:
            sender = st.text_input(
                "Your Name (Sender)",
                value=st.session_state.sender or "Vamsi",
                placeholder="e.g., Vamsi"
            )
            to_name = st.text_input(
                "Recipient Name",
                value=st.session_state.to_name or "Priya",
                placeholder="e.g., Priya"
            )
        with col2:
            to_email = st.text_input(
                "Recipient Email",
                value=st.session_state.to or "priya@example.com",
                placeholder="e.g., priya@example.com"
            )

        request_text = st.text_area(
            "What should the email say?",
            value=st.session_state.request_text or "Ask Priya to move tomorrow's meeting to 3pm",
            placeholder="e.g., Ask Priya to move tomorrow's meeting to 3pm",
            height=100
        )

        submit = st.form_submit_button("Generate Draft", type="primary", use_container_width=True)

    if submit:
        # Validate inputs
        if not is_valid_name(sender):
            st.error("Please enter a valid sender name (cannot be empty or 'approve' / 'q').")
        elif not is_valid_name(to_name):
            st.error("Please enter a valid recipient name (cannot be empty or 'approve' / 'q').")
        elif not is_valid_email(to_email):
            st.error("Please enter a valid email address (e.g. name@domain.com).")
        elif not is_valid_request(request_text):
            st.error("Request must be at least 3 words.")
        else:
            # Store in session state
            st.session_state.sender = sender.strip()
            st.session_state.to_name = to_name.strip()
            st.session_state.to = to_email.strip()
            st.session_state.request_text = request_text.strip()

            new_thread_id = str(uuid.uuid4())
            st.session_state.thread_id = new_thread_id
            config = {"configurable": {"thread_id": new_thread_id}}

            initial_state: EmailState = {
                "request": st.session_state.request_text,
                "to": st.session_state.to,
                "to_name": st.session_state.to_name,
                "sender": st.session_state.sender,
                "draft": "",
                "feedback": "",
                "result": "",
            }

            with st.spinner("Drafting email with Ollama llama3.2..."):
                try:
                    out = app_agent.invoke(initial_state, config)
                    if "__interrupt__" in out:
                        st.session_state.current_draft = out["__interrupt__"][0].value.get("draft", "")
                        st.session_state.status = "review"
                        st.rerun()
                    else:
                        st.error("Graph completed without human review interruption.")
                except Exception as e:
                    st.error(f"Error invoking Ollama LLM: {e}")
                    st.info("Make sure Ollama is running (`ollama serve`) and model `llama3.2` is pulled (`ollama pull llama3.2`).")

# Stage 2: Human Review & Feedback Loop
elif st.session_state.status == "review":
    st.subheader("2. Human Review & Approval")
    st.warning("⚠️ **Execution Paused**: Human approval is required before the email is sent.")

    # Email Preview Card
    with st.container(border=True):
        st.markdown(f"**To:** `{st.session_state.to}` ({st.session_state.to_name})")
        st.markdown(f"**From:** `{st.session_state.sender}`")
        st.markdown(f"**Subject:** `Regarding: {st.session_state.request_text[:40]}...`")
        st.divider()
        st.text_area("Generated Draft", value=st.session_state.current_draft, height=180, disabled=True)

    st.markdown("### Your Decision:")
    col_review_left, col_review_right = st.columns([3, 2])

    with col_review_left:
        st.markdown("#### ✏️ Request Changes")
        feedback = st.text_input(
            "What would you like changed?",
            placeholder="e.g., make it shorter, mention 3:30pm instead, add phone number...",
            key="feedback_input"
        )
        if st.button("Request Re-draft", use_container_width=True):
            if not feedback.strip():
                st.error("Please enter revision instructions before requesting changes.")
            elif feedback.strip().lower() in ["approve", "a", "yes", "y"]:
                st.error("To approve, please click the green 'Approve & Send Email' button on the right.")
            else:
                config = {"configurable": {"thread_id": st.session_state.thread_id}}
                with st.spinner("Redrafting email with your feedback..."):
                    try:
                        out = app_agent.invoke(Command(resume=feedback.strip()), config)
                        if "__interrupt__" in out:
                            st.session_state.current_draft = out["__interrupt__"][0].value.get("draft", "")
                            st.success("Draft updated successfully!")
                            st.rerun()
                    except Exception as e:
                        st.error(f"Error resuming graph: {e}")

    with col_review_right:
        st.markdown("#### ✅ Approve & Send")
        st.write("Confirm that the draft is accurate. This will trigger the `send_email` tool and save the email to `outbox.txt`.")
        if st.button("🚀 Approve & Send Email", type="primary", use_container_width=True):
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            with st.spinner("Executing send_email tool..."):
                try:
                    out = app_agent.invoke(Command(resume="approve"), config)
                    st.session_state.agent_result = out.get("result", "Email sent successfully!")
                    st.session_state.status = "sent"
                    st.rerun()
                except Exception as e:
                    st.error(f"Error sending email: {e}")

    st.divider()
    if st.button("Cancel & Start Over", type="secondary"):
        st.session_state.status = "input"
        st.session_state.thread_id = None
        st.session_state.current_draft = ""
        st.rerun()

# Stage 3: Email Sent Confirmation
elif st.session_state.status == "sent":
    st.success("🎉 Email Approved & Dispatched!")
    st.balloons()

    st.markdown(f"**Agent Message:** `{st.session_state.agent_result}`")

    with st.expander("View Final Sent Email"):
        st.markdown(f"**To:** `{st.session_state.to}`")
        st.markdown(f"**Sender:** `{st.session_state.sender}`")
        st.text_area("Final Body", value=st.session_state.current_draft, height=160, disabled=True)

    if st.button("Draft Another Email", type="primary"):
        st.session_state.status = "input"
        st.session_state.thread_id = None
        st.session_state.current_draft = ""
        st.session_state.agent_result = ""
        st.rerun()
