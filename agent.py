"""
Human-in-the-Loop Email Assistant CLI
Built with LangChain, LangGraph, and Ollama llama3.2.
"""

import re
import sys
import uuid
from typing import TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from tools import send_email

# -----------------------------------------------------------------------------
# SYSTEM PROMPT
# -----------------------------------------------------------------------------
DRAFT_PROMPT = """You are an email assistant. Write a short, polite email (max 80 words)
for the user's request. Start with "Dear <recipient name>," and end with
"Best regards," followed by the sender's name. Use the real names you are given and
NEVER leave placeholders like [Recipient] or [Your Name].
Return ONLY the email body — no subject line, no extra comments."""

# -----------------------------------------------------------------------------
# LLM INSTANCE
# -----------------------------------------------------------------------------
llm = ChatOllama(model="llama3.2", temperature=0.4)

# -----------------------------------------------------------------------------
# STATE DEFINITION
# -----------------------------------------------------------------------------
class EmailState(TypedDict):
    request: str
    to: str
    to_name: str
    sender: str
    draft: str
    feedback: str
    result: str


# -----------------------------------------------------------------------------
# NODES
# -----------------------------------------------------------------------------
def draft_node(state: EmailState) -> dict:
    """Draft or re-draft the email using Ollama llama3.2 and DRAFT_PROMPT."""
    prompt_content = (
        f"Recipient name: {state['to_name']}\n"
        f"Sender name: {state['sender']}\n"
        f"Request: {state['request']}"
    )

    if state.get("draft") and state.get("feedback"):
        prompt_content += (
            f"\n\nPrevious draft:\n"
            f"{state['draft']}\n\n"
            f"Human feedback — please apply it:\n"
            f"{state['feedback']}"
        )

    messages = [
        SystemMessage(content=DRAFT_PROMPT),
        HumanMessage(content=prompt_content),
    ]

    try:
        response = llm.invoke(messages)
        draft_text = str(response.content).strip()
    except Exception as exc:
        print(f"\n[!] Ollama connection error: {exc}")
        print("Please ensure Ollama is running ('ollama serve') and 'llama3.2' is installed ('ollama pull llama3.2').")
        raise exc

    return {"draft": draft_text}


def human_review_node(state: EmailState) -> dict:
    """Pause graph execution to request human approval or revision feedback."""
    decision = interrupt({
        "to": state["to"],
        "draft": state["draft"]
    })
    return {"feedback": str(decision).strip() if decision is not None else ""}


def send_node(state: EmailState) -> dict:
    """Execute the risky action: calling the send_email tool."""
    subject = f"Regarding: {state['request'][:50]}"
    result = send_email.invoke({
        "to": state["to"],
        "subject": subject,
        "body": state["draft"]
    })
    return {"result": str(result)}


def route_feedback(state: EmailState) -> str:
    """Conditional edge: approve -> send -> END; anything else -> draft."""
    feedback = state.get("feedback", "").strip().lower()
    if feedback in ["approve", "a", "yes", "y"]:
        return "send"
    return "draft"


# -----------------------------------------------------------------------------
# GRAPH BUILDER
# -----------------------------------------------------------------------------
def build_agent(custom_llm=None):
    """Construct and compile the LangGraph StateGraph with MemorySaver checkpointer."""
    global llm
    if custom_llm is not None:
        llm = custom_llm

    builder = StateGraph(EmailState)

    builder.add_node("draft", draft_node)
    builder.add_node("human_review", human_review_node)
    builder.add_node("send", send_node)

    builder.add_edge(START, "draft")
    builder.add_edge("draft", "human_review")
    builder.add_conditional_edges(
        "human_review",
        route_feedback,
        {
            "send": "send",
            "draft": "draft"
        }
    )
    builder.add_edge("send", END)

    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


# -----------------------------------------------------------------------------
# RUNNER
# -----------------------------------------------------------------------------
def run_once(agent, initial_state: EmailState) -> dict:
    """Run a single email lifecycle, handling interrupts and resumption."""
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    out = agent.invoke(initial_state, config)

    while "__interrupt__" in out:
        interrupt_info = out["__interrupt__"][0].value
        to_email = interrupt_info.get("to", initial_state.get("to", ""))
        draft_text = interrupt_info.get("draft", "")

        print(f"\n📧 DRAFT for {to_email}:")
        print("-" * 40)
        print(draft_text)
        print("-" * 40)

        while True:
            try:
                answer = input("Reply with ONLY 'approve' to send, or tell me what to change: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nExiting...")
                sys.exit(0)

            if answer.lower() in ["quit", "exit", "q"]:
                print("Exiting...")
                sys.exit(0)

            # An empty answer must NEVER count as approval
            if not answer:
                print("An empty response is not permitted. Please enter 'approve' or specify changes.")
                continue

            break

        out = agent.invoke(Command(resume=answer), config)

    result = out.get("result", "")
    if result:
        print(f"\nAgent: {result}\n")

    return out


# -----------------------------------------------------------------------------
# INPUT VALIDATION HELPERS
# -----------------------------------------------------------------------------
EMAIL_REGEX = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def is_valid_name(val: str) -> bool:
    cleaned = val.strip().lower()
    return bool(cleaned) and cleaned not in ["approve", "q", "quit", "exit"]


def is_valid_request(val: str) -> bool:
    words = val.strip().split()
    return len(words) >= 3


def is_valid_email(val: str) -> bool:
    return bool(EMAIL_REGEX.match(val.strip()))


def ask_until_valid(prompt: str, is_ok, hint: str) -> str:
    """Prompt the user until the input satisfies is_ok(value)."""
    while True:
        try:
            val = input(prompt).strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            sys.exit(0)

        if val.lower() in ["quit", "exit", "q"]:
            print("Exiting...")
            sys.exit(0)

        if is_ok(val):
            return val

        print(f"  [!] {hint}\n")


# -----------------------------------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("      HUMAN-IN-THE-LOOP EMAIL ASSISTANT")
    print("=" * 60)
    print("Type 'quit', 'exit', or 'q' at any prompt to exit.\n")

    agent = build_agent()

    # 1) Sender name (asked once at start-up)
    sender = ask_until_valid(
        "1) Your name (e.g., Vamsi, Alice): ",
        is_valid_name,
        "Name cannot be empty, 'approve', or 'q'."
    )

    while True:
        print()
        # 2) Request
        request = ask_until_valid(
            "2) Request (e.g., Ask Priya to move tomorrow's meeting to 3pm): ",
            is_valid_request,
            "Request must be at least 3 words."
        )

        # 3) Recipient email
        to_email = ask_until_valid(
            "3) Send to (email) (e.g., priya@example.com): ",
            is_valid_email,
            "Email must look like name@domain.com."
        )

        # 4) Recipient name
        to_name = ask_until_valid(
            "4) Recipient's name (e.g., Priya): ",
            is_valid_name,
            "Recipient name cannot be empty, 'approve', or 'q'."
        )

        initial_state: EmailState = {
            "request": request,
            "to": to_email,
            "to_name": to_name,
            "sender": sender,
            "draft": "",
            "feedback": "",
            "result": "",
        }

        run_once(agent, initial_state)

        try:
            again = input("Would you like to draft another email? (y/n, default 'y'): ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            sys.exit(0)

        if again in ["n", "no", "q", "quit", "exit"]:
            print("Goodbye!")
            break


if __name__ == "__main__":
    main()
