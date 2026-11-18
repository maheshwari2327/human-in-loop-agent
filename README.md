# Human-in-the-Loop Email Assistant

An interactive Human-in-the-Loop (HITL) email drafting assistant built with Python, **LangChain**, and **LangGraph**, powered by local LLMs via **Ollama** (`llama3.2`).

The assistant drafts an email based on user intent and **pauses** execution right before the irreversible action (saving/sending the email). The user can either request revisions iteratively or approve the draft.

Available as both a **Terminal CLI** and a **Streamlit Web Application**.

---

## Architecture & Graph Flow

```
   [START]
      │
      ▼
   [draft] ◄───────────────┐
      │                    │
      ▼                    │
 [human_review]            │ (human feedback)
  (interrupt)              │
      │                    │
  {Decision} ──────────────┘
      │
      │ (approved: 'approve' / 'a' / 'yes' / 'y')
      ▼
    [send] ──► [END]
 (send_email)
```

- **`MemorySaver` Checkpointer**: Saves paused graph state across interrupts.
- **`interrupt()` & `Command(resume=...)`**: Native LangGraph mechanism to pause execution and await user input.
- **`thread_id` Isolation**: Every email transaction receives an isolated conversation session.

---

## File Structure

- [app.py](file:///c:/Users/HP/Desktop/gen%20ai%20exercises/day%2019/human%20in%20loop%20agent/app.py): **Streamlit Web Interface** for visual drafting, review, revision, and outbox monitoring.
- [agent.py](file:///c:/Users/HP/Desktop/gen%20ai%20exercises/day%2019/human%20in%20loop%20agent/agent.py): Main StateGraph definition, node functions, `run_once` runner, input validation, and interactive CLI.
- [tools.py](file:///c:/Users/HP/Desktop/gen%20ai%20exercises/day%2019/human%20in%20loop%20agent/tools.py): `@tool` definition for `send_email` appending to `outbox.txt`.
- [requirements.txt](file:///c:/Users/HP/Desktop/gen%20ai%20exercises/day%2019/human%20in%20loop%20agent/requirements.txt): Python dependencies (`langchain`, `langchain-ollama`, `langgraph`, `requests`, `streamlit`).
- [test_agent.py](file:///c:/Users/HP/Desktop/gen%20ai%20exercises/day%2019/human%20in%20loop%20agent/test_agent.py): Automated unit and flow tests covering validation, tool execution, graph interrupts, and redrafting loops.

---

## Prerequisites

1. **Python 3.10+**
2. **Ollama**:
   - Install from [ollama.com](https://ollama.com/)
   - Pull the `llama3.2` model:
     ```bash
     ollama pull llama3.2
     ```
   - Ensure Ollama is running (`ollama serve`).

---

## Installation

```bash
pip install -r requirements.txt
```

---

## 🌐 Running in the Web Browser (Streamlit)

Launch the web application with:

```bash
streamlit run app.py
```

This will automatically open the browser at **`http://localhost:8501`**.

### Web Features:
1. **Interactive Form**: Enter sender, recipient email, recipient name, and request.
2. **Execution Paused Banner**: Displays the LangGraph interrupt pause right before sending.
3. **Live Re-drafting**: Type notes like "make it shorter" and click **Request Re-draft** to loop through revisions.
4. **One-Click Approval**: Click **🚀 Approve & Send Email** to trigger the `send_email` tool.
5. **Sidebar Outbox Viewer**: Inspect all saved emails directly in the web UI.

---

## 💻 Running in Terminal CLI

Launch the interactive CLI:

```bash
python agent.py
```

---

## 🧪 Running Tests

Run the test suite:

```bash
python test_agent.py
```
