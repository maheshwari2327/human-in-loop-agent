import os
import unittest
from unittest.mock import MagicMock, patch
from langchain_core.messages import AIMessage
from langgraph.types import Command

import agent
import tools


class TestEmailAssistant(unittest.TestCase):
    def setUp(self):
        if os.path.exists("outbox.txt"):
            os.remove("outbox.txt")

    def tearDown(self):
        if os.path.exists("outbox.txt"):
            os.remove("outbox.txt")

    def test_validation_helpers(self):
        # Name validation
        self.assertTrue(agent.is_valid_name("Vamsi"))
        self.assertTrue(agent.is_valid_name("Alice Bob"))
        self.assertFalse(agent.is_valid_name("approve"))
        self.assertFalse(agent.is_valid_name("APPROVE"))
        self.assertFalse(agent.is_valid_name("q"))
        self.assertFalse(agent.is_valid_name("quit"))
        self.assertFalse(agent.is_valid_name("   "))

        # Request validation (>= 3 words)
        self.assertTrue(agent.is_valid_request("Ask Priya to move meeting"))
        self.assertTrue(agent.is_valid_request("one two three"))
        self.assertFalse(agent.is_valid_request("one two"))
        self.assertFalse(agent.is_valid_request(""))

        # Email validation
        self.assertTrue(agent.is_valid_email("priya@example.com"))
        self.assertTrue(agent.is_valid_email("john.doe@company.org"))
        self.assertFalse(agent.is_valid_email("priya@"))
        self.assertFalse(agent.is_valid_email("priyaexample.com"))
        self.assertFalse(agent.is_valid_email("@example.com"))

    def test_send_email_tool(self):
        res = tools.send_email.invoke({
            "to": "priya@example.com",
            "subject": "Regarding meeting",
            "body": "Dear Priya,\nCan we move tomorrow's meeting to 3pm?\nBest regards,\nVamsi"
        })
        self.assertEqual(res, "Email to priya@example.com saved in outbox.txt")
        self.assertTrue(os.path.exists("outbox.txt"))
        with open("outbox.txt", "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("To: priya@example.com", content)
        self.assertIn("Subject: Regarding meeting", content)
        self.assertIn("Dear Priya,", content)

    def test_graph_flow_with_revision_and_approval(self):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            AIMessage(content="Dear Priya,\nI was wondering if you could move tomorrow's meeting to 3pm.\nBest regards,\nVamsi"),
            AIMessage(content="Dear Priya,\nCould you please move tomorrow's meeting to 3pm?\nBest regards,\nVamsi")
        ]

        app = agent.build_agent(custom_llm=mock_llm)

        config = {"configurable": {"thread_id": "test-thread-1"}}
        initial_state = {
            "request": "Ask Priya to move tomorrow's meeting to 3pm",
            "to": "priya@example.com",
            "to_name": "Priya",
            "sender": "Vamsi",
            "draft": "",
            "feedback": "",
            "result": ""
        }

        # Step 1: Initial invocation -> pauses at human_review
        out = app.invoke(initial_state, config)
        self.assertIn("__interrupt__", out)
        interrupt_val = out["__interrupt__"][0].value
        self.assertEqual(interrupt_val["to"], "priya@example.com")
        self.assertIn("I was wondering if you could move", interrupt_val["draft"])

        # Step 2: Human asks for changes ("make it shorter") -> redrafts and pauses at human_review
        out2 = app.invoke(Command(resume="make it shorter"), config)
        self.assertIn("__interrupt__", out2)
        interrupt_val2 = out2["__interrupt__"][0].value
        self.assertIn("Could you please move", interrupt_val2["draft"])

        # Step 3: Human approves -> sends email and ends
        out3 = app.invoke(Command(resume="approve"), config)
        self.assertNotIn("__interrupt__", out3)
        self.assertEqual(out3["result"], "Email to priya@example.com saved in outbox.txt")
        self.assertTrue(os.path.exists("outbox.txt"))
        with open("outbox.txt", "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("Could you please move", content)

    def test_run_once_with_mock_inputs(self):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = [
            AIMessage(content="Dear Priya,\nFirst draft.\nBest regards,\nVamsi"),
            AIMessage(content="Dear Priya,\nSecond draft.\nBest regards,\nVamsi")
        ]

        app = agent.build_agent(custom_llm=mock_llm)

        initial_state = {
            "request": "Ask Priya to move meeting",
            "to": "priya@example.com",
            "to_name": "Priya",
            "sender": "Vamsi",
            "draft": "",
            "feedback": "",
            "result": ""
        }

        # Input sequence: empty input (rejected), change request, empty input (rejected), approve
        inputs = ["", "make it shorter", "", "approve"]
        with patch("builtins.input", side_effect=inputs):
            out = agent.run_once(app, initial_state)

        self.assertNotIn("__interrupt__", out)
        self.assertEqual(out["result"], "Email to priya@example.com saved in outbox.txt")


if __name__ == "__main__":
    unittest.main()
