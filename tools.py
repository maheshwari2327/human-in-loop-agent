from langchain_core.tools import tool


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """
    'Send' an email. For this demo it appends the email to outbox.txt
    (a real app would call SMTP / Gmail API here). This is the RISKY action
    that a human must approve first.
    """
    with open("outbox.txt", "a", encoding="utf-8") as f:
        f.write(f"To: {to}\n")
        f.write(f"Subject: {subject}\n")
        f.write(f"Body:\n{body.strip()}\n")
        f.write("-" * 40 + "\n\n")
    return f"Email to {to} saved in outbox.txt"
