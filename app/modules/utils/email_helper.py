import os

import resend


class EmailHelper:
    def __init__(self):
        self.api_key = os.environ.get("RESEND_API_KEY")
        self.transaction_emails_enabled = (
            os.environ.get("TRANSACTION_EMAILS_ENABLED", "false").lower() == "true"
        )
        self.from_address = os.environ.get("EMAIL_FROM_ADDRESS", "dhiren@updates.potpie.ai")
        resend.api_key = self.api_key

    async def send_email(self, to_address, repo_name, branch_name ):
        if not self.transaction_emails_enabled:
            return

        params = {
            "from": f"Dhiren Mathur <{self.from_address}>",
            "to": to_address,
            "subject": f"Your repository {repo_name} is ready! 🥧",
            "reply_to": "dhiren@potpie.ai",
            "html": f"""
                <p>Hi!</p>
                <p>Your repository <strong>{repo_name}</strong> at branch <strong>{branch_name}</strong> has been processed successfully.</p>
                <p>You can use any of Potpie's ready-to-use agents to chat with it at: 
                    <a href='https://app.potpie.ai/newchat'>https://app.potpie.ai/newchat</a>.
                </p>
                <p>Please refer this document to get started: <a href='https://potpieai.notion.site/potpie-s-beta-program-10cc13a23aa8801e8e2bd34d8f1488f5'>Potpie User Guide</a></p>
                <p>Feel free to reply to this email if you have any questions.</p>
                <p>Thanks, <br />Dhiren Mathur,<br /> Co-Founder, Potpie 🥧</p>
            """,
        }

        email = resend.Emails.send(params)
        return email
