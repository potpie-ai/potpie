import os

import resend


class EmailHelper:
    def __init__(self):
        self.api_key = os.environ.get("RESEND_API_KEY")
        self.transaction_emails_enabled = (
            os.environ.get("TRANSACTION_EMAILS_ENABLED", "false").lower() == "true"
        )
        self.from_address = os.environ.get("EMAIL_FROM_ADDRESS", "hi@potpie.ai")
        resend.api_key = self.api_key

    async def send_email(self, to_address, repo_name, branch_name ):
        if not self.transaction_emails_enabled:
            return

        params = {
            "from": self.from_address,
            "to": to_address,
            "subject": f"Your repository {repo_name} is ready! 🥧",
            "html": f"""
                <p>Hi!</p>
                <p>Your repo <strong>{repo_name}</strong> at branch <strong>{branch_name}</strong> has been processed successfully.</p>
                <p>You can use any of Potpie's ready-to-use agents to chat with it at: 
                    <a href='https://app.potpie.ai/newchat'>https://app.potpie.ai/newchat</a>.
                </p>
                <p>Please refer this document to get started: <a href='https://potpieai.notion.site/potpie-s-beta-program-10cc13a23aa8801e8e2bd34d8f1488f5'>Potpie User Guide</a></p>
                <p>Reply to this email or reach out to the founders at <a href='mailto:dhiren@potpie.ai'>dhiren@potpie.ai</a> if you have any questions.</p>
                <p>Thanks, <br />The Potpie Team 🥧</p>
            """,
        }

        email = resend.Emails.send(params)
        return email
