import os

import resend


class EmailHelper:
    def __init__(self):
        self.api_key = os.environ.get("RESEND_API_KEY")
        self.transaction_emails_enabled = (
            os.environ.get("TRANSACTION_EMAILS_ENABLED", "false").lower() == "true"
        )
        self.from_address = os.environ.get("EMAIL_FROM_ADDRESS", "support@momentum.sh")
        resend.api_key = self.api_key

    async def send_email(self, to_address):
        if not self.transaction_emails_enabled:
            return

        params = {
            "from": self.from_address,
            "to": to_address,
            "subject": "Project Update Email",
            "html": "Your Project Has Been Parsed Successfully",
        }

        email = resend.Emails.send(params)
        return email
