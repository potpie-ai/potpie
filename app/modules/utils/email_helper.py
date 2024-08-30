import os
import resend

class EmailHelper:
    def __init__(self):
        self.api_key = os.environ.get("RESEND_API_KEY")
        resend.api_key = self.api_key 

    async def send_email(self, to_address):
        params = {
            "from": "raj@momentum.sh",
            "to": to_address,
            "subject": "Project Update Email",
            "html": "Your Project Has Been Parsed Successfully"
        }

        email = resend.Emails.send(params)
        return email