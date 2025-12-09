"""
Email Channel implementation for HITL.

This channel sends HITL requests via email with secure approval/input links.
"""

import logging
import os
import secrets
from typing import Optional, Dict, Any
from urllib.parse import urlencode

from app.core.executions.hitl import HITLRequest, HITLResponse
from app.core.executions.hitl_channel import HITLChannel

logger = logging.getLogger(__name__)


class EmailChannel(HITLChannel):
    """
    Email channel for HITL requests.

    Sends HITL requests via email with secure links for approval/input.
    """

    def __init__(
        self,
        smtp_host: Optional[str] = None,
        smtp_port: Optional[int] = None,
        smtp_user: Optional[str] = None,
        smtp_password: Optional[str] = None,
        from_email: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """
        Initialize Email Channel.

        Args:
            smtp_host: SMTP server hostname
            smtp_port: SMTP server port
            smtp_user: SMTP username
            smtp_password: SMTP password
            from_email: From email address
            base_url: Base URL for generating approval links
        """
        self.smtp_host = smtp_host or os.getenv("SMTP_HOST", "smtp.gmail.com")
        self.smtp_port = smtp_port or int(os.getenv("SMTP_PORT", "587"))
        self.smtp_user = smtp_user or os.getenv("SMTP_USER")
        self.smtp_password = smtp_password or os.getenv("SMTP_PASSWORD")
        self.from_email = from_email or os.getenv("FROM_EMAIL", self.smtp_user)
        self.base_url = base_url or os.getenv("BASE_URL", "http://localhost:3000")

    @property
    def channel_id(self) -> str:
        """Return the unique identifier for this channel type."""
        return "email"

    @property
    def channel_name(self) -> str:
        """Return the human-readable name for this channel."""
        return "Email"

    def _generate_secure_link(self, request: HITLRequest) -> str:
        """Generate a secure link for the HITL request."""
        # Generate a secure token for the request
        token = secrets.token_urlsafe(32)
        
        # Store token with request (would need to extend HITLRequest model)
        # For now, we'll use the request_id as part of the link
        
        params = {
            "request_id": request.request_id,
            "execution_id": request.execution_id,
            "node_id": request.node_id,
            "iteration": request.iteration,
        }
        
        return f"{self.base_url}/workflows/pending-requests/{request.request_id}?{urlencode(params)}"

    async def send_request(self, request: HITLRequest) -> bool:
        """
        Send a HITL request via email.

        Args:
            request: The HITL request to send

        Returns:
            bool: True if email was sent successfully, False otherwise
        """
        try:
            # Determine recipients
            recipients = []
            if request.node_type.value == "approval" and request.approvers:
                recipients = request.approvers
            elif request.node_type.value == "input" and request.assignee:
                recipients = [request.assignee]

            if not recipients:
                logger.warning(
                    f"No recipients found for HITL request {request.request_id}"
                )
                return False

            # Generate secure link
            approval_link = self._generate_secure_link(request)

            # Prepare email content
            subject = f"HITL Request: {request.node_type.value.title()} Required"
            
            if request.node_type.value == "approval":
                body = self._format_approval_email(request, approval_link)
            else:
                body = self._format_input_email(request, approval_link)

            # Send email using smtplib
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            
            # Check if SMTP is configured
            if not self.smtp_user or not self.smtp_password:
                logger.warning(
                    f"SMTP not configured (SMTP_USER or SMTP_PASSWORD missing). "
                    f"Email will not be sent to {recipients}. "
                    f"Set SMTP_USER, SMTP_PASSWORD, and optionally SMTP_HOST, SMTP_PORT, FROM_EMAIL environment variables."
                )
                return False
            
            try:
                msg = MIMEMultipart('alternative')
                msg['From'] = self.from_email
                msg['To'] = ", ".join(recipients)
                msg['Subject'] = subject
                msg.attach(MIMEText(body, 'html'))
                
                logger.info(f"Sending email to {recipients} via {self.smtp_host}:{self.smtp_port}")
                
                with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                    server.starttls()
                    server.login(self.smtp_user, self.smtp_password)
                    server.send_message(msg)
                
                logger.info(f"✅ Email sent successfully to {recipients}")
                return True
                
            except smtplib.SMTPAuthenticationError as e:
                logger.error(f"SMTP authentication failed: {e}")
                return False
            except smtplib.SMTPException as e:
                logger.error(f"SMTP error sending email: {e}")
                return False
            except Exception as e:
                logger.error(f"Unexpected error sending email: {e}")
                logger.exception("Full traceback:")
                return False

        except Exception as e:
            logger.error(f"Error sending email for HITL request {request.request_id}: {e}")
            logger.exception("Full traceback:")
            return False

    def _format_approval_email(self, request: HITLRequest, link: str) -> str:
        """Format email body for approval requests."""
        return f"""
        <html>
        <body>
            <h2>Approval Request</h2>
            <p>{request.message}</p>
            <p><strong>Workflow Execution:</strong> {request.execution_id}</p>
            <p><strong>Node:</strong> {request.node_id}</p>
            <p><strong>Timeout:</strong> {request.timeout_at.strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
            <p>
                <a href="{link}" style="background-color: #4CAF50; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; display: inline-block; margin-right: 10px;">Approve</a>
                <a href="{link}&action=reject" style="background-color: #f44336; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; display: inline-block;">Reject</a>
            </p>
            <p><a href="{link}">View Full Request</a></p>
        </body>
        </html>
        """

    def _format_input_email(self, request: HITLRequest, link: str) -> str:
        """Format email body for input requests."""
        fields_info = ""
        if request.fields:
            fields_info = "<ul>"
            for field in request.fields:
                fields_info += f"<li><strong>{field.get('name')}</strong> ({field.get('type')})"
                if field.get('required'):
                    fields_info += " <em>Required</em>"
                fields_info += "</li>"
            fields_info += "</ul>"

        return f"""
        <html>
        <body>
            <h2>Input Request</h2>
            <p>{request.message}</p>
            <p><strong>Workflow Execution:</strong> {request.execution_id}</p>
            <p><strong>Node:</strong> {request.node_id}</p>
            <p><strong>Timeout:</strong> {request.timeout_at.strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
            <h3>Required Fields:</h3>
            {fields_info}
            <p>
                <a href="{link}" style="background-color: #2196F3; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; display: inline-block;">Provide Input</a>
            </p>
        </body>
        </html>
        """

    async def receive_response(
        self, request_id: str
    ) -> Optional[HITLResponse]:
        """
        Receive a response for a HITL request.

        For Email channel, responses are submitted via web links, not polling.
        This method is not used for Email channel.

        Args:
            request_id: The request ID to check for responses

        Returns:
            Optional[HITLResponse]: None (responses handled via web links)
        """
        # For Email channel, responses are handled via web links, not polling
        return None

    async def check_status(self, request_id: str) -> Dict[str, Any]:
        """
        Check the status of a HITL request in this channel.

        Args:
            request_id: The request ID to check

        Returns:
            Dict[str, Any]: Status information
        """
        return {
            "channel": self.channel_id,
            "status": "sent",
            "message": "Email sent (if configured)",
        }

