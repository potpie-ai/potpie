import asyncio
import logging
import os
import re

import resend

# Try to import email-inspector library for robust email domain detection
try:
    from email_inspector import inspect as email_inspect

    EMAIL_INSPECTOR_AVAILABLE = True
except ImportError:
    EMAIL_INSPECTOR_AVAILABLE = False
    logging.warning(
        "email-inspector library not available. "
        "Falling back to built-in personal email domain list. "
        "Install with: pip install email-inspector"
    )

# Try to import tldextract for proper domain extraction (handles multi-part TLDs)
try:
    import tldextract

    TLDEXTRACT_AVAILABLE = True
except ImportError:
    TLDEXTRACT_AVAILABLE = False
    logging.warning(
        "tldextract library not available. "
        "Falling back to simple domain extraction. "
        "Install with: pip install tldextract"
    )


class EmailHelper:
    def __init__(self):
        self.api_key = os.environ.get("RESEND_API_KEY")
        self.transaction_emails_enabled = (
            os.environ.get("TRANSACTION_EMAILS_ENABLED", "false").lower() == "true"
        )
        self.from_address = os.environ.get(
            "EMAIL_FROM_ADDRESS", "dhiren@updates.potpie.ai"
        )
        resend.api_key = self.api_key

    async def send_email(self, to_address, repo_name, branch_name):
        if not self.transaction_emails_enabled:
            return

        if not to_address:
            return

        params = {
            "from": f"Dhiren Mathur <{self.from_address}>",
            "to": to_address,
            "subject": f"Your repository {repo_name} is ready! 🥧",
            "reply_to": "dhiren@potpie.ai",
            "html": f"""
<p>Hi!</p>

<p>Great news! Your repository <strong>{repo_name}</strong> (branch: <strong>{branch_name}</strong>) has been successfully processed.</p>

<p>Ready to get started? You can now chat with your repository using our AI agents at <a href='https://app.potpie.ai/newchat?repo={repo_name}&branch={branch_name}'>app.potpie.ai</a>.</p>

<p>Check out our <a href='https://docs.potpie.ai'>documentation</a> to make the most of Potpie's features.</p>

<p>Have questions? Just reply to this email - we're here to help!</p>

<p>Best,<br />
Dhiren Mathur<br />
Co-Founder, Potpie 🥧</p>

<p>P.S. Love Potpie? Give us a ⭐ on <a href='https://github.com/Potpie-AI/potpie/'>GitHub</a>! And don't hesitate to open an issue if you run into any problems.</p>
            """,
        }

        # Resend SDK is sync-only; offload to thread to avoid blocking the event loop
        email = await asyncio.to_thread(resend.Emails.send, params)
        return email

    async def send_parsing_failure_alert(
        self,
        repo_name: str,
        branch_name: str,
        error_message: str,
        auth_method: str,
        failure_type: str = "cloning_auth",
        user_id: str = None,
        project_id: str = None,
        stack_trace: str = None,
    ):
        """Send internal alert when parsing/cloning/worktree creation fails.

        Sends to internal team addresses only (not users).

        Args:
            repo_name: Repository name that failed
            branch_name: Branch name that failed
            error_message: Error message/details
            auth_method: Authentication method attempted (github_app/user_oauth/environment)
            failure_type: Type of failure (cloning_auth/worktree_creation/bare_repo_clone)
            user_id: Optional user ID
            project_id: Optional project ID
            stack_trace: Optional stack trace (will be truncated)
        """
        if not self.transaction_emails_enabled:
            return

        # Internal recipients from env (no PII in source)
        internal_recipients = _get_internal_recipients()
        if not internal_recipients:
            logging.debug(
                "EMAIL_INTERNAL_RECIPIENTS unset or invalid; skipping parsing failure alert"
            )
            return

        # Format auth method for display
        auth_method_display = {
            "github_app": "GitHub App Installation Token",
            "user_oauth": "User OAuth Token",
            "environment": "Environment Token (GH_TOKEN_LIST)",
        }.get(auth_method, auth_method)

        # Format failure type for display
        failure_type_display = {
            "cloning_auth": "Authentication/Cloning Failure",
            "worktree_creation": "Worktree Creation Failure",
            "bare_repo_clone": "Bare Repository Clone Failure",
        }.get(failure_type, failure_type)

        # Build error details
        details = f"""
        <p><strong>Failure Type:</strong> {failure_type_display}</p>
        <p><strong>Authentication Method:</strong> {auth_method_display}</p>
        <p><strong>Error:</strong> {error_message}</p>
        """
        if user_id:
            details += f"<p><strong>User ID:</strong> {user_id}</p>"
        if project_id:
            details += f"<p><strong>Project ID:</strong> {project_id}</p>"

        # Add stack trace if available (truncate to ~2000 chars)
        if stack_trace:
            truncated_trace = stack_trace[:2000]
            if len(stack_trace) > 2000:
                truncated_trace += "\n\n[Stack trace truncated...]"
            details += f"""
            <p><strong>Stack Trace:</strong></p>
            <pre style="background: #f4f4f4; padding: 10px; overflow-x: auto;">{truncated_trace}</pre>
            """

        params = {
            "from": f"Potpie Alerts <{self.from_address}>",
            "to": internal_recipients,
            "subject": f"🚨 [{failure_type_display}] Failed: {repo_name}",
            "html": f"""
<p><strong>ALERT:</strong> Repository processing has failed.</p>

<p><strong>Repository:</strong> {repo_name}<br />
<strong>Branch:</strong> {branch_name}</p>

{details}

<p>Please investigate the issue.</p>

<p>— Potpie System</p>
            """,
        }

        try:
            # Resend SDK is sync-only; offload to thread to avoid blocking
            email = await asyncio.to_thread(resend.Emails.send, params)
            return email
        except Exception as e:
            logging.error(f"Failed to send parsing failure alert: {e}")
            return None


def is_valid_email(email: str) -> bool:
    """Simple regex-based email validation."""
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return re.match(pattern, email) is not None


def _get_internal_recipients() -> list[str]:
    """Load internal alert recipients from EMAIL_INTERNAL_RECIPIENTS env var.

    Parses comma-separated list, trims whitespace, lowercases and validates
    each entry. Returns empty list if unset or if no valid emails found.
    """
    raw = os.environ.get("EMAIL_INTERNAL_RECIPIENTS", "").strip()
    if not raw:
        return []
    entries = [e.strip() for e in raw.split(",") if e.strip()]
    return [e.lower() for e in entries if is_valid_email(e)]


# Comprehensive list of personal/free email domains
# This list can be extended or replaced with a library/API in the future
PERSONAL_EMAIL_DOMAINS = {
    # Major free email providers
    "gmail.com",
    "yahoo.com",
    "hotmail.com",
    "outlook.com",
    "icloud.com",
    "protonmail.com",
    "mail.com",
    "aol.com",
    "zoho.com",
    "yandex.com",
    "gmx.com",
    "live.com",
    "msn.com",
    "inbox.com",
    "fastmail.com",
    "tutanota.com",
    "mail.ru",
    "qq.com",
    "163.com",
    "126.com",
    "sina.com",
    "rediffmail.com",
    "mailinator.com",
    "guerrillamail.com",
    "10minutemail.com",
    "tempmail.com",
    "throwaway.email",
    # Additional common providers
    "proton.me",
    "pm.me",
    "me.com",
    "mac.com",
    "yahoo.co.uk",
    "yahoo.fr",
    "yahoo.de",
    "yahoo.it",
    "yahoo.es",
    "yahoo.com.au",
    "yahoo.com.br",
    "yahoo.co.jp",
    "hotmail.co.uk",
    "hotmail.fr",
    "hotmail.de",
    "hotmail.it",
    "hotmail.es",
    "outlook.co.uk",
    "outlook.fr",
    "outlook.de",
    "outlook.it",
    "outlook.es",
    "gmail.co.uk",
    "googlemail.com",
    # Disposable email services (common ones)
    "tempmail.org",
    "getnada.com",
    "maildrop.cc",
    "mohmal.com",
    "trashmail.com",
    "sharklasers.com",
    "guerrillamailblock.com",
    "pokemail.net",
    "spamgourmet.com",
    "temp-mail.org",
    "emailondeck.com",
    "fakeinbox.com",
    "mintemail.com",
    "meltmail.com",
    "melt.li",
    "33mail.com",
    "spambox.us",
    "spamfree24.org",
    "spamfree24.de",
    "spamfree24.eu",
    "spamfree24.net",
    "spamfree24.com",
    "spamhole.com",
    "spam.la",
    "spamobox.com",
    "spamspot.com",
    "tempail.com",
    "tempalias.com",
    "tempe-mail.com",
    "tempemail.biz",
    "tempemail.com",
    "tempinbox.co.uk",
    "tempinbox.com",
    "tempmail2.com",
    "tempmailer.com",
    "tempthe.net",
    "thankyou2010.com",
    "thisisnotmyrealemail.com",
    "throwam.com",
    "tilien.com",
    "tmail.ws",
    "tmailinator.com",
    "toiea.com",
    "tradermail.info",
    "trash-amil.com",
    "trash2009.com",
    "trashymail.com",
    "trialmail.de",
    "trillianpro.com",
    "turual.com",
    "twinmail.de",
    "tyldd.com",
    "uggsrock.com",
    "umail.net",
    "upliftnow.com",
    "uplipht.com",
    "uroid.com",
    "us.af",
    "venompen.com",
    "veryrealemail.com",
    "viditag.com",
    "viewcastmedia.com",
    "viewcastmedia.net",
    "viewcastmedia.org",
    "webemail.me",
    "webm4il.info",
    "wh4f.org",
    "whyspam.me",
    "willselfdestruct.com",
    "winemaven.info",
    "wronghead.com",
    "wuzup.net",
    "wuzupmail.net",
    "xagloo.com",
    "xemaps.com",
    "xents.com",
    "xmaily.com",
    "xoxy.net",
    "yapped.net",
    "yeah.net",
    "yep.it",
    "yogamaven.com",
    "yopmail.com",
    "yopmail.fr",
    "yopmail.net",
    "youmailr.com",
    "ypmail.webnetic.net",
    "zippymail.info",
    "zoemail.org",
    "zomg.info",
}


def is_personal_email_domain(email: str) -> bool:
    """
    Check if an email address belongs to a personal/free email provider.

    This function uses the email-inspector library (if available) which maintains
    a database of over 16,000 free email providers. Falls back to a built-in
    comprehensive list if the library is not available.

    Args:
        email: The email address to check (e.g., "user@example.com")

    Returns:
        True if the email domain is a personal/free email provider, False otherwise.
        Returns False if the email format is invalid or domain cannot be extracted.

    Examples:
        >>> is_personal_email_domain("user@gmail.com")
        True
        >>> is_personal_email_domain("user@company.com")
        False
        >>> is_personal_email_domain("invalid-email")
        False
    """
    if not email or "@" not in email:
        return False

    # Try using email-inspector library first (more robust, 16,000+ domains)
    if EMAIL_INSPECTOR_AVAILABLE:
        try:
            result = email_inspect(email)
            # email-inspector returns a dict with 'free' key indicating if it's a free email
            if isinstance(result, dict) and "free" in result:
                return result["free"]
            # Fallback if result format is unexpected
        except Exception as e:
            logging.warning(
                f"Error using email-inspector for {email}: {e}. Falling back to built-in list."
            )

    # Fallback to built-in comprehensive list
    try:
        domain = email.split("@")[1].lower().strip()
        return domain in PERSONAL_EMAIL_DOMAINS
    except (IndexError, AttributeError):
        return False


def extract_organization_from_email(email: str) -> str | None:
    """
    Extract organization domain from an email address.

    If the email belongs to a personal/free email provider, returns None.
    Otherwise, returns the domain as the organization.

    Args:
        email: The email address to extract organization from

    Returns:
        The organization domain (e.g., "company.com") or None if personal email

    Examples:
        >>> extract_organization_from_email("user@company.com")
        'company.com'
        >>> extract_organization_from_email("user@gmail.com")
        None
    """
    if not email or "@" not in email:
        return None

    try:
        domain = email.split("@")[1].lower().strip()
        if is_personal_email_domain(email):
            return None
        return domain
    except (IndexError, AttributeError):
        return None


def extract_registrable_domain(email: str) -> str:
    """
    Extracts the registrable domain from an email address using public suffix list.
    Handles multi-part TLDs correctly (e.g., .co.uk, .com.au).

    Validates and normalizes the email (strip and lower), parses the domain with
    tldextract to get the registrable domain, and returns that lowercase string
    or empty string on invalid input.

    Args:
        email: User's email address

    Returns:
        The registrable domain in lowercase, or empty string if invalid

    Example:
        extract_registrable_domain('user@GmAiL.CoM') -> 'gmail.com'
        extract_registrable_domain('user@eng.company.com') -> 'company.com'
        extract_registrable_domain('user@gmail.co.uk') -> 'gmail.co.uk' (not 'co.uk')
    """
    # Validate and normalize email input
    if not email or not isinstance(email, str):
        return ""

    # Strip whitespace and convert to lowercase
    email = email.strip().lower()
    if not email:
        return ""

    # Split email to extract domain part
    parts = email.split("@")
    if len(parts) != 2:
        return ""

    domain = parts[1]
    if not domain:
        return ""

    # Use tldextract library to get the registrable domain
    # This properly handles multi-part TLDs like .co.uk, .com.au, etc.
    if TLDEXTRACT_AVAILABLE:
        try:
            extracted = tldextract.extract(domain)
            # tldextract returns: ExtractResult(subdomain='www', domain='example', suffix='co.uk')
            # registered_domain combines domain + suffix (e.g., 'example.co.uk')
            registrable_domain = extracted.registered_domain
            # If registered_domain is empty, fall back to the domain itself
            return registrable_domain if registrable_domain else domain
        except Exception:
            # If extraction fails for any reason, fall back
            pass

    # Fallback to simple logic if tldextract is not available
    # This is less accurate but won't break if the library isn't installed
    # Note: This will fail for multi-part TLDs like .co.uk, .com.au
    domain_parts = domain.split(".")
    if len(domain_parts) >= 2:
        # Take the last two parts (e.g., 'company.com')
        return ".".join(domain_parts[-2:])

    return domain
