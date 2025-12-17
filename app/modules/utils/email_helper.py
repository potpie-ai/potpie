import logging
import os

import resend

import re

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

        email = resend.Emails.send(params)
        return email


def is_valid_email(email: str) -> bool:
    """Simple regex-based email validation."""
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return re.match(pattern, email) is not None


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
    "icloud.com",
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
    "throwaway.email",
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
    "spamgourmet.com",
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
