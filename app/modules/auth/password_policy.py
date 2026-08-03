"""FW001 password policy — mirrors potpie-ui lib/auth/password-policy.ts.

Validates passwords before Firebase Admin create_user so weak passwords
never reach Firebase. Never log the password value.
"""

from __future__ import annotations

PASSWORD_POLICY_ERROR = (
    "Use 15 or more characters with uppercase, lowercase, a number, "
    "and a special character."
)

# Firebase Identity Platform non-alphanumeric set (same as frontend).
_FIREBASE_SPECIAL_CHARACTERS = frozenset(
    {
        "^",
        "$",
        "*",
        ".",
        "[",
        "]",
        "{",
        "}",
        "(",
        ")",
        "?",
        '"',
        "!",
        "@",
        "#",
        "%",
        "&",
        "/",
        "\\",
        ",",
        ">",
        "<",
        "'",
        ":",
        ";",
        "|",
        "_",
        "~",
    }
)

_MIN_LENGTH = 15


class PasswordPolicyError(ValueError):
    """Raised when a password fails the FW001 policy."""

    def __init__(self, message: str = PASSWORD_POLICY_ERROR):
        super().__init__(message)
        self.message = message


def is_password_valid(password: str) -> bool:
    """Return True when password meets all FW001 requirements."""
    if not isinstance(password, str):
        return False
    characters = list(password)
    if len(characters) < _MIN_LENGTH:
        return False
    if not any(c.isupper() for c in characters):
        return False
    if not any(c.islower() for c in characters):
        return False
    if not any(c.isdigit() for c in characters):
        return False
    if not any(c in _FIREBASE_SPECIAL_CHARACTERS for c in characters):
        return False
    return True


def validate_password(password: str) -> None:
    """Raise PasswordPolicyError if the password is weak.

    Does not include the password in the exception message.
    """
    if not is_password_valid(password):
        raise PasswordPolicyError(PASSWORD_POLICY_ERROR)
