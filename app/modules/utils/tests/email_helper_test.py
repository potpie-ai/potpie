import pytest
from app.modules.utils.email_helper import is_valid_email


class TestEmailValidation:
    """Test cases for email validation functionality."""

    class TestValidEmails:
        """Test cases for valid email formats."""

        def test_simple_email(self):
            """Test simple email format."""
            assert is_valid_email("test@example.com") is True

        def test_email_with_plus(self):
            """Test email with plus sign."""
            assert is_valid_email("user+tag@example.com") is True

        def test_email_with_dots(self):
            """Test email with dots in username."""
            assert is_valid_email("first.last@example.com") is True

        def test_email_with_numbers(self):
            """Test email with numbers."""
            assert is_valid_email("user123@example.com") is True

        def test_email_with_hyphen_in_domain(self):
            """Test email with hyphen in domain."""
            assert is_valid_email("user@my-domain.com") is True

        def test_email_with_subdomain(self):
            """Test email with subdomain."""
            assert is_valid_email("user@mail.example.com") is True

        def test_email_with_long_tld(self):
            """Test email with long TLD."""
            assert is_valid_email("user@example.international") is True

        def test_email_with_underscore(self):
            """Test email with underscore in username."""
            assert is_valid_email("user_name@example.com") is True

        def test_email_with_percent(self):
            """Test email with percent sign."""
            assert is_valid_email("user%test@example.com") is True

    class TestInvalidEmails:
        """Test cases for invalid email formats."""

        def test_empty_string(self):
            """Test empty string."""
            assert is_valid_email("") is False

        def test_whitespace_only(self):
            """Test whitespace-only string."""
            assert is_valid_email("   ") is False

        def test_missing_at_symbol(self):
            """Test email without @ symbol."""
            assert is_valid_email("userexample.com") is False

        def test_missing_domain(self):
            """Test email without domain."""
            assert is_valid_email("user@") is False

        def test_missing_username(self):
            """Test email without username."""
            assert is_valid_email("@example.com") is False

        def test_missing_tld(self):
            """Test email without TLD."""
            assert is_valid_email("user@example") is False

        def test_multiple_at_symbols(self):
            """Test email with multiple @ symbols."""
            assert is_valid_email("user@@example.com") is False

        def test_spaces_in_email(self):
            """Test email with spaces."""
            assert is_valid_email("user @example.com") is False
            assert is_valid_email("user@ example.com") is False

        def test_special_chars_in_domain(self):
            """Test email with invalid special characters in domain."""
            assert is_valid_email("user@exam ple.com") is False

        def test_double_dots_in_domain(self):
            """Test email with consecutive dots in domain."""
            assert is_valid_email("user@example..com") is False

        def test_starting_with_dot(self):
            """Test email starting with dot."""
            assert is_valid_email(".user@example.com") is False

        def test_ending_with_dot(self):
            """Test email ending with dot before @."""
            assert is_valid_email("user.@example.com") is False

        def test_invalid_tld_length(self):
            """Test email with single character TLD."""
            assert is_valid_email("user@example.c") is False

        def test_only_at_symbol(self):
            """Test only @ symbol."""
            assert is_valid_email("@") is False

        def test_domain_without_dot(self):
            """Test domain without dot separator."""
            assert is_valid_email("user@examplecom") is False

    class TestEdgeCases:
        """Test edge cases for email validation."""

        def test_very_long_email(self):
            """Test very long but valid email."""
            long_username = "a" * 64
            assert is_valid_email(f"{long_username}@example.com") is True

        def test_minimum_valid_email(self):
            """Test minimum valid email format."""
            assert is_valid_email("a@b.co") is True

        def test_email_with_all_allowed_special_chars(self):
            """Test email with all allowed special characters."""
            assert is_valid_email("user+tag.name_test%value@example.com") is True

        def test_none_value(self):
            """Test None value handling."""
            with pytest.raises(AttributeError):
                is_valid_email(None)

        def test_numeric_string(self):
            """Test numeric string."""
            assert is_valid_email("12345") is False

        def test_email_like_string_without_tld(self):
            """Test email-like string without proper TLD."""
            assert is_valid_email("user@localhost") is False