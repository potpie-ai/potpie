"""
Unit tests for UnifiedAuthService
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from app.modules.auth.unified_auth_service import UnifiedAuthService
from app.modules.auth.auth_schema import AuthProviderCreate
from app.modules.auth.auth_provider_model import UserAuthProvider, PendingProviderLink


class TestUnifiedAuthService:
    """Test UnifiedAuthService methods"""

    def test_init(self, db_session):
        """Test service initialization"""
        service = UnifiedAuthService(db_session)
        assert service.db == db_session
        assert service.user_service is not None
        assert "google" in service.sso_providers
        assert "azure" in service.sso_providers

    def test_get_user_providers(self, db_session, test_user_with_github):
        """Test getting user's providers"""
        service = UnifiedAuthService(db_session)
        providers = service.get_user_providers(test_user_with_github.uid)
        
        assert len(providers) == 1
        assert providers[0].provider_type == "firebase_github"
        assert providers[0].is_primary == True

    def test_get_user_providers_multiple(self, db_session, test_user_with_multiple_providers):
        """Test getting multiple providers ordered by primary first"""
        service = UnifiedAuthService(db_session)
        providers = service.get_user_providers(test_user_with_multiple_providers.uid)
        
        assert len(providers) == 2
        # Primary provider should be first
        assert providers[0].is_primary == True
        assert providers[0].provider_type == "firebase_github"

    def test_get_provider(self, db_session, test_user_with_github):
        """Test getting specific provider"""
        service = UnifiedAuthService(db_session)
        provider = service.get_provider(
            test_user_with_github.uid, 
            "firebase_github"
        )
        
        assert provider is not None
        assert provider.provider_type == "firebase_github"

    def test_get_provider_not_found(self, db_session, test_user):
        """Test getting non-existent provider"""
        service = UnifiedAuthService(db_session)
        provider = service.get_provider(test_user.uid, "sso_google")
        
        assert provider is None

    def test_add_provider_first(self, db_session, test_user):
        """Test adding first provider (should be primary)"""
        service = UnifiedAuthService(db_session)
        
        provider_create = AuthProviderCreate(
            provider_type="firebase_github",
            provider_uid="github-123",
            provider_data={"login": "testuser"},
            is_primary=False,  # Should be overridden
        )
        
        provider = service.add_provider(
            user_id=test_user.uid,
            provider_create=provider_create,
        )
        
        assert provider.provider_type == "firebase_github"
        assert provider.is_primary == True  # First provider is always primary

    def test_add_provider_second(self, db_session, test_user_with_github):
        """Test adding second provider (should not be primary by default)"""
        service = UnifiedAuthService(db_session)
        
        provider_create = AuthProviderCreate(
            provider_type="sso_google",
            provider_uid="google-456",
            provider_data={"email": "test@example.com"},
        )
        
        provider = service.add_provider(
            user_id=test_user_with_github.uid,
            provider_create=provider_create,
        )
        
        assert provider.provider_type == "sso_google"
        assert provider.is_primary == False

    def test_add_provider_duplicate(self, db_session, test_user_with_github):
        """Test adding duplicate provider (should return existing)"""
        service = UnifiedAuthService(db_session)
        
        provider_create = AuthProviderCreate(
            provider_type="firebase_github",
            provider_uid="github-999",  # Different UID
            provider_data={},
        )
        
        provider = service.add_provider(
            user_id=test_user_with_github.uid,
            provider_create=provider_create,
        )
        
        # Should return existing provider
        assert provider.provider_uid == "github-123"  # Original UID

    def test_set_primary_provider(self, db_session, test_user_with_multiple_providers):
        """Test setting primary provider"""
        service = UnifiedAuthService(db_session)
        
        # Set Google as primary
        result = service.set_primary_provider(
            test_user_with_multiple_providers.uid, 
            "sso_google"
        )
        
        assert result == True
        
        # Verify
        providers = service.get_user_providers(test_user_with_multiple_providers.uid)
        google_provider = next(p for p in providers if p.provider_type == "sso_google")
        github_provider = next(p for p in providers if p.provider_type == "firebase_github")
        
        assert google_provider.is_primary == True
        assert github_provider.is_primary == False

    def test_set_primary_provider_not_found(self, db_session, test_user):
        """Test setting non-existent provider as primary"""
        service = UnifiedAuthService(db_session)
        result = service.set_primary_provider(test_user.uid, "sso_azure")
        
        assert result == False

    def test_unlink_provider(self, db_session, test_user_with_multiple_providers):
        """Test unlinking non-primary provider"""
        service = UnifiedAuthService(db_session)
        
        # Unlink Google (non-primary)
        result = service.unlink_provider(
            test_user_with_multiple_providers.uid,
            "sso_google"
        )
        
        assert result == True
        
        # Verify
        providers = service.get_user_providers(test_user_with_multiple_providers.uid)
        assert len(providers) == 1
        assert providers[0].provider_type == "firebase_github"

    def test_unlink_last_provider_raises_error(self, db_session, test_user_with_github):
        """Test unlinking last provider raises error"""
        service = UnifiedAuthService(db_session)
        
        with pytest.raises(ValueError, match="Cannot unlink the only authentication provider"):
            service.unlink_provider(
                test_user_with_github.uid,
                "firebase_github"
            )

    def test_unlink_primary_promotes_another(self, db_session, test_user_with_multiple_providers):
        """Test unlinking primary provider promotes another"""
        service = UnifiedAuthService(db_session)
        
        # Unlink GitHub (primary)
        result = service.unlink_provider(
            test_user_with_multiple_providers.uid,
            "firebase_github"
        )
        
        assert result == True
        
        # Verify Google is now primary
        providers = service.get_user_providers(test_user_with_multiple_providers.uid)
        assert len(providers) == 1
        assert providers[0].provider_type == "sso_google"
        assert providers[0].is_primary == True

    def test_update_last_used(self, db_session, test_user_with_github):
        """Test updating provider last_used_at"""
        service = UnifiedAuthService(db_session)
        
        # Get original time
        provider = service.get_provider(test_user_with_github.uid, "firebase_github")
        original_time = provider.last_used_at
        
        # Update
        service.update_last_used(test_user_with_github.uid, "firebase_github")
        
        # Verify
        provider = service.get_provider(test_user_with_github.uid, "firebase_github")
        assert provider.last_used_at > original_time

    @pytest.mark.asyncio
    async def test_authenticate_or_create_new_user(self, db_session):
        """Test authentication flow for new user"""
        service = UnifiedAuthService(db_session)
        
        user, response = await service.authenticate_or_create(
            email="newuser@example.com",
            provider_type="sso_google",
            provider_uid="google-new-123",
            provider_data={"name": "New User"},
            display_name="New User",
            email_verified=True,
        )
        
        assert user is not None
        assert user.email == "newuser@example.com"
        assert response.status == "new_user"
        assert response.user_id == user.uid

    @pytest.mark.asyncio
    async def test_authenticate_or_create_existing_with_provider(
        self, db_session, test_user_with_github
    ):
        """Test authentication for existing user with provider"""
        service = UnifiedAuthService(db_session)
        
        user, response = await service.authenticate_or_create(
            email="test@example.com",
            provider_type="firebase_github",
            provider_uid="github-123",
            display_name="Test User",
            email_verified=True,
        )
        
        assert user.uid == test_user_with_github.uid
        assert response.status == "success"
        assert response.user_id == test_user_with_github.uid

    @pytest.mark.asyncio
    async def test_authenticate_or_create_needs_linking(
        self, db_session, test_user_with_github
    ):
        """Test authentication creates pending link for new provider"""
        service = UnifiedAuthService(db_session)
        
        user, response = await service.authenticate_or_create(
            email="test@example.com",
            provider_type="sso_google",
            provider_uid="google-new-456",
            display_name="Test User",
            email_verified=True,
        )
        
        assert user.uid == test_user_with_github.uid
        assert response.status == "needs_linking"
        assert response.linking_token is not None
        assert "firebase_github" in response.existing_providers

    def test_confirm_provider_link(self, db_session, pending_link):
        """Test confirming a pending provider link"""
        service = UnifiedAuthService(db_session)
        
        provider = service.confirm_provider_link(pending_link.token)
        
        assert provider is not None
        assert provider.provider_type == "sso_google"
        assert provider.user_id == pending_link.user_id
        
        # Verify pending link is deleted
        db_session.expire_all()
        remaining = db_session.query(PendingProviderLink).filter_by(
            token=pending_link.token
        ).first()
        assert remaining is None

    def test_confirm_provider_link_invalid_token(self, db_session):
        """Test confirming with invalid token"""
        service = UnifiedAuthService(db_session)
        provider = service.confirm_provider_link("invalid-token")
        
        assert provider is None

    def test_confirm_provider_link_expired(self, db_session, test_user):
        """Test confirming expired link"""
        service = UnifiedAuthService(db_session)
        
        # Create expired link
        expired_link = PendingProviderLink(
            user_id=test_user.uid,
            provider_type="sso_google",
            provider_uid="google-789",
            provider_data={},
            token="expired-token",
            expires_at=datetime.utcnow() - timedelta(minutes=1),  # Already expired
        )
        db_session.add(expired_link)
        db_session.commit()
        
        provider = service.confirm_provider_link("expired-token")
        
        assert provider is None

    def test_cancel_pending_link(self, db_session, pending_link):
        """Test cancelling a pending link"""
        service = UnifiedAuthService(db_session)
        
        result = service.cancel_pending_link(pending_link.token)
        assert result == True
        
        # Verify deleted
        db_session.expire_all()
        remaining = db_session.query(PendingProviderLink).filter_by(
            token=pending_link.token
        ).first()
        assert remaining is None

    def test_get_org_sso_config(self, db_session, org_sso_config):
        """Test getting organization SSO config"""
        service = UnifiedAuthService(db_session)
        config = service.get_org_sso_config("company.com")
        
        assert config is not None
        assert config.domain == "company.com"
        assert config.sso_provider == "google"

    def test_should_enforce_sso(self, db_session, org_sso_config):
        """Test checking if SSO should be enforced"""
        service = UnifiedAuthService(db_session)
        
        enforce, provider = service.should_enforce_sso("user@company.com")
        
        assert enforce == True
        assert provider == "google"

    def test_should_enforce_sso_no_config(self, db_session):
        """Test SSO enforcement for domain without config"""
        service = UnifiedAuthService(db_session)
        
        enforce, provider = service.should_enforce_sso("user@example.com")
        
        assert enforce == False
        assert provider is None

    @pytest.mark.asyncio
    async def test_verify_sso_token(self, db_session):
        """Test verifying SSO token"""
        service = UnifiedAuthService(db_session)
        
        # Mock the provider's verify_token method
        mock_user_info = Mock()
        mock_user_info.email = "test@example.com"
        mock_user_info.email_verified = True
        
        with patch.object(
            service.sso_providers["google"],
            "verify_token",
            return_value=mock_user_info
        ):
            user_info = await service.verify_sso_token("google", "mock-token")
            
            assert user_info is not None
            assert user_info.email == "test@example.com"

    @pytest.mark.asyncio
    async def test_verify_sso_token_invalid(self, db_session):
        """Test verifying invalid SSO token"""
        service = UnifiedAuthService(db_session)
        
        with patch.object(
            service.sso_providers["google"],
            "verify_token",
            side_effect=ValueError("Invalid token")
        ):
            user_info = await service.verify_sso_token("google", "invalid-token")
            
            assert user_info is None

    def test_audit_log_created(self, db_session, test_user):
        """Test that audit logs are created"""
        service = UnifiedAuthService(db_session)
        
        service._log_auth_event(
            user_id=test_user.uid,
            event_type="login",
            provider_type="firebase_github",
            status="success",
            ip_address="127.0.0.1",
        )
        
        # Verify log created
        from app.modules.auth.auth_provider_model import AuthAuditLog
        log = db_session.query(AuthAuditLog).filter_by(
            user_id=test_user.uid
        ).first()
        
        assert log is not None
        assert log.event_type == "login"
        assert log.status == "success"

