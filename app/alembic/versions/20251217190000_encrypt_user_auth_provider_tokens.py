"""encrypt_user_auth_provider_tokens

Revision ID: 20251217190000
Revises: 20251202164905_07bea433f543
Create Date: 2025-12-17 19:00:00.000000

Encrypts existing plaintext tokens in user_auth_providers table.
Handles NULL tokens gracefully and provides backward compatibility.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy import text

# revision identifiers, used by Alembic.
revision: str = '20251217190000'
down_revision: Union[str, None] = '20251202164905_07bea433f543'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """
    Encrypt existing plaintext tokens in user_auth_providers table.
    
    This migration:
    1. Identifies tokens that are likely plaintext (not encrypted)
    2. Encrypts them using the TokenEncryption utility
    3. Updates the database with encrypted versions
    4. Handles NULL tokens gracefully
    
    Note: This is a data migration that runs Python code.
    The service layer will handle decryption automatically going forward.
    """
    # Import encryption utility
    from app.modules.integrations.token_encryption import encrypt_token, decrypt_token
    
    # Get database connection
    conn = op.get_bind()
    
    # Fetch all providers with tokens
    result = conn.execute(text("""
        SELECT id, access_token, refresh_token 
        FROM user_auth_providers 
        WHERE access_token IS NOT NULL OR refresh_token IS NOT NULL
    """))
    
    providers = result.fetchall()
    encrypted_count = 0
    skipped_count = 0
    error_count = 0
    
    for provider_id, access_token, refresh_token in providers:
        try:
            updates = {}
            
            # Encrypt access_token if present
            if access_token:
                # Check if already encrypted by trying to decrypt
                # If decryption succeeds, token is already encrypted - skip
                # If decryption fails, token is likely plaintext - encrypt it
                try:
                    decrypt_token(access_token)
                    # Decryption succeeded - token is already encrypted
                    pass
                except Exception:
                    # Decryption failed - token is likely plaintext, encrypt it
                    encrypted_access = encrypt_token(access_token)
                    updates['access_token'] = encrypted_access
            
            # Encrypt refresh_token if present
            if refresh_token:
                try:
                    decrypt_token(refresh_token)
                    # Decryption succeeded - token is already encrypted
                    pass
                except Exception:
                    # Decryption failed - token is likely plaintext, encrypt it
                    encrypted_refresh = encrypt_token(refresh_token)
                    updates['refresh_token'] = encrypted_refresh
            
            # Update if we have changes
            if updates:
                set_clause = ", ".join([f"{k} = :{k}" for k in updates.keys()])
                conn.execute(
                    text(f"UPDATE user_auth_providers SET {set_clause} WHERE id = :id"),
                    {**updates, "id": str(provider_id)}
                )
                conn.commit()
                encrypted_count += 1
            else:
                skipped_count += 1
                
        except Exception as e:
            # Log error but continue with other providers
            error_count += 1
            # Use print for migration logging (Alembic captures this)
            print(f"Error encrypting tokens for provider {provider_id}: {str(e)}")
    
    print(
        f"Token encryption migration complete: "
        f"{encrypted_count} providers encrypted, "
        f"{skipped_count} skipped (already encrypted or NULL), "
        f"{error_count} errors"
    )


def downgrade() -> None:
    """
    Note: Decryption is handled automatically by the service layer.
    This migration does not need a downgrade as:
    1. The service layer handles both encrypted and plaintext tokens (backward compatible)
    2. Decrypting all tokens would expose them in plaintext, which is a security risk
    3. The encryption is transparent to the application layer
    """
    # No-op: Service layer handles decryption automatically
    pass
