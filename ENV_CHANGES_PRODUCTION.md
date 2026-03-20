# Environment Variables Checklist for Production

## Summary
This PR changes the authentication flow to use **Firebase ID tokens** instead of Google OAuth tokens for SSO. The code will automatically fall back to Google OAuth verification if Firebase is not available, but **Firebase should be properly configured for the best experience**.

---

## Backend (potpie) Environment Variables

### ✅ Required (Already in place)
- **Firebase Service Account**: 
  - `firebase_service_account.json` OR `firebase_service_account.txt` (base64 encoded)
  - Location: Root directory of backend
  - **Action**: Ensure this file exists in production

### ⚠️ Optional but Recommended (for fallback)
- `GOOGLE_SSO_CLIENT_ID` - Only needed if Firebase Admin SDK is not initialized
- `GOOGLE_SSO_CLIENT_SECRET` - Only needed if Firebase Admin SDK is not initialized  
- `GOOGLE_SSO_HOSTED_DOMAIN` - Optional, for domain-restricted SSO

### ℹ️ Other (unchanged)
- `SLACK_WEBHOOK_URL` - For new user notifications (optional)

---

## Frontend (potpie-ui) Environment Variables

### ✅ Required (Already in place)
These should already be configured, but verify they're set in production:

- `NEXT_PUBLIC_FIREBASE_API_KEY`
- `NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN`
- `NEXT_PUBLIC_FIREBASE_PROJECT_ID`
- `NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET` (optional)
- `NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID` (optional)
- `NEXT_PUBLIC_FIREBASE_APP_ID`
- `NEXT_PUBLIC_FIREBASE_MEASUREMENT_ID` (optional, for analytics)

**Note**: The frontend now sends Firebase ID tokens to the backend, so these must be correctly configured.

---

## Key Changes in This PR

### What Changed:
1. **Frontend**: Now sends Firebase ID tokens (from `user.getIdToken()`) instead of Google OAuth tokens
2. **Backend**: Verifies Firebase ID tokens first using Firebase Admin SDK, falls back to Google OAuth if needed

### What This Means:
- **Primary flow**: Uses Firebase Auth (more consistent, better integration)
- **Fallback flow**: Still works with Google OAuth if Firebase is unavailable
- **No breaking changes**: Existing users will continue to work

---

## Pre-Production Checklist

### Backend:
- [ ] Verify `firebase_service_account.json` or `firebase_service_account.txt` exists in production
- [ ] Ensure Firebase Admin SDK can initialize (check logs on startup)
- [ ] Optional: Set `GOOGLE_SSO_CLIENT_ID` and `GOOGLE_SSO_CLIENT_SECRET` as fallback
- [ ] Test that Firebase ID token verification works

### Frontend:
- [ ] Verify all `NEXT_PUBLIC_FIREBASE_*` environment variables are set
- [ ] Ensure Firebase project ID matches between frontend and backend
- [ ] Test that `user.getIdToken()` returns valid tokens
- [ ] Verify Firebase Auth domain is correctly configured

### Testing:
- [ ] Test Google SSO sign-in flow
- [ ] Test GitHub linking after SSO sign-in
- [ ] Test sign-out and sign-in again (GitHub should still be linked)
- [ ] Test email/password sign-up flow
- [ ] Verify Firebase UIDs are consistent (28 characters, alphanumeric)

---

## Migration Notes

### No Database Migration Required
The database schema is unchanged. The only change is:
- User UIDs are now consistently Firebase UIDs (28 chars) instead of Google OAuth sub (21 digits)
- This happens automatically for new users
- Existing users will continue to work with their current UIDs

### Rollback Plan
If issues occur, you can rollback by:
1. Reverting the frontend to send Google OAuth tokens (change `user.getIdToken()` back to `credential.idToken`)
2. The backend already supports both token types, so no backend rollback needed

---

## Questions?

If Firebase Admin SDK is not initialized in production:
- The code will automatically fall back to Google OAuth verification
- Set `GOOGLE_SSO_CLIENT_ID` and `GOOGLE_SSO_CLIENT_SECRET` as backup
- Users will still be able to sign in, but UIDs may be inconsistent

**Recommendation**: Ensure Firebase is properly configured for the best experience and consistent UIDs.

