# UI Authentication Integration Guide

## Overview

This guide explains how to integrate with the authentication system, including SSO login, GitHub linking, and handling credential conflicts.

## Table of Contents

1. [Authentication Flow](#authentication-flow)
2. [API Endpoints](#api-endpoints)
3. [GitHub Linking Flow](#github-linking-flow)
4. [Handling Credential Conflicts](#handling-credential-conflicts)
5. [Response Formats](#response-formats)
6. [Error Handling](#error-handling)
7. [Complete Integration Example](#complete-integration-example)

---

## Authentication Flow

### Step 1: User Signs In with SSO (Google)

```javascript
import { signInWithPopup, GoogleAuthProvider } from 'firebase/auth';

const provider = new GoogleAuthProvider();
const result = await signInWithPopup(auth, provider);
const idToken = await result.user.getIdToken();
```

### Step 2: Send Token to Backend

```javascript
const response = await fetch('/api/v1/sso/login', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    email: result.user.email,
    sso_provider: 'google',
    id_token: idToken,
  }),
});

const data = await response.json();
```

### Step 3: Check Response Status

The backend returns one of three statuses:

- `success` - User authenticated (may still need GitHub linking)
- `needs_linking` - User exists with different provider (rare)
- `new_user` - New account created

**Important**: Always check `needs_github_linking` flag!

---

## API Endpoints

### 1. SSO Login

**POST** `/api/v1/sso/login`

**Request:**
```json
{
  "email": "user@example.com",
  "sso_provider": "google",
  "id_token": "firebase_id_token_here",
  "provider_data": {}  // Optional
}
```

**Response:**
```json
{
  "status": "success" | "needs_linking" | "new_user",
  "user_id": "firebase_uid_here",
  "email": "user@example.com",
  "display_name": "John Doe",
  "message": "Login successful",
  "needs_github_linking": true | false,
  "github_token_valid": true | false | null
}
```

**Response Fields:**
- `status`: Authentication status
- `needs_github_linking`: `true` if user needs to link GitHub
- `github_token_valid`: 
  - `true` - GitHub is linked and token works
  - `false` - GitHub is linked but token expired/invalid
  - `null` - GitHub not linked

---

### 2. GitHub Linking (Signup Endpoint)

**POST** `/api/v1/signup`

**Request:**
```json
{
  "uid": "firebase_uid",
  "email": "user@example.com",
  "displayName": "John Doe",
  "emailVerified": true,
  "linkToUserId": "current_user_firebase_uid",  // Required for linking
  "githubFirebaseUid": "github_firebase_uid",
  "accessToken": "github_oauth_token",
  "providerUsername": "github_username"
}
```

**Response (Success):**
```json
{
  "uid": "firebase_uid",
  "exists": true,
  "needs_github_linking": false
}
```

**Response (Error - Already Linked):**
```json
{
  "error": "GitHub account is already linked to another account.",
  "details": "..."
}
```
Status Code: `409` (Conflict)

---

### 3. Resolve GitHub Credential Conflict

**POST** `/api/v1/resolve-github-conflict`

**When to use:** When Firebase throws `auth/credential-already-in-use` error

**Request:**
```json
{
  "current_user_uid": "your_current_firebase_uid",
  "conflicting_github_uid": "uid_of_user_with_github_linked",
  "github_access_token": "optional_token",
  "github_username": "optional_username"
}
```

**Response (Success):**
```json
{
  "success": true,
  "message": "Conflict resolved. You can now link your GitHub account.",
  "deleted_firebase_uid": "conflicting_uid"
}
```

**Response (Error):**
```json
{
  "error": "Error message here",
  "details": "Additional details"
}
```

---

## GitHub Linking Flow

### Complete Flow Diagram

```
1. User signs in with SSO
   ↓
2. Backend returns needs_github_linking: true
   ↓
3. Show GitHub linking UI
   ↓
4. User clicks "Link GitHub"
   ↓
5. Firebase GitHub OAuth popup
   ↓
6. User authorizes
   ↓
7. Try to link GitHub to Firebase user
   ↓
8a. Success → Call /api/v1/signup with linkToUserId
   ↓
9a. Backend links GitHub → Return success
   
8b. Error: credential-already-in-use
   ↓
9b. Extract conflicting UID from error
   ↓
10b. Call /api/v1/resolve-github-conflict
   ↓
11b. Backend deletes old user → Return success
   ↓
12b. Retry linking GitHub (step 7)
```

---

## Handling Credential Conflicts

### The Problem

When a user:
1. Originally signed up with GitHub → Firebase User A
2. Deleted local DB (but Firebase still has User A)
3. Signs up with Google SSO → Firebase User B
4. Tries to link GitHub to User B

Firebase throws: `auth/credential-already-in-use`

### Solution: Extract Conflicting UID

Firebase doesn't directly give you the conflicting user's UID. You need to:

```javascript
import { signInWithCredential, signOut } from 'firebase/auth';
import { GithubAuthProvider } from 'firebase/auth';

async function getConflictingUid(error, currentUser) {
  // Get the credential from the error
  const credential = GithubAuthProvider.credentialFromError(error);
  
  // Temporarily sign in with the conflicting credential
  const tempResult = await signInWithCredential(auth, credential);
  const conflictingUid = tempResult.user.uid;
  
  // Sign out immediately
  await signOut(auth);
  
  // Re-authenticate as current user
  // (You need to store current user's token before this)
  await reauthenticateCurrentUser(currentUser);
  
  return conflictingUid;
}
```

### Complete Conflict Resolution

```javascript
async function linkGitHubWithConflictResolution(currentUser) {
  const provider = new GithubAuthProvider();
  provider.addScope('repo'); // Request repo access
  
  try {
    // Try linking GitHub
    const result = await linkWithPopup(currentUser, provider);
    
    // Success! Send to backend
    await fetch('/api/v1/signup', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        uid: currentUser.uid,
        email: currentUser.email,
        displayName: currentUser.displayName,
        emailVerified: currentUser.emailVerified,
        linkToUserId: currentUser.uid,
        githubFirebaseUid: result.user.uid, // GitHub provider UID
        accessToken: result.user.accessToken, // GitHub OAuth token
        providerUsername: result.additionalUserInfo?.username,
      }),
    });
    
    return { success: true };
    
  } catch (error) {
    if (error.code === 'auth/credential-already-in-use') {
      // Step 1: Get conflicting UID
      const credential = GithubAuthProvider.credentialFromError(error);
      const tempAuth = await signInWithCredential(auth, credential);
      const conflictingUid = tempAuth.user.uid;
      await signOut(auth);
      
      // Step 2: Re-authenticate current user
      // (Store current user's token before calling getConflictingUid)
      await reauthenticateCurrentUser(currentUser);
      
      // Step 3: Call backend to resolve conflict
      const response = await fetch('/api/v1/resolve-github-conflict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          current_user_uid: currentUser.uid,
          conflicting_github_uid: conflictingUid,
        }),
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to resolve conflict');
      }
      
      // Step 4: Retry linking (should work now)
      return await linkGitHubWithConflictResolution(currentUser);
      
    } else {
      // Other errors
      throw error;
    }
  }
}
```

---

## Response Formats

### SSOLoginResponse

```typescript
interface SSOLoginResponse {
  status: 'success' | 'needs_linking' | 'new_user';
  user_id?: string;
  email: string;
  display_name?: string;
  access_token?: string;
  message: string;
  linking_token?: string;  // If status is 'needs_linking'
  existing_providers?: string[];  // If status is 'needs_linking'
  needs_github_linking?: boolean;
  github_token_valid?: boolean | null;
}
```

### Decision Tree Based on Response

```javascript
if (response.needs_github_linking) {
  if (response.github_token_valid === false) {
    // Token expired - show "Re-link GitHub" message
    showGitHubLinkingUI('Your GitHub token has expired. Please re-link your account.');
  } else {
    // Not linked - show "Connect GitHub" message
    showGitHubLinkingUI('Please connect your GitHub account to continue.');
  }
} else {
  // GitHub is working - proceed to app
  navigateToApp();
}
```

---

## Error Handling

### Common Errors

| Error Code | Meaning | Solution |
|------------|---------|----------|
| `auth/credential-already-in-use` | GitHub linked to different Firebase user | Use conflict resolution endpoint |
| `409` (Conflict) | GitHub already linked to different account | Show error message to user |
| `404` | User not found | Re-authenticate |
| `403` | Email mismatch | Show security error |

### Error Response Format

```json
{
  "error": "Error message",
  "details": "Additional context"
}
```

---

## Complete Integration Example

### React Component Example

```javascript
import { useState } from 'react';
import { signInWithPopup, GoogleAuthProvider, linkWithPopup, GithubAuthProvider } from 'firebase/auth';
import { auth } from './firebase';

function AuthFlow() {
  const [loading, setLoading] = useState(false);
  const [needsGitHub, setNeedsGitHub] = useState(false);
  const [githubTokenValid, setGithubTokenValid] = useState(null);
  
  async function handleSSOLogin() {
    setLoading(true);
    try {
      // Step 1: Firebase SSO
      const provider = new GoogleAuthProvider();
      const result = await signInWithPopup(auth, provider);
      const idToken = await result.user.getIdToken();
      
      // Step 2: Backend authentication
      const response = await fetch('/api/v1/sso/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          email: result.user.email,
          sso_provider: 'google',
          id_token: idToken,
        }),
      });
      
      const data = await response.json();
      
      // Step 3: Check GitHub status
      setNeedsGitHub(data.needs_github_linking);
      setGithubTokenValid(data.github_token_valid);
      
      if (data.needs_github_linking) {
        // Show GitHub linking UI
        return;
      }
      
      // Success - navigate to app
      navigateToApp();
      
    } catch (error) {
      console.error('Login failed:', error);
      showError(error.message);
    } finally {
      setLoading(false);
    }
  }
  
  async function handleGitHubLink(currentUser) {
    setLoading(true);
    try {
      await linkGitHubWithConflictResolution(currentUser);
      
      // Refresh user status
      const response = await fetch('/api/v1/sso/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          email: currentUser.email,
          sso_provider: 'google',
          id_token: await currentUser.getIdToken(),
        }),
      });
      
      const data = await response.json();
      setNeedsGitHub(data.needs_github_linking);
      setGithubTokenValid(data.github_token_valid);
      
      if (!data.needs_github_linking) {
        navigateToApp();
      }
      
    } catch (error) {
      console.error('GitHub linking failed:', error);
      showError(error.message);
    } finally {
      setLoading(false);
    }
  }
  
  return (
    <div>
      {needsGitHub ? (
        <GitHubLinkingUI 
          onLink={handleGitHubLink}
          tokenValid={githubTokenValid}
        />
      ) : (
        <button onClick={handleSSOLogin} disabled={loading}>
          Sign in with Google
        </button>
      )}
    </div>
  );
}

async function linkGitHubWithConflictResolution(currentUser) {
  const provider = new GithubAuthProvider();
  provider.addScope('repo');
  
  try {
    const result = await linkWithPopup(currentUser, provider);
    
    // Send to backend
    const response = await fetch('/api/v1/signup', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        uid: currentUser.uid,
        email: currentUser.email,
        displayName: currentUser.displayName,
        emailVerified: currentUser.emailVerified,
        linkToUserId: currentUser.uid,
        githubFirebaseUid: result.user.uid,
        accessToken: result.user.accessToken,
        providerUsername: result.additionalUserInfo?.username,
      }),
    });
    
    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.error || 'Failed to link GitHub');
    }
    
    return { success: true };
    
  } catch (error) {
    if (error.code === 'auth/credential-already-in-use') {
      // Handle conflict
      const credential = GithubAuthProvider.credentialFromError(error);
      
      // Store current user token before switching
      const currentToken = await currentUser.getIdToken();
      
      // Get conflicting UID
      const tempAuth = await signInWithCredential(auth, credential);
      const conflictingUid = tempAuth.user.uid;
      await signOut(auth);
      
      // Re-authenticate current user
      // (You may need to use a custom token or stored credential)
      await signInWithCustomToken(auth, currentToken); // Or your re-auth method
      
      // Resolve conflict
      const resolveResponse = await fetch('/api/v1/resolve-github-conflict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          current_user_uid: currentUser.uid,
          conflicting_github_uid: conflictingUid,
        }),
      });
      
      if (!resolveResponse.ok) {
        const errorData = await resolveResponse.json();
        throw new Error(errorData.error || 'Failed to resolve conflict');
      }
      
      // Retry linking
      return await linkGitHubWithConflictResolution(currentUser);
      
    } else {
      throw error;
    }
  }
}
```

---

## Testing Checklist

- [ ] SSO login works
- [ ] New user flow (no GitHub) shows linking UI
- [ ] Existing user with GitHub works
- [ ] Existing user without GitHub shows linking UI
- [ ] GitHub linking works for new users
- [ ] Credential conflict is resolved automatically
- [ ] Expired GitHub token is detected
- [ ] Error messages are user-friendly

---

## Support

For questions or issues, contact the backend team or refer to:
- Backend API docs: `/api/v1/docs` (if Swagger enabled)
- Firebase Auth docs: https://firebase.google.com/docs/auth

---

## Changelog

### 2025-01-XX
- Added `github_token_valid` field to SSOLoginResponse
- Added `/api/v1/resolve-github-conflict` endpoint
- Added automatic Firebase provider syncing
