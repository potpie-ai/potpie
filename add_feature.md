Firebase SSO Implementation - Detailed Requirements Document
Overview
Modify existing Firebase Google and GitHub SSO authentication to enforce work email requirements and block new GitHub signups while maintaining backward compatibility for legacy users.

Core Requirements
1. Google Sign-In (Continue with Google)
Current State: Users can sign in with any Google account (Gmail, work email, etc.)
Required Behavior:

✅ ALLOW: Users with work/corporate email addresses from ANY company domain

Examples: john@acmecorp.com, sarah@techstartup.io, mike@consulting.co.uk


❌ BLOCK: Users with generic/personal email providers

Examples: user@gmail.com, user@yahoo.com, user@outlook.com



Implementation Rules:

Block generic emails for NEW signups only
For existing users with generic emails: [CHOOSE ONE]

Option A (Recommended): Allow them to continue (legacy users grandfathered)
Option B (Strict): Block them and show migration message


Delete Firebase user account immediately if blocked
Sign out user if blocked
Show clear error message explaining why they were blocked


2. GitHub Sign-In (Continue with GitHub)
Current State: Users can sign in/sign up with any GitHub account
Required Behavior:

✅ ALLOW: Existing users who previously signed up via GitHub (legacy users)
❌ BLOCK: All new signups via GitHub
✅ KEEP: The "Continue with GitHub" button visible on UI (for legacy users)

Implementation Rules:

Check if user is new using result.additionalUserInfo?.isNewUser or result._tokenResponse?.isNewUser
If new user → Delete account, sign out, show error
If existing user → Allow sign-in normally
Error message should guide new users to use Google with work email


Blocked Email Domains List
Must Block These Generic/Personal Email Providers:
Google:
- gmail.com
- googlemail.com

Microsoft:
- outlook.com
- hotmail.com
- live.com
- msn.com

Yahoo:
- yahoo.com
- yahoo.co.uk
- yahoo.co.in
- yahoo.fr
- yahoo.de
- ymail.com
- rocketmail.com

Apple:
- icloud.com
- me.com
- mac.com

Other Popular Providers:
- aol.com
- protonmail.com
- proton.me
- mail.com
- zoho.com
- yandex.com
- yandex.ru
- gmx.com
- gmx.de
- mail.ru
- fastmail.com
- hushmail.com
- tutanota.com
- tutanota.de
- rediffmail.com
- inbox.com

Temporary/Disposable Email Services:
- tempmail.com
- 10minutemail.com
- guerrillamail.com
- mailinator.com
- maildrop.cc
- throwaway.email
- temp-mail.org
- getnada.com
- minuteinbox.com
Note: This list should be maintained in a Set for O(1) lookup performance.

Technical Implementation Details
Authentication Flow
User clicks "Continue with Google" or "Continue with GitHub"
  ↓
Firebase opens OAuth popup
  ↓
User authenticates with provider
  ↓
Firebase returns result with user info
  ↓
[VALIDATION CHECKPOINT - YOUR CODE]
  ↓
Check: Is this GitHub AND new user?
  → YES: Delete user, sign out, show error "GitHub signups disabled"
  → NO: Continue to email check
  ↓
Check: Is email domain in BLOCKED_DOMAINS?
  → YES: 
    - If new user: Delete user, sign out, show error "Use work email"
    - If existing user: [Option A: Allow] OR [Option B: Block with migration message]
  → NO: Allow sign-in
  ↓
Successful authentication → Redirect to app/dashboard

Code Structure Requirements
1. Create Utility Functions
javascript/**
 * Checks if an email domain is from a generic/personal email provider
 * @param {string} email - User's email address
 * @returns {boolean} - True if generic email, false if work email
 */
function isGenericEmail(email) {
  // Implementation needed
}

/**
 * Checks if user is a new signup
 * @param {object} result - Firebase auth result
 * @returns {boolean} - True if new user
 */
function isNewUser(result) {
  // Implementation needed
}
2. Provider Sign-In Handlers
javascript/**
 * Handles Google OAuth sign-in with work email validation
 * @throws {Error} If user has generic email (for new users)
 * @returns {Promise<User|null>} Firebase user or null
 */
async function handleGoogleSignIn() {
  // Implementation needed:
  // 1. Call signInWithPopup with googleProvider
  // 2. Check if new user
  // 3. Validate email domain
  // 4. Delete user if blocked
  // 5. Return user or throw error
}

/**
 * Handles GitHub OAuth sign-in (legacy users only)
 * @throws {Error} If user is attempting new signup
 * @returns {Promise<User|null>} Firebase user or null
 */
async function handleGitHubSignIn() {
  // Implementation needed:
  // 1. Call signInWithPopup with githubProvider
  // 2. Check if new user
  // 3. Block if new user
  // 4. Allow if existing user
  // 5. Return user or throw error
}
```

### 3. Error Handling

Must handle these error cases:
- `auth/popup-closed-by-user` - User closed OAuth popup (don't show error)
- `auth/cancelled-popup-request` - Multiple popups opened
- `auth/popup-blocked` - Browser blocked popup
- Custom validation errors (blocked email, blocked GitHub signup)

### 4. User Feedback

Error messages must be clear and actionable:

**For blocked Gmail/generic email**:
```
❌ Personal email addresses are not allowed.

Please use your work/corporate email to sign in.
Examples: yourname@yourcompany.com

Generic email providers (Gmail, Yahoo, Outlook, etc.) cannot be used.
```

**For blocked GitHub signup**:
```
❌ GitHub sign-up is no longer supported.

Please use "Continue with Google" with your work email address.

Note: If you previously created an account with GitHub, you can still sign in.
```

**For existing users with generic emails (if Option B chosen)**:
```
❌ Personal email addresses are no longer allowed.

Your account was created with a personal email. Please contact support at support@yourcompany.com to migrate your account to a work email.

Edge Cases to Handle
1. Race Conditions

User clicks sign-in button multiple times rapidly
Solution: Disable button during authentication, re-enable on completion

2. Deleted User Still in Session

User is deleted but session might briefly exist
Solution: Always call auth.signOut() after deleting user

3. Network Errors During Deletion

User created but deletion fails due to network issue
Solution: Wrap deletion in try-catch, log error, still sign out user

4. User Closes Popup Before Completion

Don't show error message for this case
Solution: Check for auth/popup-closed-by-user error code

5. Google Workspace Emails

Companies using Google Workspace have emails like user@company.com but authenticate through Google
Solution: Only check domain, not authentication provider
✅ john@acmecorp.com via Google → ALLOWED
❌ john@gmail.com via Google → BLOCKED

6. Subdomains

Some companies use subdomains: user@eng.company.com, user@sales.company.com
Solution: Only check the root domain after '@', not subdomains
✅ All work email subdomains should be allowed

7. Case Sensitivity

Email domains can be mixed case: User@GmAiL.CoM
Solution: Always convert email to lowercase before checking

8. International Domains

Yahoo has country-specific domains: yahoo.co.uk, yahoo.fr
Solution: Include all variants in blocked list

9. Existing Session

User might already be signed in when clicking button
Solution: Check auth.currentUser before initiating sign-in

10. Multiple Accounts

User might have multiple Google accounts in browser
Solution: Let user choose, validate after selection


Testing Checklist
Google Sign-In Tests

 New user with Gmail → Should be blocked
 New user with Yahoo → Should be blocked
 New user with Outlook → Should be blocked
 New user with work email (user@company.com) → Should succeed
 New user with work email via Google Workspace → Should succeed
 Existing user with Gmail (legacy) → [Should allow if Option A] OR [Should block if Option B]
 User closes popup → Should not show error
 User with User@GMAIL.COM (mixed case) → Should be blocked
 User with subdomain work email (user@eng.company.com) → Should succeed

GitHub Sign-In Tests

 New user trying to sign up → Should be blocked
 Existing GitHub user → Should succeed
 User closes popup → Should not show error
 Error message directs to Google sign-in

UI/UX Tests

 Button states update correctly (loading, disabled)
 Error messages are clear and helpful
 Success redirects to correct page
 Legacy label on GitHub button is visible
 Work email requirement is clearly communicated


Configuration Variables
Required Constants
javascript// Set of blocked email domains
const BLOCKED_DOMAINS = new Set([...]);

// Firebase providers
const googleProvider = new GoogleAuthProvider();
const githubProvider = new GithubAuthProvider();

// Redirect URL after successful sign-in
const POST_AUTH_REDIRECT = '/dashboard';

// Support contact for migration
const SUPPORT_EMAIL = 'support@yourcompany.com';
Policy Choice
javascript// Choose how to handle existing users with generic emails
const LEGACY_USER_POLICY = 'ALLOW'; // or 'BLOCK'

Security Considerations
Client-Side Only is Not Enough
Warning: Client-side validation can be bypassed. For production, you MUST implement server-side validation.
Recommended: Firebase Cloud Functions
Implement these blocking functions:
javascript// functions/index.js

/**
 * Runs before user account is created
 * Blocks GitHub signups and generic email signups
 */
exports.beforeUserCreated = functions.auth.user().beforeCreate((user) => {
  // Validate provider and email
  // Throw error if blocked
});

/**
 * (Optional) Runs before user signs in
 * Can enforce policy on existing users gradually
 */
exports.beforeUserSignedIn = functions.auth.user().beforeSignIn((user) => {
  // Additional validation
});
This ensures validation happens even if:

User modifies client-side JavaScript
User uses API directly
User uses old version of your app


UI Updates Required
Sign-In Page Updates
Before:
html<button id="google-signin">Continue with Google</button>
<button id="github-signin">Continue with GitHub</button>
After:
html<button id="google-signin">
  Continue with Google (Work Email Required)
</button>

<button id="github-signin">
  Continue with GitHub (Existing Users Only)
</button>

<p class="auth-note">
  ⚠️ Please use your corporate/work email address.
  Personal email providers (Gmail, Yahoo, Outlook, etc.) are not allowed.
</p>
CSS Considerations

Add visual distinction for legacy GitHub button (e.g., muted colors, "Legacy" badge)
Show loading state on buttons during authentication
Disable buttons during authentication to prevent double-clicks


Logging and Monitoring
Events to Log
javascript// Successful authentications
console.log('AUTH_SUCCESS', { email, provider, isNewUser });

// Blocked attempts
console.log('AUTH_BLOCKED', { email, provider, reason: 'generic_email' });
console.log('AUTH_BLOCKED', { email, provider, reason: 'github_signup' });

// Errors
console.error('AUTH_ERROR', { code, message, email, provider });
Metrics to Track

Number of blocked Gmail attempts per day
Number of blocked GitHub signup attempts per day
Number of successful work email signups
Number of legacy GitHub users still active
Error rates by provider


Migration Path for Existing Users (Optional)
If you choose Option B (block existing users with generic emails):
1. Grace Period

Detect legacy users with generic emails
Show warning banner: "Please migrate to work email by [DATE]"
Allow continued access during grace period

2. Migration Flow

Provide UI to add work email to existing account
Verify work email with confirmation link
Transfer data to new email
Delete old authentication method

3. Support Process

Document how users can contact support
Provide clear instructions in error message
Have support team manually verify and migrate accounts if needed


Implementation Steps
Phase 1: Setup

Create constants for BLOCKED_DOMAINS set
Set up Firebase providers (if not already done)
Create utility functions (isGenericEmail, isNewUser)
Choose policy for existing users (LEGACY_USER_POLICY)

Phase 2: Core Logic

Implement handleGoogleSignIn with validation
Implement handleGitHubSignIn with new user blocking
Add error handling for all edge cases
Implement user deletion and sign-out logic

Phase 3: UI/UX

Update button labels and add explanatory text
Add loading states to buttons
Implement clear error messages with actionable guidance
Add visual distinction for legacy GitHub button

Phase 4: Testing

Test all scenarios from testing checklist
Test on multiple browsers (Chrome, Firefox, Safari, Edge)
Test on mobile devices
Test with various email formats and edge cases

Phase 5: Production Safety

(Recommended) Implement Firebase Cloud Functions for server-side validation
Add logging for monitoring
Set up alerts for unusual blocking patterns
Document support process for blocked users

Phase 6: Deployment

Deploy to staging environment first
Test with real accounts (if possible)
Monitor logs for any issues
Deploy to production with monitoring enabled
Communicate changes to users if necessary


Code Template Structure
javascript// ============================================
// CONFIGURATION
// ============================================
import { getAuth, signInWithPopup, GoogleAuthProvider, GithubAuthProvider } from 'firebase/auth';

const auth = getAuth();
const googleProvider = new GoogleAuthProvider();
const githubProvider = new GithubAuthProvider();

const BLOCKED_DOMAINS = new Set([
  // ... full list of blocked domains
]);

const POST_AUTH_REDIRECT = '/dashboard';
const LEGACY_USER_POLICY = 'ALLOW'; // or 'BLOCK'

// ============================================
// UTILITY FUNCTIONS
// ============================================

function isGenericEmail(email) {
  // TODO: Implement
}

function isNewUser(result) {
  // TODO: Implement
}

// ============================================
// AUTHENTICATION HANDLERS
// ============================================

async function handleGoogleSignIn() {
  // TODO: Implement
  // 1. signInWithPopup
  // 2. Check if new user
  // 3. Validate email
  // 4. Delete if blocked
  // 5. Handle errors
  // 6. Redirect on success
}

async function handleGitHubSignIn() {
  // TODO: Implement
  // 1. signInWithPopup
  // 2. Check if new user
  // 3. Block if new
  // 4. Allow if existing
  // 5. Handle errors
  // 6. Redirect on success
}

// ============================================
// EVENT LISTENERS
// ============================================

document.getElementById('google-signin')?.addEventListener('click', async () => {
  // TODO: Disable button, call handler, re-enable button
});

document.getElementById('github-signin')?.addEventListener('click', async () => {
  // TODO: Disable button, call handler, re-enable button
});

// ============================================
// ERROR HANDLING
// ============================================

function showError(message) {
  // TODO: Implement user-friendly error display
}

function logAuthEvent(event, data) {
  // TODO: Implement logging
}

Success Criteria
The implementation is complete when:
✅ New users with Gmail/Yahoo/Outlook are blocked from Google sign-in
✅ New users with work emails can sign in via Google successfully
✅ New users cannot sign up via GitHub
✅ Existing GitHub users can still sign in
✅ Existing Google users with generic emails [allowed/blocked per policy choice]
✅ All error messages are clear and actionable
✅ All edge cases are handled gracefully
✅ No console errors during normal operation
✅ Button states update correctly (loading, disabled)
✅ User account is deleted when blocked
✅ User is signed out when blocked
✅ Works on all major browsers
✅ Works on mobile devices
✅ Code is well-commented and maintainable

Questions to Answer Before Implementation

Legacy User Policy: Should existing users with generic emails be allowed (Option A) or blocked (Option B)?
Post-Auth Redirect: Where should users be redirected after successful authentication?
Support Contact: What email/link should be provided for users who need help?
Error Display: How should errors be displayed? (alert, toast, modal, inline message)
Button Loading States: What should buttons show during authentication? (spinner, "Loading...", disabled)
Logging: Where should auth events be logged? (console, analytics service, backend API)
Server-Side Validation: Will Firebase Cloud Functions be implemented? (Highly recommended for production)


Additional Notes

Email validation is case-insensitive
Only the domain is checked, not the username part
Subdomains in work emails are allowed (e.g., eng.company.com)
Google Workspace emails using Google OAuth are allowed if domain is not in blocked list
The blocked domains list should be maintained and updated as new generic providers emerge
Consider adding a feedback mechanism for users to report if their legitimate work email was incorrectly blocked