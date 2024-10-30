# Potpie Getting Started Guide

## Firebase Setup

To set up Firebase, follow these steps:

1. **Create a Firebase Project**  
   Navigate to the [Firebase Console](https://console.firebase.google.com/) and create a new project.

2. **Generate a Service Account Key**  
   - Select **Project Overview** from the sidebar.
   - Go to the **Service Accounts** tab.
   - In the Firebase Admin SDK section, generate a new private key.
   - Confirm the warning, download the key, rename it to `firebase_service_account.json`, and move it to the root of the Potpie source code.

---

## Portkey Integration

Portkey offers observability and monitoring capabilities for AI integration with Potpie.

- **Sign Up**: Create a free account at [Portkey](https://app.portkey.ai/signup) and add your API key to `.env` as `PORTKEY_API_KEY`.

---

## GitHub App Setup

To enable GitHub login, create a GitHub app with these steps:

1. Open the [GitHub App Creation page](https://github.com/settings/apps/new).
2. **App Name**: Choose a relevant name, such as `potpie-auth`.
3. **Set Permissions**:
   - **Repository Permissions**:  
     - Contents: Read Only  
     - Metadata: Read Only  
     - Pull Requests: Read and Write  
     - Secrets: Read Only  
     - Webhooks: Read Only  
   - **Organization Permissions**:  
     - Members: Read Only  
   - **Account Permissions**:  
     - Email Address: Read Only  
4. **Generate a Private Key**: Download the private key, then add it to `.env` as `GITHUB_PRIVATE_KEY`. Add your app ID to `GITHUB_APP_ID`.
5. **Install the App**: From the left sidebar, select **Install App** and install it for your organization or user account.

---

## Enabling GitHub Authentication on Firebase

1. Open Firebase and go to **Authentication**.
2. Enable GitHub sign-in by adding a GitHub OAuth app. Obtain the client secret and client ID and add them to Firebase.
3. Copy the callback URL from Firebase and add it to your GitHub app.

---

## Google Cloud Setup

Potpie uses Google Secret Manager to securely manage API keys. When a Firebase app is created, a Google Cloud account is automatically linked. Use this account or create a new one if necessary.

Follow these steps for Secret Manager and Application Default Credentials (ADC) setup for Potpie:

1. Set up the Secret Manager.
2. Configure ADC for local use.

---

## Running Potpie

1. **Verify Docker Installation**  
   Ensure Docker is installed and running.

2. **Configure Environment Variables**  
   Create a `.env` file based on the `.env.template` provided in the repository. Include all required configuration settings.

3. **Google Cloud Authentication**  
   Log in to your Google Cloud account and set up ADC, or place the service account key file for your GCP project as `service-account.json` in the root directory.

4. **Start Potpie**  
   Run the command:
   ```bash
   ./start.sh
   ```
   If needed, make it executable with:
   ```bash
   chmod +x start.sh
   ```

---

# API Workflow Guide

## Prerequisites

1. Ensure the API server is accessible at `http://localhost:8001`.
2. Have `curl` installed to execute API requests.

### Step 1: Log in to Get a Bearer Token
```bash
curl -X 'POST' \
  'http://localhost:8001/api/v1/login' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "email": "your_email",
  "password": "your_password"
}'
```

### Step 2: Submit a Parsing Request
Replace `repo_name` and `branch_name` with the appropriate repository details.
```bash
curl -X 'POST' \
  'http://localhost:8001/api/v1/parse' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "repo_name": "repository_name",
  "branch_name": "branch_name"
}'
```

### Step 3: Check Parsing Status
Use the project ID from the previous response.
```bash
curl -X 'GET' \
  'http://localhost:8001/api/v1/parsing-status/project-id' \
  -H 'accept: application/json'
```

### Step 4: List Available Agents
```bash
curl -X 'GET' \
  'http://localhost:8001/api/v1/list-available-agents/' \
  -H 'accept: application/json'
```

### Step 5: Create a Conversation
```bash
curl -X 'POST' \
  'http://localhost:8001/api/v1/conversations/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "user_id": "your_user_id",
  "title": "Conversation Title",
  "status": "active",
  "project_ids": ["project_id"],
  "agent_ids": ["agent_id"]
}'
```

### Step 6: Send Messages in a Conversation
```bash
curl -X 'POST' \
  'http://localhost:8001/api/v1/conversations/conversation_id/message/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "content": "Your message content",
  "node_ids": [{"node_id": "node_identifier", "name": "node_name"}]
}'
```

### Step 7: Retrieve All Messages of a Conversation
```bash
curl -X 'GET' \
  'http://localhost:8001/api/v1/conversations/conversation_id/messages/?start=0&limit=10' \
  -H 'accept: application/json'
```

---

## Additional Information

For further details on additional API endpoints and comprehensive documentation, please visit [docs.potpie.ai](https://docs.potpie.ai).

---

## Workflow Summary

1. **Submit a Parsing Request**: Start the parsing of a repository.
2. **Check Parsing Status**: Monitor parsing progress.
3. **List Available Agents**: View accessible agents.
4. **Create a Conversation**: Start a new conversation.
5. **Send Messages**: Communicate within a conversation session.
