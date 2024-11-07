# Firebase Setup
To set up Firebase, follow these steps:
1. **Create a Firebase Project**: Go to [Firebase Console](https://console.firebase.google.com/) and create a new project.
2. **Generate a Service Account Key**:
   - Click on **Project Overview** from the sidebar.
   - Open the **Service Accounts** tab.
   - Click on the option to generate a new private key in the Firebase Admin SDK sub-section.
   - Read the warning and generate the key. Rename the downloaded key to `firebase_service_account.json` and move it to the root of the potpie source code.
---
# Portkey Integration
Portkey provides observability and monitoring capabilities for AI integration with Potpie.
- **Sign Up**: Create a free account at [Portkey](https://app.portkey.ai/signup) and keep your API key in .env as PORTKEY_API_KEY.
---
# Setting Up GitHub App
To enable login via GitHub, create a GitHub app by following these steps:
1. Visit [GitHub App Creation](https://github.com/settings/apps/new).
2. **Name Your App**: Choose a name relevant to Potpie (e.g., `potpie-auth`).
3. **Set Permissions**:
   - **Repository Permissions**:
     - Contents: Read Only
     - Metadata: Read Only
     - Pull Requests: Read and Write
     - Secrets: Read Only
     - Webhook: Read Only
   - **Organization Permissions**: Members : Read Only
   - **Account Permissions**: Email Address: Read Only
4. **Generate a Private Key**: Download the private key and add it to env under `GITHUB_PRIVATE_KEY`. Add your app ID to `GITHUB_APP_ID`.
5. **Install the App**: From the left sidebar, select **Install App** and install it next to your organization/user account.
---
# Enabling GitHub Auth on Firebase
1. Open Firebase and navigate to **Authentication**.
2. Enable GitHub sign-in capability by adding a GitHub OAuth app from your account. This will provide you with a client secret and client ID to add to Firebase.
3. Copy the callback URL from Firebase and add it to your GitHub app.
GitHub Auth with Firebase is now ready.
---
# Google Cloud Setup
Potpie uses Google Secret Manager to securely manage API keys. If you created a Firebase app, a linked Google Cloud account will be automatically created. You can use that or create a new one as needed.
Follow these steps to set up the Secret Manager and Application Default Credentials (ADC) for Potpie:
1. Set up the Secret Manager.
2. Configure Application Default Credentials for local use.
Once completed, you are ready to proceed with the Potpie setup.
---
# Running Potpie
1. **Ensure Docker is Installed**: Verify that Docker is installed and running on your system.
2. **Set Up the Environment**: Create a `.env` file based on the provided `.env.template` in the repository. This file should include all necessary configuration settings for the application.
3. **Google Cloud Authentication**: Log in to your Google Cloud account and set up Application Default Credentials (ADC). Detailed instructions can be found in the documentation. Alternatively place the service account key file for your gcp project in service-account.json file in the root of the codebase.
5. **Run Potpie**: Execute the following command:
   ```bash
   ./start.sh
   ```
   You may need to make it executable by running:
   ```bash
   chmod +x start.sh
   ```
