# Development mode
## Running Potpie
**Install Python 3.10**: Download and install Python 3.10 from the official Python website:
https://www.python.org/downloads/release/python-3100/
1. **Ensure Docker is Installed**: Verify that Docker is installed and running on your system.
2. **Set Up the Environment**: Create a `.env` file based on the provided `.env.template` in the repository. This file should include all necessary configuration settings for the application.
   Ensure that:
   ```
   isDevelopmentMode=enabled
   ENV=development
   OPENAI_API_KEY=<your-openai-key>
   ```
   Create a Virtual Environment using Python 3.10:
   ```bash
   python3.10 -m venv venv
   source venv/bin/activate
   ```
   alternatively, you can also use the `virtualenv` library.

   Install dependencies in your venv:
   ```bash
   pip install -r requirements.txt
   ```
   If you face any issues with the dependencies, you can try installing the dependencies using the following command:
   ```bash
   pip install -r requirements.txt --use-deprecated=legacy-resolver
   ```

3. You can use the following env config to run potpie with local models:
   ```
   INFERENCE_MODEL=ollama_chat/qwen2.5-coder:7b
   CHAT_MODEL=ollama_chat/qwen2.5-coder:7b
   ```

   To run potpie with any other models, you can use the following env configuration:
   ```
   {PROVIDER}_API_KEY=sk-or-your-key #your provider key e.g. OPENAI_API_KEY for Openai
   INFERENCE_MODEL=openrouter/deepseek/deepseek-chat #provider model name
   CHAT_MODEL=openrouter/deepseek/deepseek-chat #provider model name
   ```
   **`INFERENCE_MODEL`** and **`CHAT_MODEL`** correspond to the models that will be used for generating knowledge graph and for agent reasoning respectively. These model names should be in the format of `provider/model_name` format or as expected by Litellm. For more information, refer to the [Litellm documentation](https://docs.litellm.ai/docs/providers).
   <br>
4. **Run Potpie**: Execute the following command:
   ```bash
   ./start.sh
   ```
   You may need to make it executable by running:
   ```bash
   chmod +x start.sh
   ```
5. Start using Potpie with your local codebases!


# Production setup
For a production deployment with Firebase authentication, Github access, Secret Management etc

## Firebase Setup
To set up Firebase, follow these steps:
1. **Create a Firebase Project**: Go to [Firebase Console](https://console.firebase.google.com/) and create a new project.
2. **Generate a Service Account Key**:
   - Click on **Project Overview Gear ⚙** from the sidebar.
   - Open the **Service Accounts** tab.
   - Click on the option to generate a new private key in the Firebase Admin SDK sub-section.
   - Read the warning and generate the key. Rename the downloaded key to `firebase_service_account.json` and move it to the root of the potpie source code.
3. **Create a Firebase App**
   - Go to the **Project Overview Gear ⚙** from the sidebar.
   - Create a Firebase app.
   - You will find keys for hosting, storage, and other services. Use these keys in your `.env` file.
---
## PostHog Integration
PostHog is an open-source platform that helps us analyze user behavior on Potpie.
- **Sign Up**: Create a free account at [PostHog](https://us.posthog.com/signup) and keep your API key in `.env` as `POSTHOG_API_KEY`, and `POSTHOG_HOST`
---
## Portkey Integration
Portkey provides observability and monitoring capabilities for AI integration with Potpie.
- **Sign Up**: Create a free account at [Portkey](https://app.portkey.ai/signup) and keep your API key in `.env` as `PORTKEY_API_KEY`.
---
## Setting Up GitHub App
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
   - **Homepage URL** : https://potpie.ai
   - **Webhook** : Inactive
4. **Generate a Private Key**: Download the private key and place it in the project root . Add your app ID to `GITHUB_APP_ID`.
5. **Format your Private Key**: Use the `format_pem.sh` to format your key:
   ```bash
   chmod +x format_pem.sh
   ./format_pem.sh your-key.pem
   ```
   The formatted key will be displayed in the terminal. Copy the formatted key and add it to env under `GITHUB_PRIVATE_KEY`.
6. **Install the App**: From the left sidebar, select **Install App** and install it next to your organization/user account.
7. **Create a GitHub Token**: Go to your GitHub Settings > Developer Settings > Personal Access Tokens > Tokens (classic). Add the token to your `.env` file under `GH_TOKEN_LIST`
---
## Enabling GitHub Auth on Firebase
1. Open Firebase and navigate to **Authentication**.
2. Enable GitHub sign-in capability by adding a GitHub OAuth app from your account. This will provide you with a client secret and client ID to add to Firebase.
3. Copy the callback URL from Firebase and add it to your GitHub app.
GitHub Auth with Firebase is now ready.
---
## Google Cloud Setup
Potpie uses Google Secret Manager to securely manage API keys. If you created a Firebase app, a linked Google Cloud account will be automatically created. You can use that or create a new one as needed.
Follow these steps to set up the Secret Manager and Application Default Credentials (ADC) for Potpie:
1. Install gcloud CLI. Follow the official installation guide:
   https://cloud.google.com/sdk/docs/install

   After installation, initialize gcloud CLI:
   ```bash
   gcloud init
   ```
   Say yes to configuring a default compute region.
   Select your local region when prompted.
2. Set up the gcloud Secret Manager API.
3. Configure Application Default Credentials for local use:
   https://cloud.google.com/docs/authentication/set-up-adc-local-dev-environment

Once completed, you are ready to proceed with the Potpie setup.
---
## Running Potpie
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
