# **Poitre CLI Tool**

A command-line interface (CLI) tool to manage and interact with the Poitre application, including managing servers, parsing repositories, listing projects, and interacting with conversations.


## **Usage**

The CLI provides multiple commands for managing the application. Below is a list of commands and their functionalities.

### **Global Help**
```bash
poitre --help
```
Displays the list of available commands and their descriptions.

---

### **Server Commands**

#### **Start the Server**
```bash
poitre start
```
- Starts the server and all related services, including Docker containers, database migrations, and the Celery worker.

#### **Stop the Server**
```bash
poitre stop
```
- Stops the server and all related services.

---

### **Project Management**

#### **Parse a Repository**
```bash
poitre parse <repository_path> --branch <branch_name>
```
- Parses a local repository for the specified branch.
- **Arguments**:
  - `repository_path`: Path to the repository on the local machine.
  - `--branch`: (Optional) Name of the branch to parse (default: `main`).

#### **List Projects**
```bash
poitre projects
```
- Lists all projects with their ID, name, and status.

#### **Delete a Project**
```bash
poitre projects --delete
```
- Prompts the user to select and delete a project.

---

### **Conversation Management**

#### **List Conversations**
```bash
poitre  conversation list
```
- Lists all active conversations.

#### **Create a Conversation**
```bash
poitre  conversation create <title>
```
- Starts a new conversation with a project and an agent.
- **Arguments**:
  - `title`: Title of the conversation.

#### **Message an Agent**
```bash
poitre conversation message
```
- Interact with a selected agent in a specific conversation.

---

