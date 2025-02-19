# Potpie Project Usage Summary

## Overview
Potpie is a CLI tool designed to manage and interact with repositories, perform codebase analysis, and communicate with AI-powered agents for various development tasks. Below is a summary of my experience using Potpie.

## Installation
To install Potpie from the built distribution:
```sh
pip install cli/dist/potpie-0.0.1-py3-none-any.whl
```
This installs all dependencies required for Potpie to function properly.

## Starting the Server
After installation, the Potpie server can be started with:
```sh
potpie start
```
### Logs:
- Environment set to **development**
- Docker containers started successfully
- PostgreSQL server detected and running
- Database migrations completed successfully
- Server started at `http://localhost:8001`

## Listing Projects
To list all projects currently available in Potpie:
```sh
potpie projects
```
### Output:
```
╒══════╤════════╤══════════╕
│ ID   │ Name   │ Status   │
╞══════╪════════╪══════════╡
╘══════╧════════╧══════════╛
```

## Parsing a Repository
Parsing a local repository with Potpie is straightforward:
```sh
potpie parse "/home/deepesh/Development/public/opensource/you-get"
```
### Parsing Status Updates:
- `submitted`
- `parsed`
- `ready`

Once completed, the project appears in the projects list:
```
╒══════════════════════════════════════╤═════════╤══════════╕
│ ID                                   │ Name    │ Status   │
╞══════════════════════════════════════╪═════════╪══════════╡
│ 01950a23-27f0-7b7d-bcb9-5d6c65773253 │ you-get │ ready    │
╘══════════════════════════════════════╧═════════╧══════════╛
```

## Starting a Conversation
To interact with the AI-powered agents:
```sh
potpie conversation create "My first conversation with youget"
```
After selecting the project and agent, the conversation is successfully created.

### Available Agents:
1. Codebase Q&A Agent
2. Debugging with Knowledge Graph Agent
3. Unit Test Agent
4. Integration Test Agent
5. Low-Level Design Agent
6. Code Changes Agent
7. Code Generation Agent

To start messaging:
```sh
potpie conversation message
```
Example interaction:
```
You: hello, can you tell me about the codebase? and what it is doing?
Bot: The "you-get" project is a CLI utility for downloading media content from various websites...
```

## Stopping the Server
To stop the Potpie server and its associated services:
```sh
potpie stop
```
### Logs:
- Terminated active processes
- Stopped Docker containers (Redis, PostgreSQL, Neo4j)
- All services stopped successfully

## Summary
- Successfully installed and started Potpie
- Parsed and listed repositories
- Created and engaged in a conversation with an AI agent
- Explored the functionality of "you-get" through the conversation feature
- Stopped the Potpie server and services gracefully

Potpie proves to be an efficient tool for managing codebases, analyzing repositories, and interacting with AI-driven development assistants. 🚀

