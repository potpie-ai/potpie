# Contributing to Potpie

Thank you for your interest in contributing to Potpie! We welcome and appreciate all contributions, whether you’re fixing bugs, improving documentation, or adding new features. This guide will help you get started.

## Table of Contents
1. [Code of Conduct](#code-of-conduct)
2. [How to Contribute](#how-to-contribute)
3. [Getting Started](#getting-started)
4. [Development Workflow](#development-workflow)
5. [Submitting Pull Requests](#submitting-pull-requests)
6. [Community and Support](#community-and-support)

---

## Code of Conduct

Please note that by participating in the Potpie project, you agree to abide by our [Code of Conduct](./CODE_OF_CONDUCT.md). This is to ensure a positive, welcoming environment for all contributors.

## How to Contribute

There are several ways you can contribute to Potpie:
- Reporting bugs or suggesting improvements.
- Submitting feature requests.
- Adding documentation and improving existing content.
- Reviewing pull requests.
- Writing code and fixing issues.

If you’re unsure about how to get started, feel free to browse our issues labeled `good first issue` and `help wanted`. We also welcome you to join our discussions to understand the project better.

## Getting Started

### Prerequisites

Ensure you have the following software installed:
- [Git](https://git-scm.com/)
- [Python](https://www.python.org/) (or your project's primary language)
- [Docker](https://www.docker.com/) (if applicable)

### Fork and Clone the Repository

1. **Fork** the repository on GitHub.
2. **Clone** your forked repository:
   ```bash
   git clone https://github.com/your-username/potpie.git
   cd potpie
   ```
3. Add the main repository as a remote:
   ```bash
   git remote add upstream https://github.com/potpie-ai/potpie.git
   ```

### Set Up Your Environment

Follow the project’s installation and setup instructions in the [README.md](./README.md) file to prepare your local environment.

## Development Workflow

To ensure consistency and quality in the codebase, please adhere to the following workflow:

1. **Create a Branch**:
   - Branches should be named according to the feature or bug fix. For example:
     ```bash
     git checkout -b feature/your-feature-name
     ```

2. **Make Changes**: Write clear, concise code and ensure that you follow any established coding conventions.


3. **Commit Your Changes**:
   - Write descriptive commit messages. Use imperative mood and provide context if necessary:
     ```bash
     git commit -m "Add feature to handle XYZ"
     ```

4. **Push to Your Fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

## Testing

Testing is essential for maintaining the stability of Potpie. We encourage you to:
- Write unit tests for new features.
- Update existing tests when modifying functionality.
- Ensure all tests pass before submitting a pull request.

## Submitting Pull Requests

When you’re ready to submit your changes:

1. **Open a Pull Request (PR)**:
   - Go to the Potpie repository and select **New Pull Request**.
   - Choose your branch and provide a meaningful title and description.

2. **Describe Your Changes**:
   - Include a detailed description of what you have done and why.
   - Link related issues (e.g., `Closes #123`).

3. **Request a Review**:
   - After submitting your PR, request a review from a project maintainer.

4. **Respond to Feedback**:
   - Be responsive to any feedback you receive. Once changes are approved, a maintainer will merge your PR.

## FAQs

### What is the difference between the ENV and the isDevelopmentMode environment variables ?
`ENV` signifies the environment the app is running in,
signfying configuration to load in dev / stage or prod env.

`isDevelopmentMode` tells the app that it can run without dependencies such as firebase, GitHub configuration etc.

The `ENV=development` still needs firebase / gcp / GitHub etc because it simply is about running potpie's backend locally.
But `isDevelopmentMode=enabled` disables all these to bring in support for running without most dependencies and support local parsing for example.


## Community and Support

Open a Github issue to connect with other contributors, ask questions, and get support.

Thank you for your contribution to Potpie! 🥧
