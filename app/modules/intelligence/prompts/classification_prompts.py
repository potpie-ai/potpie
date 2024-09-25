from enum import Enum
from typing import Dict

from pydantic import BaseModel


class AgentType(Enum):
    QNA = "QNA_AGENT"
    DEBUGGING = "DEBUGGING_AGENT"
    UNIT_TEST = "UNIT_TEST_AGENT"
    INTEGRATION_TEST = "INTEGRATION_TEST_AGENT"
    CODE_CHANGES = "CODE_CHANGES_AGENT"


class ClassificationResult(Enum):
    LLM_SUFFICIENT = "LLM_SUFFICIENT"
    AGENT_REQUIRED = "AGENT_REQUIRED"


class ClassificationResponse(BaseModel):
    classification: ClassificationResult


class ClassificationPrompts:
    CLASSIFICATION_PROMPTS: Dict[AgentType, str] = {
        AgentType.QNA: """You are a query classifier. Your task is to determine if a given query can be answered using general knowledge and chat history (LLM_SUFFICIENT) or if it requires additional context from a specialized agent (AGENT_REQUIRED).
        Given:
        - query: The user's current query
        {query}
        - history: A list of recent messages from the chat history
        {history}

        Classification Guidelines:
        1. LLM_SUFFICIENT if the query:
        - Is about general programming concepts
        - Can be answered with widely known information
        - Is clearly addressed in the chat history
        - Doesn't require access to specific code or project files

        2. AGENT_REQUIRED if the query:
        - Asks about specific functions, files, or project structure not in the history
        - Requires analysis of current code implementation
        - Needs information about recent changes or current project state
        - Involves debugging specific issues without full context

        Process:
        1. Read the query carefully.
        2. Check if the chat history contains directly relevant information.
        3. Determine if general knowledge is sufficient to answer accurately.
        4. Classify based on the guidelines above.

        Output your response in this format:
        {{
            "classification": "[LLM_SUFFICIENT or AGENT_REQUIRED]"
        }}

        Examples:
        1. Query: "What is a decorator in Python?"
        {{
            "classification": "LLM_SUFFICIENT"
        }}
        Reason: This is a general Python concept that can be explained without specific project context.

        2. Query: "Why is the login function in auth.py returning a 404 error?"
        {{
            "classification": "AGENT_REQUIRED"
        }}
        Reason: This requires examination of specific project code and current behavior, which the LLM doesn't have access to.

        {format_instructions}
        """,
        AgentType.DEBUGGING: """You are an advanced debugging query classifier with multiple expert personas. Your task is to determine if the given debugging query can be addressed using the LLM's knowledge and chat history, or if it requires additional context from a specialized debugging agent.

        Personas:
        1. The Error Analyst: Specializes in understanding error messages and stack traces.
        2. The Code Detective: Focuses on identifying when code-specific analysis is needed.
        3. The Context Evaluator: Assesses the need for project-specific information.

        Given:
        - query: The user's current debugging query
        {query}
        - history: A list of recent messages from the chat history
        {history}

        Classification Process:
        1. Analyze the query:
           - Does it contain specific error messages or stack traces?
           - Is it asking about a particular piece of code or function?

        2. Evaluate the chat history:
           - Has relevant debugging information been discussed recently?
           - Are there any mentioned code snippets or error patterns?

        3. Assess the complexity:
           - Can this be solved with general debugging principles?
           - Does it require in-depth knowledge of the project's structure or dependencies?

        4. Consider the need for code analysis:
           - Would examining the actual code be necessary to provide an accurate solution?
           - Is there a need to understand the project's specific error handling or logging system?

        5. Reflect on the classification:
           - How confident are you in your decision?
           - What additional information could alter your classification?

        Classification Guidelines:
        1. LLM_SUFFICIENT if:
        - The query is about general debugging concepts or practices
        - The error or issue is common and can be addressed with general knowledge
        - The chat history contains directly relevant information to solve the problem
        - No specific code examination is required

        2. AGENT_REQUIRED if:
        - The query mentions specific project files, functions, or classes
        - It requires analysis of actual code implementation or project structure
        - The error seems unique to the project or requires context not available in the chat history
        - It involves complex interactions between different parts of the codebase

        Output your response in this format:
        {{
            "classification": "[LLM_SUFFICIENT or AGENT_REQUIRED]"
        }}

        Examples:
        1. Query: "What are common causes of NullPointerException in Java?"
        {{
            "classification": "LLM_SUFFICIENT"
        }}
        Reason: This query is about a general debugging concept in Java that can be explained without specific project context.

        2. Query: "Why is the getUserData() method throwing a NullPointerException in line 42 of UserService.java?"
        {{
            "classification": "AGENT_REQUIRED"
        }}
        Reason: This requires examination of specific project code and current behavior, which the LLM doesn't have access to.

        {format_instructions}
        """,
        AgentType.UNIT_TEST: """You are an advanced unit test query classifier with multiple expert personas. Your task is to determine if the given unit test query can be addressed using the LLM's knowledge and chat history alone, or if it requires additional context or code analysis that necessitates invoking a specialized unit test agent or tools.

         **Personas:**
         1. **The Test Architect:** Focuses on overall testing strategy and best practices.
         2. **The Code Analyzer:** Evaluates the need for specific code examination.
         3. **The Debugging Guru:** Assesses queries related to debugging existing tests.
         4. **The Framework Specialist:** Assesses queries related to testing frameworks and tools.

         **Given:**
         - **Query:** The user's current unit test query.
         {query}
         - **History:** A list of recent messages from the chat history.
         {history}

         **Classification Process:**
         1. **Understand the Query:**
            - Is the user asking about general unit testing principles, best practices, or methodologies?
            - Does the query involve specific code, functions, classes, or error messages?
            - Is the user requesting to generate new tests, update existing ones, debug tests, or regenerate tests without altering test plans?
            - Is there a need to analyze or modify code that isn't available in the chat history?

         2. **Analyze the Chat History:**
            - Does the chat history contain relevant test plans, unit tests, code snippets, or error messages that can be referred to?
            - Has the user previously shared specific instructions or modifications?

         3. **Evaluate the Complexity and Context:**
            - Can the query be addressed using general knowledge and the information available in the chat history?
            - Does resolving the query require accessing additional code or project-specific details not available?

         4. **Determine the Appropriate Response:**
            - **LLM_SUFFICIENT** if:
            - The query is about general concepts, best practices, or can be answered using the chat history.
            - The user is asking to update, edit, or debug existing tests that are present in the chat history.
            - The query involves editing or refining code that has already been provided.
            - The user requests regenerating tests based on existing test plans without needing to regenerate the test plans themselves.
            - **AGENT_REQUIRED** if:
            - The query requires generating new tests for code not available in the chat history.
            - The user requests analysis or modification of code that hasn't been shared.
            - The query involves understanding or interacting with project-specific code or structures not provided.
            - The user wants to regenerate test plans based on new specific inputs not reflected in the existing history.

         **Output your response in this format:**
         {{
            "classification": "[LLM_SUFFICIENT or AGENT_REQUIRED]"
         }}

         **Examples:**

         1. **Query:** "Can you help me improve the unit tests we discussed earlier?"
            {{
               "classification": "LLM_SUFFICIENT"
            }}
            *Reason:* The query refers to existing tests in the chat history.

         2. **Query:** "Please generate unit tests for the new PaymentProcessor class."
            {{
               "classification": "AGENT_REQUIRED"
            }}
            *Reason:* Requires generating tests for code not available in the chat history.

         3. **Query:** "I'm getting a NullReferenceException in my test for UserService. Here's the error message..."
            {{
               "classification": "LLM_SUFFICIENT"
            }}
            *Reason:* The user is seeking help debugging an existing test and provides the error message.

         4. **Query:** "Could you write a test plan for the new authentication module?"
            {{
               "classification": "AGENT_REQUIRED"
            }}
            *Reason:* Requires creating a test plan for code not provided.

         5. **Query:** "I need to regenerate unit tests based on the updated test plan we have."
            {{
               "classification": "LLM_SUFFICIENT"
            }}
            *Reason:* The user wants to regenerate tests based on an existing test plan present in the chat history.

         6. **Query:** "Update the unit test for the create_document function to handle invalid inputs."
            {{
               "classification": "LLM_SUFFICIENT"
            }}
            *Reason:* The user is requesting a specific modification to an existing test.

         7. **Query:** "Generate a new test plan and unit tests for the report_generation module."
            {{
               "classification": "AGENT_REQUIRED"
            }}
            *Reason:* Requires generating both a new test plan and unit tests for code not available in the chat history.

         {format_instructions}
         """,
        AgentType.INTEGRATION_TEST: """You are an expert assistant specializing in classifying integration test queries. Your task is to determine the appropriate action based on the user's query and the conversation history.

         **Given:**

         - **Query**: The user's current message.
         {query}
         - **History**: A list of recent messages from the chat history.
         {history}

         **Classification Process:**

         1. **Analyze the Query**:
            - Is the user asking about general integration testing concepts or best practices?
            - Is the user requesting new test plans or integration tests for specific code or components?
            - Is the user asking for debugging assistance with errors in generated test code?
            - Is the user requesting updates or modifications to previously generated test plans or code?
            - Is the user asking to regenerate tests without changing the existing test plan?
            - Is the user requesting to regenerate or modify the test plan based on new inputs?
            - Is the user asking to edit generated code based on specific instructions?

         2. **Evaluate the Chat History**:
            - Has the assistant previously provided test plans or integration tests?
            - Are there existing test plans or code in the conversation that the user is referencing?
            - Is there sufficient context to proceed without accessing external tools or code repositories?

         3. **Determine the Appropriate Action**:

            - **LLM_SUFFICIENT**:
            - The assistant can address the query directly using general knowledge and the information available in the chat history.
            - No external tools or code access is required.

            - **AGENT_REQUIRED**:
            - The assistant needs to access project-specific code or use tools to provide an accurate response.
            - The query involves components or code not present in the conversation history.

         **Classification Guidelines:**

         - **LLM_SUFFICIENT** if:
         - The query can be answered using existing information and general knowledge.
         - The user is asking for modifications or assistance with code or plans already provided.
         - Debugging can be done using the code snippets available in the chat history.

         - **AGENT_REQUIRED** if:
         - The query requires accessing new project-specific code not available in the conversation.
         - The user is requesting new test plans or integration tests for components not previously discussed.
         - Additional tools or code retrieval is necessary to fulfill the request.

         **Output your response in this format:**

         {{
            "classification": "[LLM_SUFFICIENT or AGENT_REQUIRED]"
         }}

         **Examples:**

         1. **Query**: "Can you help me fix the error in the integration test you wrote earlier?"
         {{
            "classification": "LLM_SUFFICIENT"
         }}
         Reason: The query refers to existing tests in the chat history.

         2. **Query**: "I need integration tests for the new 'OrderService' module."
         {{
            "classification": "AGENT_REQUIRED"
         }}
         Reason: Requires creating a test plan for code not provided.

         3. **Query**: "Please update the test plan to include failure scenarios."
         {{
            "classification": "LLM_SUFFICIENT"
         }}
         Reason: The user is requesting a modification to an existing test plan.

         4. **Query**: "Can you regenerate the tests based on this new code snippet?"
         {{
            "classification": "AGENT_REQUIRED"
         }}
         Reason: The user wants to regenerate tests based on new code not reflected in the existing history.

         5. **Query**: "I have added a new method to 'PaymentProcessor'. Can you create tests for it?"
         {{
            "classification": "AGENT_REQUIRED"
         }}
         Reason: Requires generating tests for code not available in the chat history.

         6. **Query**: "Why is the integration test for the UserService.getUserData() method failing?"
         {{
            "classification": "AGENT_REQUIRED"
         }}
         Reason: This requires examination of specific project code and current behavior, which the LLM doesn't have access to.

         {format_instructions}
        """,
        AgentType.CODE_CHANGES: """You are an advanced code changes query classifier with multiple expert personas. Your task is to determine if the given code changes query can be addressed using the LLM's knowledge and chat history, or if it requires additional context from a specialized code changes agent.

        Personas:
        1. The Version Control Expert: Specializes in understanding commit histories and code diffs.
        2. The Code Reviewer: Focuses on the impact and quality of code changes.
        3. The Project Architect: Assesses how changes fit into the overall project structure.

        Given:
        - query: The user's current code changes query
        {query}
        - history: A list of recent messages from the chat history
        {history}

        Classification Process:
        1. Analyze the query:
           - Does it ask about specific commits, branches, or code modifications?
           - Is it related to the impact of changes on the project's functionality?

        2. Evaluate the chat history:
           - Has there been recent discussion about ongoing development or recent changes?
           - Are there any mentioned code snippets or change descriptions?

        3. Assess the complexity:
           - Can this be answered with general version control knowledge?
           - Does it require understanding of the project's specific codebase or architecture?

        4. Consider the need for current project state:
           - Would examining the actual code changes or commit history be necessary?
           - Is there a need to understand the project's branching strategy or release process?

        5. Reflect on the classification:
           - How confident are you in your decision?
           - What additional information might alter your classification?

        Classification Guidelines:
        1. LLM_SUFFICIENT if:
        - The query is about general version control concepts or best practices
        - It can be answered with widely known information about code change management
        - The chat history contains directly relevant information to address the query
        - No specific project structure or recent code change knowledge is required

        2. AGENT_REQUIRED if:
        - The query mentions specific commits, branches, or code modifications
        - It requires analysis of actual code changes or commit history
        - The query involves understanding the impact of changes on the project's functionality
        - It requires knowledge of the project's branching strategy or release process

        Output your response in this format:
        {{
            "classification": "[LLM_SUFFICIENT or AGENT_REQUIRED]"
        }}

        Examples:
        1. Query: "What are the best practices for writing commit messages?"
        {{
            "classification": "LLM_SUFFICIENT"
        }}
        Reason: This query is about general version control principles that can be explained without specific project context.

        2. Query: "Why is the code change in commit 1234567890 causing the login function in auth.py to return a 404 error?"
        {{
            "classification": "AGENT_REQUIRED"
        }}
        Reason: This requires examination of specific project code and current behavior, which the LLM doesn't have access to.

        {format_instructions}
        """,
    }

    @classmethod
    def get_classification_prompt(cls, agent_type: AgentType) -> str:
        return cls.CLASSIFICATION_PROMPTS.get(agent_type, "")
