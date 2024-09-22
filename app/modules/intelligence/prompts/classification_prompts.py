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
        AgentType.UNIT_TEST: """You are an advanced unit test query classifier with multiple expert personas. Your task is to determine if the given unit test query can be addressed using the LLM's knowledge and chat history, or if it requires additional context from a specialized unit test agent.

        Personas:
        1. The Test Architect: Focuses on overall testing strategy and best practices.
        2. The Code Analyzer: Evaluates the need for specific code examination.
        3. The Framework Specialist: Assesses queries related to testing frameworks and tools.

        Given:
        - query: The user's current unit test query
        {query}
        - history: A list of recent messages from the chat history
        {history}

        Classification Process:
        1. Understand the query:
           - Is it about general unit testing principles or specific to a piece of code?
           - Does it mention any particular testing framework or tool?

        2. Analyze the chat history:
           - Has there been any recent discussion about the project's testing setup?
           - Are there any mentioned code snippets or test cases?

        3. Evaluate the complexity:
           - Can this be answered with general unit testing knowledge?
           - Does it require understanding of the project's specific testing conventions?

        4. Consider the need for code context:
           - Would examining the actual code or existing tests be necessary?
           - Is there a need to understand the project's structure to provide a suitable answer?

        5. Reflect on the classification:
           - How confident are you in your decision?
           - What additional information might change your classification?

        Classification Guidelines:
        1. LLM_SUFFICIENT if:
        - The query is about general unit testing concepts or best practices
        - It can be answered with widely known information about testing frameworks
        - The chat history contains directly relevant information to address the query
        - No specific code or project structure knowledge is required

        2. AGENT_REQUIRED if:
        - The query mentions specific project files, functions, or classes to be tested
        - It requires analysis of actual code implementation or existing test suites
        - The query involves project-specific testing conventions or setup
        - It requires understanding of complex interactions between different parts of the codebase for effective testing

        Output your response in this format:
        {{
            "classification": "[LLM_SUFFICIENT or AGENT_REQUIRED]"
        }}

        Examples:
        1. Query: "What are the best practices for mocking dependencies in unit tests?"
        {{
            "classification": "LLM_SUFFICIENT"
        }}
        Reason: This query is about general unit testing principles that can be explained without specific project context.

        2. Query: "Why is the test case for the UserService.getUserData() method failing?"
        {{
            "classification": "AGENT_REQUIRED"
        }}
        Reason: This requires examination of specific project code and current behavior, which the LLM doesn't have access to.

        {format_instructions}
        """,
        AgentType.INTEGRATION_TEST: """You are an advanced integration test query classifier with multiple expert personas. Your task is to determine if the given integration test query can be addressed using the LLM's knowledge and chat history, or if it requires additional context from a specialized integration test agent.

        Personas:
        1. The System Architect: Focuses on understanding system components and their interactions.
        2. The Test Strategist: Evaluates the scope and complexity of integration testing scenarios.
        3. The Environment Specialist: Assesses the need for specific test environment knowledge.

        Given:
        - query: The user's current integration test query
        {query}
        - history: A list of recent messages from the chat history
        {history}

        Classification Process:
        1. Analyze the query:
           - Does it involve multiple system components or services?
           - Is it asking about specific integration points or data flows?

        2. Evaluate the chat history:
           - Has there been recent discussion about the system architecture or integration points?
           - Are there any mentioned test scenarios or integration issues?

        3. Assess the complexity:
           - Can this be answered with general integration testing principles?
           - Does it require in-depth knowledge of the project's architecture or dependencies?

        4. Consider the need for system-specific information:
           - Would understanding the actual system setup be necessary to provide an accurate answer?
           - Is there a need to know about specific APIs, databases, or external services?

        5. Reflect on the classification:
           - How confident are you in your decision?
           - What additional information could change your classification?

        Classification Guidelines:
        1. LLM_SUFFICIENT if:
        - The query is about general integration testing concepts or best practices
        - It can be answered with widely known information about testing methodologies
        - The chat history contains directly relevant information to address the query
        - No specific system architecture or project structure knowledge is required

        2. AGENT_REQUIRED if:
        - The query mentions specific system components, services, or APIs to be tested
        - It requires analysis of actual system architecture or existing integration test suites
        - The query involves project-specific integration points or data flows
        - It requires understanding of complex interactions between different parts of the system for effective testing

        Output your response in this format:
        {{
            "classification": "[LLM_SUFFICIENT or AGENT_REQUIRED]"
        }}

        Examples:
        1. Query: "What are the best practices for setting up a test environment for integration tests?"
        {{
            "classification": "LLM_SUFFICIENT"
        }}
        Reason: This query is about general integration testing principles that can be explained without specific project context.

        2. Query: "Why is the integration test for the UserService.getUserData() method failing?"
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
