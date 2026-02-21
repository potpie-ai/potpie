from enum import Enum
from typing import Dict

from pydantic import BaseModel


class AgentType(Enum):
    QNA = "QNA_AGENT"
    DEBUGGING = "DEBUGGING_AGENT"
    CODE_CHANGES = "CODE_CHANGES_AGENT"
    LLD = "LLD_AGENT"
    GENERAL = "GENERAL_PURPOSE_AGENT"


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
        AgentType.LLD: """You are a Low Level Design (LLD) classifier. Your task is to determine if a design query can be answered using general knowledge (LLM_SUFFICIENT) or requires leveraging the knowledge graph and code-fetching capabilities (AGENT_REQUIRED).

        Given:
        - query: The user's current query
        {query}
        - history: A list of recent messages from the chat history
        {history}

        Classification Guidelines:
        1. LLM_SUFFICIENT if the combined context (query + history):
        - Discusses general design patterns or principles
        - Requests theoretical design approaches
        - Involves new design with no dependencies on existing code
        - Contains all necessary context within the conversation

        2. AGENT_REQUIRED if the combined context (query + history):
        - References specific files, classes, or functions
        - Requires understanding existing codebase structure
        - Involves modifying or extending existing designs
        - Uses pronouns or references to previously discussed components
        - Needs compatibility analysis with current implementation

        Process:
        1. Review chat history for referenced components and context
        2. Analyze if query builds upon previous design discussions
        3. Check if codebase context would enhance the response
        4. Classify based on the combined context of query and history

        Output your response in this format:
        {{
            "classification": "[LLM_SUFFICIENT or AGENT_REQUIRED]"
        }}

        Examples:
        1. History: "Let's design a new caching system"
           Query: "What pattern should we use for cache invalidation?"
        {{
            "classification": "LLM_SUFFICIENT"
        }}
        Reason: Discusses general design patterns without specific implementation context.

        2. History: "Our UserService handles authentication"
           Query: "How should we add password reset?"
        {{
            "classification": "AGENT_REQUIRED"
        }}
        Reason: Requires understanding of existing UserService implementation.

        3. History: ""
           Query: "Design a notification system that follows our existing event handling patterns"
        {{
            "classification": "AGENT_REQUIRED"
        }}
        Reason: Requires analysis of existing event handling patterns in codebase even without specific file references.

        {format_instructions}
        """,
        AgentType.GENERAL: """
        You are a general purpose query classifier. Your task is to determine if the query can be answered with just given history or
        will require access to internet

        Given:
        - query: The user's current query
        {query}
        - history: A list of recent messages from the chat history
        {history}

        Classification Guidelines:
        1. LLM_SUFFICIENT if the combined context (query + history):
        - If the query could be responded well with just the query and history

        2. AGENT_REQUIRED if the combined context (query + history):
        - If we need internet access and deep thinking

        Output your response in this format:
        {{
            "classification": "[LLM_SUFFICIENT or AGENT_REQUIRED]"
        }}

        Examples:
        1. History: "*Some code snippet*"
           Query: "Can you refactor above code to accomodate dependency injection?"
        {{
            "classification": "LLM_SUFFICIENT"
        }}
        Reason: No need to access external content

        2. History: "PydanticAI helps to build ai agents from scratch"
           Query: "How can i implement retry mechanism in pydanticai agents"
        {{
            "classification": "AGENT_REQUIRED"
        }}
        Reason: Requires fetching documentation and other resources from the web

        {format_instructions}
        """,
    }

    REDUNDANT_INHIBITION_TAIL: str = "\n\nReturn ONLY JSON content, and nothing else. Don't provide reason or any other text in the response."

    @classmethod
    def get_classification_prompt(cls, agent_type: AgentType) -> str:
        return (
            cls.CLASSIFICATION_PROMPTS.get(agent_type, "")
            + cls.REDUNDANT_INHIBITION_TAIL
        )
