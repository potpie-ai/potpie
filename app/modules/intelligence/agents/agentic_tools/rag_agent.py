from typing import Annotated, Any, AsyncGenerator, Dict, List, Sequence, TypedDict, Union
from typing_extensions import TypedDict
import asyncio
from contextlib import redirect_stdout
import os
import logging
from pydantic import BaseModel, Field
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, AIMessageChunk
from langchain.tools import BaseTool
from langgraph.graph import END, StateGraph, MessageGraph, START
from langgraph.prebuilt import ToolNode

from app.modules.conversations.message.message_schema import NodeContext
from app.modules.github.github_service import GithubService
from app.modules.intelligence.tools.code_query_tools.get_node_neighbours_from_node_id_tool import (
    get_node_neighbours_from_node_id_tool,
)
from app.modules.intelligence.tools.kg_based_tools.ask_knowledge_graph_queries_tool import (
    get_ask_knowledge_graph_queries_tool,
)
from app.modules.intelligence.tools.kg_based_tools.get_code_from_multiple_node_ids_tool import (
    GetCodeFromMultipleNodeIdsTool,
    get_code_from_multiple_node_ids_tool,
)
from app.modules.intelligence.tools.kg_based_tools.get_code_from_probable_node_name_tool import (
    get_code_from_probable_node_name_tool,
)
from app.modules.intelligence.tools.kg_based_tools.get_nodes_from_tags_tool import (
    get_nodes_from_tags_tool,
)
logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__)
# State definitions
class RAGState(TypedDict):
    """Type for the graph state"""
    messages: Sequence[BaseMessage]
    project_id: str
    chat_history: List
    node_ids: List[NodeContext] 
    file_structure: str
    code_results: List[Dict[str, Any]]
    sql_db: Any
    user_id: str

def create_agent_prompt(state: RAGState) -> str:
    """Creates the agent prompt from the state"""
    query = state['messages'][-1].content if state['messages'] else ''
    
    return f"""Adhere to {os.getenv("MAX_ITER", 5)} iterations max. Analyze input:
    - Chat History: {state['chat_history']}
    - Query: {query}
    - Project ID: {state['project_id']}
    - User Node IDs: {[node.model_dump() for node in state['node_ids']]}
    - File Structure: {state['file_structure']}
    - Code Results: {state['code_results']}
    
    0. Analyze chat history and query to determine if the query is a follow up question or a new question. If the question can be answered using the chat history and the context from the query itself, then answer the question immediately.

    1. Analyze project structure:
       - Identify key directories, files, and modules
       - Guide search strategy and provide context
       - Locate files relevant to query
       - Use relevant file names with "Get Code and docstring From Probable Node Name" tool

    2. Initial context retrieval:
       - Analyze provided Code Results for user node ids
       - If code results are not relevant move to next step

    3. Knowledge graph query (if needed):
       - Transform query for knowledge graph tool
       - Execute query and analyze results

    4. Additional context retrieval (if needed):
       - Extract probable node names
       - Use "Get Code and docstring From Probable Node Name" tool

    5. Use "Get Nodes from Tags" tool as last resort only if absolutely necessary

    6. Analyze and enrich results:
       - Evaluate relevance, identify gaps
       - Develop scoring mechanism
       - Retrieve code only if docstring insufficient

    7. Compose response:
       - Organize results logically
       - Include citations and references
       - Provide comprehensive, focused answer

    8. Final review:
       - Check coherence and relevance
       - Identify areas for improvement
       - Format any file paths as follows (only include relevant project details from file path):
         path: potpie/projects/username-reponame-branchname-userid/gymhero/models/training_plan.py
         output: gymhero/models/training_plan.py

    Note:
    - Prioritize "Get Code and docstring From Probable Node Name" tool for stacktraces or specific file/function mentions
    - Use available tools as directed
    - Proceed to next step if insufficient information found
    - Use markdown for code snippets with language name in the code block like ```python or ```javascript

    Ground your responses in provided code context and tool results. Use markdown for code snippets. Be concise and avoid repetition. 
    If unsure, state it clearly. 

    Tailor your response based on question type:
    - New questions: Provide comprehensive answers
    - Follow-ups: Build on previous explanations from the chat history
    - Clarifications: Offer clear, concise explanations
    - Comments/feedback: Incorporate into your understanding

    Indicate when more information is needed. Use specific code references. Adapt to user's expertise level. Maintain a conversational tone and context from previous exchanges.
    Ask clarifying questions if needed. Offer follow-up suggestions to guide the conversation.

    Final Output: Markdown formatted chat response to user's query grounded in provided code context and tool results"""


def create_agent_node(llm):
    """Creates the agent node function"""
    
    async def agent_node(state: RAGState):
        """Agent node implementation"""
        prompt = create_agent_prompt(state)
        
        response = await llm.ainvoke([
            HumanMessage(content=prompt)
        ])

        
        # Check if tools are needed
        if hasattr(response, 'tool_calls') and response.tool_calls:
            state["messages"].append(response)
            return state
        
        # If no tools needed, return response directly
        state["messages"].append(AIMessageChunk(content=response.content))
        return state
    
    return agent_node

def should_continue(state: RAGState):
    """Determines if we should continue to tools or end"""
    messages = state["messages"]
    last_message = messages[-1]
    
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    return END

class RAGGraph:
    def __init__(self, sql_db, llm, mini_llm, user_id: str):
        self.sql_db = sql_db
        self.llm = llm
        self.mini_llm = mini_llm
        self.user_id = user_id
        self.max_iter = int(os.getenv("MAX_ITER", "5"))
        
        # Initialize tools
        self.tools = [
            get_nodes_from_tags_tool(sql_db, user_id),
            get_ask_knowledge_graph_queries_tool(sql_db, user_id),
            get_code_from_multiple_node_ids_tool(sql_db, user_id),
            get_code_from_probable_node_name_tool(sql_db, user_id),
            get_node_neighbours_from_node_id_tool(sql_db)
        ]
        
    def create_graph(self) -> StateGraph:
        """Creates the LangGraph workflow"""
        llm_with_tools = self.llm.bind_tools(self.tools)
        # Create the graph
        workflow = StateGraph(RAGState)
        
        # Add nodes
        workflow.add_node("agent", create_agent_node(llm_with_tools))
        workflow.add_node("tools", ToolNode(self.tools))
        
        # Add edges - Fixed to use START instead of END as initial node
        workflow.add_edge(START, "agent")
        workflow.add_conditional_edges(
            "agent",
            should_continue,
            [
                "tools",
                END
            ]
        )
        workflow.add_edge("tools", "agent")
        
        return workflow.compile()


async def kickoff_rag_crew(
    query: str,
    project_id: str,
    chat_history: List,
    node_ids: List[NodeContext],
    sql_db,
    llm,
    mini_llm,
    user_id: str,
) -> AsyncGenerator[str, None]:
    """Main entry point - maintains same interface as CrewAI version"""
    
    # Initialize graph
    rag = RAGGraph(sql_db, llm, mini_llm, user_id)
    graph = rag.create_graph()
    
    # Get file structure
    file_structure = await GithubService(sql_db).get_project_structure_async(project_id)
    
    # Get initial code results if we have node IDs
    code_results = []
    if len(node_ids) > 0:
        code_results = await GetCodeFromMultipleNodeIdsTool(
            sql_db, user_id
        ).run_multiple(project_id, [node.node_id for node in node_ids])
    
    # Prepare initial state
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "project_id": project_id,
        "chat_history": chat_history,
        "node_ids": node_ids,
        "file_structure": file_structure,
        "code_results": code_results,
        "sql_db": sql_db,
        "user_id": user_id
    }
    

    async for msg in graph.astream(initial_state, stream_mode="values"):
        if isinstance(msg["messages"][-1], AIMessage):
            yield msg["messages"][-1].content
