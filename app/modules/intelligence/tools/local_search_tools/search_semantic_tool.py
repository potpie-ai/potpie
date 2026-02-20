"""
Semantic Search Tool

Search codebase using semantic understanding via knowledge graph embeddings.
This provides natural language code search capabilities.
"""

from typing import Optional, List
from pydantic import BaseModel, Field
from app.modules.utils.logger import setup_logger
from .tunnel_utils import route_to_local_server, get_context_vars

logger = setup_logger(__name__)


class SearchSemanticInput(BaseModel):
    query: str = Field(description="Natural language query (e.g., 'authentication code', 'error handling', 'database queries')")
    project_id: str = Field(description="Project ID (UUID) for the codebase to search")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of results to return (default: 10, max: 50)")
    node_ids: Optional[List[str]] = Field(
        default=None,
        description="Optional list of node IDs to search within (for context-aware search)"
    )


def search_semantic_tool(input_data: SearchSemanticInput) -> str:
    """Search codebase using semantic understanding via knowledge graph.
    
    This tool uses embeddings to find code that semantically matches the query,
    even if it doesn't contain the exact keywords. Perfect for natural language queries.
    
    Examples:
    - "authentication code" → finds login, auth, token validation, etc.
    - "error handling" → finds try-catch, error handlers, exception management
    - "database queries" → finds SQL, ORM calls, data access patterns
    - "user registration" → finds signup, create account, user creation
    
    The search uses vector embeddings stored in the knowledge graph, which are
    generated from docstrings and code context. Results are ranked by semantic similarity.
    
    This tool works best when:
    - The project has been parsed and knowledge graph is available
    - You're looking for code by meaning rather than exact keywords
    - You want to find related functionality across the codebase
    """
    logger.info(
        f"🔍 [Tool Call] search_semantic_tool: Query='{input_data.query}', "
        f"project_id={input_data.project_id}, top_k={input_data.top_k}"
    )
    
    user_id, conversation_id = get_context_vars()
    
    # Route to LocalServer (which will call backend knowledge graph)
    result = route_to_local_server(
        "search_semantic",
        {
            "query": input_data.query,
            "project_id": input_data.project_id,
            "top_k": input_data.top_k,
            "node_ids": input_data.node_ids,
        },
        user_id=user_id,
        conversation_id=conversation_id,
    )
    
    if result:
        logger.info("✅ [search_semantic_tool] Executed via LocalServer")
        return result
    
    # Fallback: Direct backend call if tunnel not available
    logger.info("⚠️ [search_semantic_tool] LocalServer not available, using direct backend call")
    try:
        from app.modules.parsing.knowledge_graph.inference_service import InferenceService
        from app.core.database import get_db
        
        db = next(get_db())
        inference_service = InferenceService(db, user_id)
        results = inference_service.query_vector_index(
            project_id=input_data.project_id,
            query=input_data.query,
            node_ids=input_data.node_ids,
            top_k=input_data.top_k,
        )
        
        if not results:
            return f"📋 No semantically similar code found for '{input_data.query}'. " \
                   f"Ensure the project is parsed and knowledge graph is available."
        
        formatted = f"📋 **Found {len(results)} semantically similar result(s) for '{input_data.query}':**\n\n"
        for i, r in enumerate(results[:input_data.top_k], 1):
            file_path = r.get("file_path", "unknown")
            start_line = r.get("start_line", 0) or 0
            similarity = r.get("similarity", 0.0)
            docstring = r.get("docstring", "")
            name = r.get("name", "")
            
            formatted += f"{i}. **{file_path}:{start_line}**"
            if name:
                formatted += f" - `{name}`"
            formatted += f" (similarity: {similarity:.3f})\n"
            
            if docstring:
                formatted += f"   {docstring[:200]}{'...' if len(docstring) > 200 else ''}\n"
            formatted += "\n"
        
        return formatted
    except Exception as e:
        logger.exception(f"Error in semantic search fallback: {e}")
        return (
            f"❌ Semantic search failed: {str(e)}\n\n"
            "Please ensure:\n"
            "1. Project is parsed and knowledge graph is available\n"
            "2. Tunnel is active for local execution, OR\n"
            "3. Backend knowledge graph service is accessible"
        )
