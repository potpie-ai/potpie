from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.modules.auth.auth_service import AuthService
from app.modules.parsing.knowledge_graph.inference_service import InferenceService
from app.modules.projects.projects_service import ProjectService
from app.modules.projects.projects_model import Project
from app.modules.utils.logger import setup_logger

logger = setup_logger(__name__)
router = APIRouter()


class SemanticSearchRequest(BaseModel):
    query: str = Field(..., description="Natural language query for semantic search")
    project_id: str = Field(..., description="Project ID (UUID)")
    top_k: int = Field(default=10, ge=1, le=50, description="Number of results to return (1-50)")
    node_ids: Optional[List[str]] = Field(
        default=None,
        description="Optional list of node IDs to search within (for context-aware search)"
    )


class SemanticSearchResult(BaseModel):
    node_id: str
    file_path: str
    start_line: int
    end_line: int
    docstring: Optional[str] = None
    similarity: float
    name: Optional[str] = None
    type: Optional[str] = None


class SemanticSearchResponse(BaseModel):
    results: List[SemanticSearchResult]
    query: str
    project_id: str
    total_results: int


@router.post(
    "/knowledge-graph/semantic-search",
    response_model=SemanticSearchResponse,
    description="Perform semantic search on codebase using knowledge graph embeddings",
)
async def semantic_search(
    request: SemanticSearchRequest,
    db: Session = Depends(get_db),
    user=Depends(AuthService.check_auth),
):
    """Semantic search endpoint that queries Neo4j vector index.
    
    This endpoint uses embeddings stored in the knowledge graph to find
    semantically similar code based on natural language queries.
    """
    user_id = user["user_id"]
    
    # Verify project belongs to user
    project_service = ProjectService(db)
    try:
        # Project.id is Text (string UUID), but method signature expects int
        # Try to convert or query directly
        try:
            project_id_int = int(request.project_id)
            project = await project_service.get_project_repo_details_from_db(
                project_id_int, user_id
            )
        except (ValueError, TypeError):
            # If it's a UUID string that can't be converted to int, query directly
            project = (
                db.query(Project)
                .filter(Project.id == request.project_id, Project.user_id == user_id)
                .first()
            )
            if project:
                project = {
                    "id": project.id,
                    "repo_name": project.repo_name,
                    "branch_name": project.branch_name,
                    "user_id": project.user_id,
                    "repo_path": project.repo_path,
                    "commit_id": project.commit_id,
                }
        
        if not project:
            raise HTTPException(
                status_code=404,
                detail=f"Project {request.project_id} not found or access denied"
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error verifying project access: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error verifying project access: {str(e)}"
        )
    
    # Perform semantic search
    inference_service = InferenceService(db, user_id)
    try:
        results = inference_service.query_vector_index(
            project_id=request.project_id,  # Pass as string (Neo4j uses string IDs)
            query=request.query,
            node_ids=request.node_ids,
            top_k=request.top_k,
        )
        
        # Format results (name and type are now included in query results)
        formatted_results = [
            SemanticSearchResult(
                node_id=r.get("node_id", ""),
                file_path=r.get("file_path", ""),
                start_line=r.get("start_line", 0) or 0,
                end_line=r.get("end_line", 0) or 0,
                docstring=r.get("docstring"),
                similarity=float(r.get("similarity", 0.0)),
                name=r.get("name"),
                type=r.get("type"),
            )
            for r in results
        ]
        
        logger.info(
            f"Semantic search: query='{request.query}', project_id={request.project_id}, "
            f"found {len(formatted_results)} results"
        )
        
        return SemanticSearchResponse(
            results=formatted_results,
            query=request.query,
            project_id=request.project_id,
            total_results=len(formatted_results),
        )
    except Exception as e:
        logger.exception(f"Error in semantic search: {e}")
        raise HTTPException(status_code=500, detail=f"Semantic search failed: {str(e)}")
    finally:
        try:
            inference_service.close()
        except Exception:
            pass
