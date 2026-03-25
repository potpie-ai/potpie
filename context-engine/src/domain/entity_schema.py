"""Typed entity/edge schemas used for Graphiti extraction."""

from typing import Optional

from pydantic import BaseModel


class PullRequest(BaseModel):
    pr_number: Optional[int] = None
    title: Optional[str] = None
    why_summary: Optional[str] = None
    change_type: Optional[str] = None
    feature_area: Optional[str] = None
    author: Optional[str] = None
    merged_at: Optional[str] = None
    files_changed: Optional[int] = None


class Commit(BaseModel):
    sha: Optional[str] = None
    message: Optional[str] = None
    author: Optional[str] = None
    branch: Optional[str] = None


class Issue(BaseModel):
    issue_number: Optional[int] = None
    title: Optional[str] = None
    problem_statement: Optional[str] = None


class Feature(BaseModel):
    feature_name: Optional[str] = None
    description: Optional[str] = None


class Decision(BaseModel):
    decision_made: Optional[str] = None
    alternatives_rejected: Optional[str] = None
    rationale: Optional[str] = None


class Developer(BaseModel):
    github_login: Optional[str] = None
    display_name: Optional[str] = None
    expertise_areas: Optional[str] = None


class Modified(BaseModel):
    file_path: Optional[str] = None


class Fixes(BaseModel):
    confidence: Optional[float] = None


class PartOfFeature(BaseModel):
    confidence: Optional[float] = None


class MadeIn(BaseModel):
    confidence: Optional[float] = None


class AuthoredBy(BaseModel):
    confidence: Optional[float] = None


class Owns(BaseModel):
    confidence: Optional[float] = None


ENTITY_TYPES = {
    "PullRequest": PullRequest,
    "Commit": Commit,
    "Issue": Issue,
    "Feature": Feature,
    "Decision": Decision,
    "Developer": Developer,
}

EDGE_TYPES = {
    "Modified": Modified,
    "Fixes": Fixes,
    "PartOfFeature": PartOfFeature,
    "MadeIn": MadeIn,
    "AuthoredBy": AuthoredBy,
    "Owns": Owns,
}

EDGE_TYPE_MAP = {
    ("PullRequest", "Commit"): ["MadeIn"],
    ("PullRequest", "Issue"): ["Fixes"],
    ("PullRequest", "Feature"): ["PartOfFeature"],
    ("PullRequest", "Developer"): ["AuthoredBy"],
    ("Commit", "Developer"): ["AuthoredBy"],
    ("Developer", "Feature"): ["Owns"],
    ("Commit", "PullRequest"): ["MadeIn"],
}
