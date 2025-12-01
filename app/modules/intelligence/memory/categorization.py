"""
Memory categorization utilities for user and project-level preferences
"""
import logging

logger = logging.getLogger(__name__)

# Define categories
USER_LEVEL_CATEGORIES = {
    "user_coding_style",
    "user_tool_preferences",
    "user_work_habits",
    "user_communication_style"
}

PROJECT_LEVEL_CATEGORIES = {
    "project_tech_stack",
    "project_database",
    "project_architecture",
    "project_infrastructure",
    "project_business_rules"
}


def categorize_memory(memory_text: str) -> str:
    """
    Categorize memory based on content keywords
    
    Since mem0ai OSS SDK doesn't support custom categories,
    we implement our own keyword-based categorization.
    
    Args:
        memory_text: The memory text to categorize
    
    Returns:
        Category name (one of USER_LEVEL or PROJECT_LEVEL categories) or "uncategorized"
    """
    memory_lower = memory_text.lower()
    
    # User-level categories (cross-project preferences)
    if any(kw in memory_lower for kw in [
        'camelcase', 'snake_case', 'indentation', 'naming', 'comment style',
        'code style', 'formatting', 'variable name'
    ]):
        return "user_coding_style"
    
    if any(kw in memory_lower for kw in [
        'fcm', 'firebase', 'auth0', 'sentry', 'datadog', 'notification service',
        'monitoring', 'analytics', 'push notification', 'alert service'
    ]):
        return "user_tool_preferences"
    
    if any(kw in memory_lower for kw in [
        'prefers', 'likes to', 'always uses', 'habit', 'workflow',
        'routine', 'working style', 'development approach'
    ]):
        return "user_work_habits"
    
    if any(kw in memory_lower for kw in [
        'communicate', 'updates', 'notifications', 'alert', 'inform'
    ]):
        return "user_communication_style"
    
    # Project-level categories (project-specific decisions)
    if any(kw in memory_lower for kw in [
        'nestjs', 'react', 'vue', 'angular', 'fastapi', 'django',
        'express', 'framework', 'library', 'next.js', 'nuxt'
    ]):
        return "project_tech_stack"
    
    if any(kw in memory_lower for kw in [
        'mysql', 'postgresql', 'mongo', 'redis', 'database', 'db',
        'sql', 'orm', 'prisma', 'typeorm', 'sequelize', 'timescaledb'
    ]):
        return "project_database"
    
    if any(kw in memory_lower for kw in [
        'architecture', 'microservice', 'monolith', 'design pattern',
        'structure', 'layered', 'hexagonal', 'clean architecture'
    ]):
        return "project_architecture"
    
    if any(kw in memory_lower for kw in [
        'deploy', 'hosting', 'ci/cd', 'docker', 'kubernetes', 'aws',
        'gcp', 'azure', 'vercel', 'heroku', 'infrastructure'
    ]):
        return "project_infrastructure"
    
    if any(kw in memory_lower for kw in [
        'business rule', 'requirement', 'workflow', 'process',
        'domain', 'business logic'
    ]):
        return "project_business_rules"
    
    return "uncategorized"

