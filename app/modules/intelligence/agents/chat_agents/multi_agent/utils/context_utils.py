"""Context utility functions for multi-agent system"""

from app.modules.intelligence.agents.chat_agent import ChatContext


def create_project_context_info(ctx: ChatContext) -> str:
    """Create project context information for both supervisor and subagents"""
    if ctx.node_ids is None:
        ctx.node_ids = []
    if isinstance(ctx.node_ids, str):
        ctx.node_ids = [ctx.node_ids]

    # Add image context information
    image_context = ""
    if ctx.has_images():
        all_images = ctx.get_all_images()
        image_details = []
        for attachment_id, image_data in all_images.items():
            file_name = image_data.get("file_name", "unknown")
            file_size = image_data.get("file_size", 0)
            image_details.append(f"- {file_name} ({file_size} bytes)")

        image_context = f"""
ATTACHED IMAGES:
{chr(10).join(image_details)}

Image Analysis Notes:
- These images are provided for visual analysis and debugging
- Reference specific details from the images in your response
- Correlate visual evidence with the user's query"""

    context_parts = []

    # Project information
    if ctx.project_id:
        context_parts.append(f"Project: {ctx.project_name} (ID: {ctx.project_id})")

    # Node information
    if ctx.node_ids:
        context_parts.append(f"Nodes: {', '.join(ctx.node_ids)}")

    # Image context
    if image_context:
        context_parts.append(image_context.strip())

    # Additional context
    if ctx.additional_context:
        context_parts.append(f"Additional Context: {ctx.additional_context}")

    return "\n".join(context_parts) if context_parts else "No additional context"


def create_supervisor_task_description(ctx: ChatContext) -> str:
    """Create a task description for the supervisor agent"""
    return create_project_context_info(ctx)
