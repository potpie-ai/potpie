from .schema import CreateConversationRequest
from app.modules.intelligence.agents.agent_registry import get_agent

class ConversationService:
    
    def create_conversation(self, conversation: CreateConversationRequest):
        # Generate a mock conversation ID (could be replaced with DB logic if needed)
        conversation_id = "mock-conversation-id"
        
        print("conversation title",conversation.title)

        # Perform a search using the DuckDuckGo agent
        agent = get_agent("langchain_duckduckgo")
        search_result = agent.run("Search for momentum")

        # Return the generated ID and search result as the first message
        return conversation_id, search_result
