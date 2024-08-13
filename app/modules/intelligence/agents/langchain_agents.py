from langchain_openai import ChatOpenAI
from app.modules.intelligence.agents.base_agent import BaseAgent
from app.modules.intelligence.tools.duckduckgo_search_tool import DuckDuckGoSearchTool

class LangChainAgent(BaseAgent):

    def __init__(self, model_name="gpt-4-turbo"):
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        self.tools = self.load_tools()

    def load_tools(self):
        ddg_search = DuckDuckGoSearchTool()
        return [ddg_search]

    def run(self, input_data: str):
        tool_results = [tool.execute(input_data) for tool in self.tools]
        prompt = f"Search results: {tool_results}. Now, {input_data}"
        
        # Get the response from the language model
        response = self.llm.invoke(prompt) 
        
        # Assuming the response is an AIMessage object, extract the content
        if isinstance(response, list) and response:
            message_content = response[0].content  # Extract content from the first AIMessage
        else:
            message_content = response.content  # Handle cases where it's not a list

        return message_content