from langchain_openai import ChatOpenAI
from app.modules.intelligence.agents.base_agent import BaseAgent
from app.modules.intelligence.tools.duckduckgo_search_tool import DuckDuckGoSearchTool

class LangChainAgent(BaseAgent):

    def __init__(self, model_name="gpt-4-turbo"):
        self.llm = ChatOpenAI(model=model_name, temperature=0, streaming=True)
        self.tools = self.load_tools()

    def load_tools(self):
        ddg_search = DuckDuckGoSearchTool()
        return [ddg_search]

    async def run(self, input_data: str):
        tool_results = [tool.execute(input_data) for tool in self.tools]
        prompt = f"Search results: {tool_results}. Now, {input_data}"

        # Call the appropriate method for streaming the response
        async for chunk in self.llm_stream(prompt):
            yield chunk

    async def llm_stream(self, prompt: str):
        response = self.llm.invoke(prompt)

        # Manually iterate over the content if the response is not directly iterable
        if hasattr(response, '__aiter__'):
            async for message in response:
                content = message.get('content', '').strip()
                if content:  # Only yield non-empty content
                    yield content
        else:
            # If streaming is not supported, manually break the response
            for chunk in response.content.split('\n'):
                chunk = chunk.strip()
                if chunk:  # Only yield non-empty chunks
                    yield chunk

