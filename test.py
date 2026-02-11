from pydantic_ai import Agent
from dataclasses import dataclass
from dotenv import load_dotenv
import logfire
import os

load_dotenv()
logfire.configure()

logfire.instrument_pydantic_ai()

@dataclass
class Deps:
    tenant: str
    
agent = Agent(
    model="gpt-4o-mini",  # gpt-3.5-turbo is deprecated
    deps_type=Deps,
    system_prompt="You are an expert in Bollywood critically acclaimed movies."
)

result = agent.run_sync(
    "What are the reviews about the movie Raincoat?",
    deps=Deps(tenant='tenant-123'),
)

print(result.output)
print(result.all_messages())
print(f"Usage: {result.usage()}")