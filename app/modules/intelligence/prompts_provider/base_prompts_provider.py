from types import SimpleNamespace
from typing import List, Dict, Any
from abc import ABC

from app.modules.intelligence.prompts.prompt_model import PromptType
from app.modules.intelligence.prompts_provider.agent_types import AgentRuntimeLLMType
from app.modules.intelligence.prompts_provider.agent_types import SystemAgentType

      
class BasePromptsProvider(ABC):
    PROMPTS: Dict = {}
    CLASSIFICATION_PROMPTS: Dict = {}
    AGENTS_DICT: Dict[str, Dict[str, str]] ={}
    TASK_PROMPTS: Dict[str, Dict[str, str]] = {}

    @classmethod
    async def get_prompts(
        cls, agent_id: str,
        prompt_types: List[PromptType] = [PromptType.SYSTEM],
        **kwargs: Dict[str, Any],
    ) -> List[SimpleNamespace]:
        
        def list_to_dict(prompts_list):
            if not isinstance(prompts_list, list):
                return {} 
            return {
                "SYSTEM" if prompt.get("type", f"unknown_{i}") == PromptType.SYSTEM else "HUMAN": prompt
                for i, prompt in enumerate(prompts_list)
            }

        unified_prompts = {
        **list_to_dict(cls.PROMPTS.get(agent_id, {}).get("prompts", {})),
        **list_to_dict(cls.CLASSIFICATION_PROMPTS.get(agent_id, {}).get("prompts", {})),
        **list_to_dict(cls.AGENTS_DICT.get(agent_id, {}).get("prompts", {})),
        **list_to_dict(cls.TASK_PROMPTS.get(agent_id, {}).get("prompts", {})),
        }

        if agent_id in [agent_dict_id for agent_dict_id, _ in cls.AGENTS_DICT.items()]:
            prompt =  cls.AGENTS_DICT.get(agent_id, {})
            if prompt and kwargs:
                backstory = prompt.get("backstory", "")
                if backstory:
                    backstory = f"""{backstory}"""
                    prompt["backstory"] = backstory.format(**kwargs)
            return prompt
        elif agent_id in [task_id for task_id, _ in cls.TASK_PROMPTS.items()]:
            description = cls.TASK_PROMPTS.get(agent_id, "")["description"]

            if description and kwargs:
                try:
                    # Format the description string using the processed kwargs
                    description = description.format(**kwargs)
                except KeyError as e:
                    raise ValueError(f"Missing key in kwargs for formatting: {e}")
                except IndexError as e:
                    raise ValueError(f"Positional argument index out of range: {e}")
            return description

        filtered_prompts = []
        for _, value in unified_prompts.items():
            if value.get("type").value in [pt.value for pt in prompt_types]: 
                if agent_id in [classification_prompt for classification_prompt  in SystemAgentType]:
                    return value.get("text")
                
                filtered_prompts.append(value)
        return [SimpleNamespace(**prompt) for prompt in filtered_prompts]

