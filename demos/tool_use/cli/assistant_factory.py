# Import Project Modules
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
from assistant import Assistant

# Import different assistant agents
from open_ai_assistant import OpenAIAssistant
from qwen25vl_assistant import Qwen25VLAssistant
from qwen3_transformers_assistant import Qwen3TransformersAssistant
from qwen3vl_assistant import Qwen3VLAssistant

class AgentFactory:
    """
    Factory class to create assistant agents based on specified types.
    """

    """Agent Dict mapping agent type strings to their corresponding classes."""
    _agent_dict = {
        "openai": OpenAIAssistant,
        "qwen25vl": Qwen25VLAssistant,
        "qwen3vl_transformers": Qwen3TransformersAssistant,
        "qwen3vl": Qwen3VLAssistant,
    }

    @staticmethod
    def create_agent(agent_type: str, model_name: str = None) -> Assistant:
        """
        Create an assistant agent based on the specified type.
        Args:
            agent_type (str): The type of assistant agent to create ('openai', 'qwen25vl', 'qwen3vl').
            model_name (str, optional): The model name to use for Qwen agents. Defaults to None.
        Returns:
            Assistant: An instance of the specified assistant agent.
        """
        if agent_type.lower() == "openai":
            return OpenAIAssistant()
        elif agent_type.lower() == "qwen25vl":
            model_name = model_name or "Qwen/Qwen2.5-VL-3B-Instruct"
            return Qwen25VLAssistant(model_name=model_name)
        elif agent_type.lower() == "qwen3vl":
            model_name = model_name or "Qwen/Qwen3-VL-4B-Instruct"
            return Qwen3VLAssistant(model_name=model_name)
        elif agent_type.lower() == "qwen3vl_transformers":
            model_name = model_name or "Qwen/Qwen3-VL-4B-Instruct"
            return Qwen3TransformersAssistant(model_name=model_name)
        else:
            raise ValueError(f"Unsupported agent type: {agent_type}.")
        
    def get_supported_agents() -> list:
        """
        Get a list of supported agent types.
        Returns:
            List[str]: A list of supported agent type strings.
        """
        return list(AgentFactory._agent_dict.keys())
