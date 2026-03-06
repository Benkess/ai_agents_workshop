# Imports
from abc import ABC, abstractmethod
from typing_extensions import TypedDict
# from langgraph.graph import CompiledStateGraph


class Assistant(ABC):
    """Abstract base class for an assistant agent."""
    
    initial_state: TypedDict # Set in subclasses
    
    @abstractmethod
    def create_graph(self, checkpointer=None): # -> CompiledStateGraph:
        '''
        Create the LangGraph state graph.
        Returns:
            CompiledStateGraph: The compiled state graph for the assistant.
        '''
        pass # return CompiledStateGraph


    