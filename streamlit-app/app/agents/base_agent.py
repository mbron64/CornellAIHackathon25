from langchain.agents import BaseSingleActionAgent
from langchain.schema import AgentAction, AgentFinish
from typing import List, Union
from pydantic import BaseModel

class BaseRAGAgent(BaseSingleActionAgent, BaseModel):
    llm: object
    tools: List[object]
    docs_path: str = None  # Add this to allow docs_path

    class Config:
        arbitrary_types_allowed = True

    def plan(
        self, 
        intermediate_steps: List[tuple], 
        **kwargs
    ) -> Union[AgentAction, AgentFinish]:
        # Implement planning logic
        pass
        
    @property
    def input_keys(self):
        return ["input"] 