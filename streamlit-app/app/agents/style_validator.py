from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.prompts import StringPromptTemplate, PromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from langchain.schema import AgentAction, AgentFinish
from langchain.chains import LLMChain
from langchain.agents import AgentOutputParser
from typing import List, Union, ClassVar
import re
from .style_matcher import StyleMatcher
from .style_analyzer import StyleAnalyzer

class StyleValidatorPrompt(StringPromptTemplate):
    template: ClassVar[str] = """Analyze the writing style match between a response and source documents.

Available tools:
{tools}

Response to analyze: {response}
Source documents: {source_docs}

Think through this step by step:
1. Analyze vocabulary and word choice
2. Compare sentence structures
3. Evaluate tone and formality
4. Calculate overall style similarity

Format your response as a detailed analysis with scores.

Available tools: {tool_names}
Action Input: Analyze the style match between the response and source documents.
"""
    
    input_variables: List[str] = ["tools", "response", "source_docs", "tool_names"]

    def format(self, **kwargs) -> str:
        kwargs["tool_names"] = ", ".join([tool.name for tool in kwargs["tools"]])
        return self.template.format(**kwargs)

class StyleValidatorOutputParser(AgentOutputParser):
    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        if "Style analysis complete" in text:
            return AgentFinish(
                return_values={"output": text},
                log=text
            )
        
        match = re.match(r"Action: (.*?)\nAction Input: (.*)", text, re.DOTALL)
        if not match:
            return AgentFinish(
                return_values={"output": text},
                log=text
            )
            
        action = match.group(1).strip()
        action_input = match.group(2).strip()
        
        return AgentAction(tool=action, tool_input=action_input, log=text)

class StyleValidator:
    def __init__(self, api_key, base_url):
        self.llm = ChatOpenAI(
            temperature=0,
            api_key=api_key,
            base_url=base_url,
            model="anthropic.claude-3-haiku"
        )
        self.style_analyzer = StyleAnalyzer(self.llm)
        
    def analyze_style_match(self, response: str, source_docs: List[str]) -> str:
        # Extract style patterns from source documents
        source_style = self.style_analyzer.extract_style_patterns(
            [doc.page_content for doc in source_docs]
        )
        
        # Analyze response using the same method
        response_style = self.style_analyzer.extract_style_patterns([response])
        
        # Compare styles using embeddings and LLM analysis
        comparison_prompt = PromptTemplate(
            template="""Compare the writing styles of the source documents and the response:

Source Style Analysis:
{source_style}

Response Style Analysis:
{response_style}

Provide a detailed comparison focusing on:
1. Style consistency
2. Voice and tone alignment
3. Sentence structure similarity
4. Word choice patterns
5. Overall authenticity

Score each aspect from 0-100 and explain your reasoning.
""",
            input_variables=["source_style", "response_style"]
        )
        
        comparison = self.llm(
            comparison_prompt.format(
                source_style=source_style["style_analyses"][0],
                response_style=response_style["style_analyses"][0]
            )
        )
        
        return comparison

    def _analyze_vocabulary(self, texts: List[str]) -> str:
        return "Vocabulary analysis completed"

    def _analyze_structure(self, texts: List[str]) -> str:
        return "Structure analysis completed"

    def _score_style_match(self, response: str, source_docs: List[str]) -> dict:
        return {"style_match_score": 0.85}

    def analyze_style_match(self, response, source_docs):
        # Add tool_names to the input
        tool_names = [tool.name for tool in self.tools]
        return self.agent_executor.run(
            response=response,
            source_docs=source_docs,
            tools=self.tools,
            tool_names=tool_names
        ) 