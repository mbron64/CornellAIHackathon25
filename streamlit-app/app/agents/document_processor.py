from .base_agent import BaseRAGAgent
from langchain.agents import Tool
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import AgentAction, AgentFinish
from typing import Union, List
from pathlib import Path

class DocumentProcessor(BaseRAGAgent):
    def __init__(self, llm, docs_path: Path):
        tools = [
            Tool(
                name="LoadDocuments",
                func=self._load_documents,
                description="Loads documents from specified path"
            ),
            Tool(
                name="SplitDocuments",
                func=self._split_documents,
                description="Splits documents into chunks"
            )
        ]
        super().__init__(llm=llm, tools=tools, docs_path=str(docs_path))

    def load_and_split(self):
        """Legacy method for compatibility"""
        documents = self._load_documents()
        return self._split_documents(documents)
    
    def aplan(
        self, 
        intermediate_steps: List[tuple], 
        **kwargs
    ) -> Union[AgentAction, AgentFinish]:
        if not intermediate_steps:
            return AgentAction(tool="LoadDocuments", tool_input={}, log="Loading documents")
            
        if len(intermediate_steps) == 1:
            documents = intermediate_steps[0][1]
            return AgentAction(tool="SplitDocuments", tool_input=documents, log="Splitting documents")
            
        return AgentFinish(return_values={"output": intermediate_steps[-1][1]}, log="Processing complete")
    
    def _load_documents(self):
        loader = DirectoryLoader(
            self.docs_path,
            glob="**/*.pdf",
            loader_cls=PyPDFLoader
        )
        return loader.load()
        
    def _split_documents(self, documents):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        return splitter.split_documents(documents) 