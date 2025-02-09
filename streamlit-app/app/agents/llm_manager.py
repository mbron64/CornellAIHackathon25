from langchain_openai.chat_models import ChatOpenAI

class LLMManager:
    def __init__(self, api_key, base_url):
        self.api_key = api_key
        self.base_url = base_url
        
    def get_llm(self, model_name="anthropic.claude-3-haiku", temperature=0.7):
        return ChatOpenAI(
            temperature=temperature,
            api_key=self.api_key,
            base_url=self.base_url,
            model=model_name
        ) 