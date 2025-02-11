from langchain_community.llms.bedrock import Bedrock
import os

class LLMManager:
    def __init__(self, region_name=None, aws_access_key_id=None, aws_secret_access_key=None):
        self.region_name = region_name or os.getenv("AWS_REGION")
        self.aws_access_key_id = aws_access_key_id or os.getenv("AWS_ACCESS_KEY_ID")
        self.aws_secret_access_key = aws_secret_access_key or os.getenv("AWS_SECRET_ACCESS_KEY")
        
    def get_llm(self, model_name="anthropic.claude-3-haiku-20240307-v1:0", temperature=0.7):
        return Bedrock(
            model_id=model_name,
            region_name=self.region_name,
            credentials_profile_name=None,
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            model_kwargs={
                "temperature": temperature,
                "max_tokens": 2048,
                "top_p": 0.9,
            }
        ) 