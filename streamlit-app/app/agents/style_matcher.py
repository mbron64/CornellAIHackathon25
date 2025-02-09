from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from .style_features import StyleFeatureExtractor
from typing import List, Dict

class StyleMatcher:
    def __init__(self, llm):
        self.llm = llm
        self.feature_extractor = StyleFeatureExtractor()
        
    def extract_style_from_samples(self, documents: List[str]) -> Dict:
        """Extract style features from sample documents"""
        all_features = []
        for doc in documents:
            features = self.feature_extractor.extract_features(doc)
            all_features.append(features)
        
        return self._aggregate_style_features(all_features)
    
    # ... rest of the implementation ... 