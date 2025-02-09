from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from .style_features import StyleFeatureExtractor

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
    
    def create_style_guided_prompt(self, style_features: Dict, content_prompt: str) -> str:
        template = """
        Write in the following specific style:
        
        Sentence Structure:
        - Use {avg_sentence_length} words per sentence on average
        - Prefer {primary_clause_pattern} clause structures
        - Follow these dependency patterns: {dep_patterns}
        
        Vocabulary and Tone:
        - Maintain a formality level of {formality_score}
        - Use transition phrases like: {transition_examples}
        - Match word sophistication level: {sophistication_level}
        
        Document Structure:
        - Use {paragraph_structure} paragraph organization
        - Include discourse markers: {discourse_markers}
        - Follow rhetorical pattern: {rhetorical_moves}
        
        Distinctive Style Elements:
        - Use hedging phrases: {hedging_examples}
        - Apply emphasis patterns: {emphasis_patterns}
        - Match pronoun usage style: {pronoun_patterns}
        
        Content Request: {content_prompt}
        
        Generate content that exactly matches both the style and addresses the content request.
        """
        
        return PromptTemplate(
            template=template,
            input_variables=["content_prompt"] + list(style_features.keys())
        ) 