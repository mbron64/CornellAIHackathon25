from typing import List, Dict
import re
from collections import defaultdict

class StyleFeatureExtractor:
    def __init__(self):
        self.sentence_end = re.compile(r'[.!?]+')
        
    def extract_features(self, text: str) -> Dict:
        """Extract style features without spaCy dependency"""
        return {
            "syntactic_features": self._get_syntactic_patterns(text),
            "lexical_features": self._get_lexical_features(text),
            "structural_features": self._get_structural_features(text),
            "stylistic_markers": self._get_stylistic_markers(text)
        }
    
    def _get_syntactic_patterns(self, text: str) -> Dict:
        """Extract sentence structure patterns"""
        sentences = [s.strip() for s in self.sentence_end.split(text) if s.strip()]
        words_per_sentence = [len(s.split()) for s in sentences]
        
        return {
            "sentence_lengths": words_per_sentence,
            "avg_sentence_length": sum(words_per_sentence) / len(words_per_sentence) if words_per_sentence else 0,
            "sentence_complexity": self._estimate_sentence_complexity(sentences)
        }
    
    def _estimate_sentence_complexity(self, sentences: List[str]) -> float:
        """Estimate sentence complexity based on conjunctions and punctuation"""
        complexity_markers = ['and', 'but', 'or', 'because', 'although', 'however', 'therefore']
        scores = []
        for sentence in sentences:
            words = sentence.lower().split()
            score = 1.0
            score += sum(0.2 for word in words if word in complexity_markers)
            score += 0.1 * sentence.count(',')
            scores.append(score)
        return sum(scores) / len(scores) if scores else 1.0

    def _get_lexical_features(self, text: str) -> Dict:
        """Extract word choice and vocabulary patterns"""
        words = text.lower().split()
        return {
            "formality_score": self._measure_formality(text),
            "transition_phrases": self._get_transition_phrases(text),
            "vocabulary_sophistication": self._estimate_vocabulary_level(words)
        }

    def _measure_formality(self, text: str) -> float:
        """Measure text formality based on key markers"""
        formal_markers = ['therefore', 'thus', 'consequently', 'furthermore', 'moreover']
        informal_markers = ['like', 'you know', 'kind of', 'sort of', 'basically']
        
        text_lower = text.lower()
        formal_count = sum(text_lower.count(marker) for marker in formal_markers)
        informal_count = sum(text_lower.count(marker) for marker in informal_markers)
        
        total = formal_count + informal_count
        return formal_count / total if total > 0 else 0.5

    def _get_transition_phrases(self, text: str) -> List[str]:
        """Find transition phrases in text"""
        transitions = [
            'in addition', 'moreover', 'furthermore',
            'however', 'nevertheless', 'nonetheless',
            'therefore', 'thus', 'consequently',
            'for example', 'for instance',
            'in conclusion', 'to summarize'
        ]
        found = []
        text_lower = text.lower()
        for phrase in transitions:
            if phrase in text_lower:
                found.append(phrase)
        return found

    def _estimate_vocabulary_level(self, words: List[str]) -> float:
        """Estimate vocabulary sophistication"""
        sophisticated_words = set([
            'therefore', 'however', 'nevertheless', 'furthermore',
            'consequently', 'subsequently', 'accordingly', 'hence'
        ])
        return len([w for w in words if w in sophisticated_words]) / len(words) if words else 0

    def _get_structural_features(self, text: str) -> Dict:
        """Extract document-level structural patterns"""
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        return {
            "paragraph_lengths": [len(p.split()) for p in paragraphs],
            "discourse_markers": self._get_discourse_markers(text),
            "document_structure": self._analyze_document_structure(paragraphs)
        }

    def _get_discourse_markers(self, text: str) -> List[str]:
        """Find discourse markers"""
        markers = [
            'firstly', 'secondly', 'finally',
            'in other words', 'that is to say',
            'in particular', 'specifically',
            'in fact', 'actually', 'indeed',
            'in contrast', 'on the other hand'
        ]
        found = []
        text_lower = text.lower()
        for marker in markers:
            if marker in text_lower:
                found.append(marker)
        return found

    def _analyze_document_structure(self, paragraphs: List[str]) -> Dict:
        """Analyze document structure"""
        return {
            "num_paragraphs": len(paragraphs),
            "avg_paragraph_length": sum(len(p.split()) for p in paragraphs) / len(paragraphs) if paragraphs else 0
        }

    def _get_stylistic_markers(self, text: str) -> Dict:
        """Extract distinctive stylistic choices"""
        return {
            "hedging_phrases": self._find_hedging(text),
            "emphasis_patterns": self._find_emphasis(text),
            "pronouns": self._count_pronouns(text)
        }

    def _find_hedging(self, text: str) -> List[str]:
        """Find hedging phrases"""
        hedging = [
            'may', 'might', 'could', 'would',
            'seem', 'appear', 'suggest',
            'possibly', 'perhaps', 'probably',
            'generally', 'usually', 'often', 'sometimes'
        ]
        found = []
        text_lower = text.lower()
        for phrase in hedging:
            if phrase in text_lower:
                found.append(phrase)
        return found

    def _find_emphasis(self, text: str) -> List[str]:
        """Find emphasis patterns"""
        emphasis = [
            'clearly', 'obviously', 'certainly',
            'important', 'crucial', 'essential',
            'indeed', 'in fact', 'actually',
            'particularly', 'especially', 'notably'
        ]
        found = []
        text_lower = text.lower()
        for phrase in emphasis:
            if phrase in text_lower:
                found.append(phrase)
        return found

    def _count_pronouns(self, text: str) -> Dict[str, int]:
        """Count pronoun usage"""
        pronouns = {
            'first_person': ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours'],
            'second_person': ['you', 'your', 'yours'],
            'third_person': ['he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'they', 'them', 'their', 'theirs']
        }
        counts = defaultdict(int)
        words = text.lower().split()
        for category, prons in pronouns.items():
            counts[category] = sum(words.count(p) for p in prons)
        return dict(counts) 