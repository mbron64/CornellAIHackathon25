from typing import List, Dict
import spacy
import numpy as np
from collections import defaultdict
import re

class StyleFeatureExtractor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")
        
    def extract_features(self, text: str) -> Dict:
        doc = self.nlp(text)
        
        return {
            "syntactic_features": self._get_syntactic_patterns(doc),
            "lexical_features": self._get_lexical_features(doc),
            "structural_features": self._get_structural_features(doc),
            "stylistic_markers": self._get_stylistic_markers(doc)
        }
    
    def _get_syntactic_patterns(self, doc) -> Dict:
        """Extract sentence structure patterns"""
        patterns = {
            "sentence_lengths": [],
            "clause_patterns": defaultdict(int),
            "dependency_patterns": defaultdict(int)
        }
        
        for sent in doc.sents:
            # Sentence length
            patterns["sentence_lengths"].append(len([t for t in sent if not t.is_punct]))
            
            # Clause structure (e.g., main + subordinate)
            clause_structure = self._analyze_clause_structure(sent)
            patterns["clause_patterns"][clause_structure] += 1
            
            # Dependency patterns
            dep_pattern = self._get_dependency_pattern(sent)
            patterns["dependency_patterns"][dep_pattern] += 1
            
        return patterns
    
    def _get_lexical_features(self, doc) -> Dict:
        """Extract word choice and vocabulary patterns"""
        return {
            "formality_markers": self._measure_formality(doc),
            "transition_phrases": self._get_transition_phrases(doc),
            "word_sophistication": self._measure_word_sophistication(doc)
        }
    
    def _get_structural_features(self, doc) -> Dict:
        """Extract document-level structural patterns"""
        return {
            "paragraph_lengths": self._get_paragraph_lengths(doc),
            "discourse_markers": self._get_discourse_markers(doc),
            "rhetorical_moves": self._analyze_rhetorical_moves(doc)
        }
    
    def _get_stylistic_markers(self, doc) -> Dict:
        """Extract distinctive stylistic choices"""
        return {
            "hedging_phrases": self._find_hedging(doc),
            "emphasis_patterns": self._find_emphasis(doc),
            "personal_pronouns": self._analyze_pronoun_usage(doc)
        } 