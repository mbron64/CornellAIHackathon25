import requests
import os
import streamlit as st

class TextHumanizer:
    def __init__(self):
        self.api_url = "https://ai-text-humanizer.com/api.php"
        self.email = os.getenv("AI_HUMANIZER_EMAIL")
        self.password = os.getenv("AI_HUMANIZER_PASSWORD")
        self.max_chunk_size = 2000  # Maximum size for API request
        self.min_chunk_size = 1000  # Minimum size for splitting
        
        if not self.email or not self.password:
            raise ValueError("AI_HUMANIZER_EMAIL and AI_HUMANIZER_PASSWORD environment variables must be set")
    
    def _split_text(self, text: str) -> list[str]:
        """Split text into chunks, preserving context and meaning.
        
        For texts under 1000 characters, returns the entire text as a single chunk.
        For longer texts, splits into chunks of at least 500 words at sentence boundaries.
        """
        # If text is under min_chunk_size, return as single chunk
        if len(text) < self.min_chunk_size:
            return [text]
            
        chunks = []
        sentences = text.replace("\n", " ").split(". ")
        current_chunk = ""
        current_word_count = 0
        min_words_per_chunk = 500  # Minimum words per chunk for longer texts
        
        for sentence in sentences:
            if not sentence.strip():
                continue
            
            # Add period back if it was removed by split
            sentence = sentence + "." if not sentence.endswith(".") else sentence
            sentence_word_count = len(sentence.split())
            
            # Always add sentence to current chunk first
            new_chunk = current_chunk + (" " + sentence if current_chunk else sentence)
            new_word_count = current_word_count + sentence_word_count
            
            # If adding this sentence exceeds max size AND we have enough words, store current chunk
            if len(new_chunk) > self.max_chunk_size and current_word_count >= min_words_per_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
                current_word_count = sentence_word_count
            else:
                # Keep building current chunk
                current_chunk = new_chunk
                current_word_count = new_word_count
        
        # Handle the last chunk
        if current_chunk:
            if chunks and current_word_count < min_words_per_chunk:
                # If last chunk is too small, append to previous chunk
                chunks[-1] = chunks[-1] + " " + current_chunk
            else:
                chunks.append(current_chunk.strip())
        
        return chunks
    
    def _humanize_chunk(self, chunk: str) -> str:
        """Humanize a single chunk of text."""
        try:
            # If text is too short, return it as is
            if len(chunk.strip()) < 50:  # Don't process very short texts
                return chunk
                
            payload = {
                'email': self.email,
                'pw': self.password,
                'text': chunk.strip()
            }
            
            response = requests.post(
                self.api_url,
                data=payload,
                timeout=60
            )
            
            response.raise_for_status()
            
            # If response contains error message or is empty, return original text
            if not response.text or \
               "out of credits" in response.text.lower() or \
               "cannot make any improvements" in response.text.lower() or \
               "need more text" in response.text.lower() or \
               response.text.startswith("[Note:") or \
               len(response.text) < len(chunk) * 0.5:  # Response suspiciously short
                return chunk
            
            return response.text
            
        except Exception as e:
            return chunk
    
    def humanize(self, text: str) -> str:
        """
        Process the text through the AI-Text-Humanizer API to make it more natural.
        Handles long texts by processing in chunks of at least 500 characters.
        
        Args:
            text (str): The AI-generated text to humanize
            
        Returns:
            str: The humanized text
        """
        try:
            # If text is too short, return it as is
            if len(text.strip()) < 50:  # Don't process very short texts
                return text
            
            # Split text into paragraphs
            paragraphs = text.split("\n")
            humanized_paragraphs = []
            
            for paragraph in paragraphs:
                if not paragraph.strip():
                    humanized_paragraphs.append("")
                    continue
                
                # Split paragraph into chunks of at least min_chunk_size
                chunks = self._split_text(paragraph)
                humanized_chunks = []
                
                for chunk in chunks:
                    humanized_chunk = self._humanize_chunk(chunk)
                    if humanized_chunk:
                        humanized_chunks.append(humanized_chunk)
                
                # Combine chunks back into paragraph
                humanized_paragraph = " ".join(humanized_chunks)
                humanized_paragraphs.append(humanized_paragraph)
            
            # Combine paragraphs with original formatting
            result = "\n".join(humanized_paragraphs)
            
            # Final check - if result looks like an error message, return original
            if result.startswith("[Note:") or \
               "cannot make any improvements" in result.lower() or \
               "need more text" in result.lower():
                return text
            
            return result
            
        except Exception as e:
            return text 