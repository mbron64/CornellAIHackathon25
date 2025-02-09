import requests
import os
import streamlit as st

class TextHumanizer:
    def __init__(self):
        self.api_url = "https://ai-text-humanizer.com/api.php"
        self.email = os.getenv("AI_HUMANIZER_EMAIL")
        self.password = os.getenv("AI_HUMANIZER_PASSWORD")
        self.max_chunk_size = 2000  # Maximum size for API request
        self.min_chunk_size = 500   # Minimum size for each chunk
        
        if not self.email or not self.password:
            raise ValueError("AI_HUMANIZER_EMAIL and AI_HUMANIZER_PASSWORD environment variables must be set")
    
    def _split_text(self, text: str) -> list[str]:
        """Split text into chunks at sentence boundaries, ensuring minimum chunk size."""
        if len(text) < self.min_chunk_size:
            return [text]
            
        chunks = []
        sentences = text.replace("\n", " ").split(". ")
        current_chunk = ""
        
        for sentence in sentences:
            if not sentence.strip():
                continue
            
            # Add period back if it was removed by split
            sentence = sentence + "." if not sentence.endswith(".") else sentence
            
            # Always add sentence to current chunk first
            new_chunk = current_chunk + (" " + sentence if current_chunk else sentence)
            
            # If adding this sentence exceeds max size, store current chunk and start new one
            if len(new_chunk) > self.max_chunk_size and len(current_chunk) >= self.min_chunk_size:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
            else:
                # Keep building current chunk
                current_chunk = new_chunk
        
        # Handle the last chunk
        if current_chunk:
            if len(current_chunk) >= self.min_chunk_size:
                chunks.append(current_chunk.strip())
            elif chunks:
                # If last chunk is too small, append to previous chunk
                chunks[-1] = chunks[-1] + " " + current_chunk
            else:
                # If it's the only chunk, keep it
                chunks.append(current_chunk.strip())
        
        return chunks
    
    def _humanize_chunk(self, chunk: str) -> str:
        """Humanize a single chunk of text."""
        try:
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
            
            if not response.text or "out of credits" in response.text.lower():
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
            # If text is shorter than minimum chunk size, process it as is
            if len(text) < self.min_chunk_size:
                return self._humanize_chunk(text)
            
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
            
            return result
            
        except Exception as e:
            return text 