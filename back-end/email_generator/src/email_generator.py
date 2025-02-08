import os
import time
import requests
from PyPDF2 import PdfReader
import google.generativeai as genai
from collections import Counter
import re
from typing import List, Dict
from dotenv import load_dotenv

class EmailStyler:
    def __init__(self):
        # Initialize Gemini
        load_dotenv()
        self.gemini_key = os.getenv('GEMINI_API_KEY')
        self.humanize_key = os.getenv('HUMANIZE_API_KEY')
        
        if not self.gemini_key:
            raise ValueError("Please set GEMINI_API_KEY in your .env file")
        if not self.humanize_key:
            raise ValueError("Please set HUMANIZE_API_KEY in your .env file")
            
        genai.configure(api_key=self.gemini_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Humanize AI API endpoints
        self.humanize_api_url = "https://api.humanizeai.pro/v1"

    async def humanize_text(self, text: str) -> str:
        """Submit text to Humanize AI API and get humanized result."""
        headers = {
            "x-api-key": self.humanize_key,
            "Content-Type": "application/json"
        }
        
        # Submit humanization task
        submit_response = requests.post(
            self.humanize_api_url,
            headers=headers,
            json={"text": text}
        )
        
        if submit_response.status_code != 200:
            raise Exception(f"Humanization submission failed: {submit_response.text}")
        
        task_id = submit_response.json().get('id')
        
        # Poll for results
        max_attempts = 10
        attempt = 0
        while attempt < max_attempts:
            result_response = requests.get(
                f"{self.humanize_api_url}?id={task_id}",
                headers=headers
            )
            
            if result_response.status_code == 200:
                result = result_response.json()
                if 'humanized_text' in result:
                    return result['humanized_text']
            
            attempt += 1
            time.sleep(2)  # Wait 2 seconds between polling attempts
        
        raise Exception("Humanization timed out")

    def extract_text_from_pdfs(self, pdf_paths: List[str]) -> str:
        """Extract text content from multiple PDFs."""
        combined_text = ""
        for pdf_path in pdf_paths:
            try:
                reader = PdfReader(pdf_path)
                for page in reader.pages:
                    combined_text += page.extract_text() + "\n"
            except Exception as e:
                print(f"Error processing {pdf_path}: {e}")
        return combined_text

    def analyze_writing_style(self, text: str) -> Dict:
        """Extract key writing patterns from text."""
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        words = text.lower().split()
        
        # Get common phrases and patterns
        phrases = []
        for i in range(len(words)-2):
            phrases.append(' '.join(words[i:i+2]))
            phrases.append(' '.join(words[i:i+3]))
        
        return {
            "avg_sentence_length": sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 15,
            "common_phrases": [phrase for phrase, _ in Counter(phrases).most_common(5)],
            "greetings": self._extract_patterns(text, [
                r"^[Hh]i\b", r"^[Hh]ello\b", r"^[Dd]ear\b"
            ]),
            "closings": self._extract_patterns(text, [
                r"[Bb]est regards", r"[Cc]heers", r"[Ss]incerely"
            ])
        }

    def _extract_patterns(self, text: str, patterns: List[str]) -> List[str]:
        """Find matches for given patterns in text."""
        matches = []
        for line in text.split('\n'):
            line = line.strip()
            for pattern in patterns:
                if re.search(pattern, line):
                    matches.append(line)
                    break
        return list(set(matches))[:2]

    async def generate_email(self, pdf_paths: List[str], user_prompt: str) -> str:
        """Complete workflow: analyze style, generate email, and humanize."""
        # 1. Extract text from PDFs
        text_content = self.extract_text_from_pdfs(pdf_paths)
        
        # 2. Analyze writing style
        style = self.analyze_writing_style(text_content)
        
        # 3. Create prompt for Gemini
        prompt = f"""
        Write an email following this request: {user_prompt}

        Match this writing style exactly:
        - Use about {style['avg_sentence_length']} words per sentence
        - Include these types of phrases naturally: {', '.join(style['common_phrases'])}
        - Use greetings like: {', '.join(style['greetings']) if style['greetings'] else 'Hi'}
        - Use closings like: {', '.join(style['closings']) if style['closings'] else 'Best regards'}

        The email should sound completely natural and match the writing style of the provided samples.
        """
        
        # 4. Generate email with Gemini
        response = await self.model.generate_content_async(prompt)
        generated_email = response.text
        
        # 5. Humanize the output using Humanize AI API
        humanized_email = await self.humanize_text(generated_email)
        
        return humanized_email

