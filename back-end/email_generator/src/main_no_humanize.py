import asyncio
import os
import google.generativeai as genai
from collections import Counter
import re
from PyPDF2 import PdfReader
from typing import List, Dict
from dotenv import load_dotenv

class EmailStylerSimple:
    def __init__(self):
        # Initialize Gemini
        load_dotenv()
        self.gemini_key = os.getenv('GEMINI_API_KEY')
        
        if not self.gemini_key:
            raise ValueError("Please set GEMINI_API_KEY in your .env file")
            
        genai.configure(api_key=self.gemini_key)
        self.model = genai.GenerativeModel('gemini-pro')

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
        """Generate an email matching the style from PDFs."""
        # 1. Extract text from PDFs
        text_content = self.extract_text_from_pdfs(pdf_paths)
        
        # 2. Analyze writing style
        style = self.analyze_writing_style(text_content)
        
        # Show style analysis
        print("\nAnalyzed Writing Style:")
        print(f"- Average sentence length: {style['avg_sentence_length']:.1f} words")
        print(f"- Common phrases: {', '.join(style['common_phrases'])}")
        print(f"- Typical greetings: {', '.join(style['greetings']) if style['greetings'] else 'Hi'}")
        print(f"- Typical closings: {', '.join(style['closings']) if style['closings'] else 'Best regards'}")
        
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
        return response.text

async def main():
    try:
        # Initialize
        styler = EmailStylerSimple()
        
        # Use absolute path to samples directory
        samples_dir = "/Users/desikao/Cornell_Hackathon/CornellAIHackathon25/back-end/email_generator/samples"
        
        # List available PDFs
        pdf_files = [
            os.path.join(samples_dir, f) 
            for f in os.listdir(samples_dir) 
            if f.endswith('.pdf')
        ]
        
        if not pdf_files:
            print("No PDF files found in the samples directory.")
            print(f"Directory contents: {os.listdir(samples_dir)}")
            return
        
        print(f"\nFound {len(pdf_files)} PDF files:")
        for pdf in pdf_files:
            print(f"- {os.path.basename(pdf)}")
        
        # Get user input
        print("\nWhat kind of email would you like to write?")
        print("Example: Write a professional email to schedule a team meeting")
        user_prompt = input("\nYour prompt: ").strip()
        
        if not user_prompt:
            print("Please provide a prompt for the email.")
            return
        
        print("\nAnalyzing your writing style and generating email...")
        
        # Generate email
        email = await styler.generate_email(pdf_files, user_prompt)
        
        print("\nGenerated Email:")
        print("-" * 60)
        print(email)
        print("-" * 60)
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        print("\nFull error trace:")
        print(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(main())