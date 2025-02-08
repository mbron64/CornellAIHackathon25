import asyncio
import os
from email_generator import EmailStyler

async def main():
    try:
        # Initialize
        styler = EmailStyler()
        
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
        
        print("\nGenerating email using your writing style from the essays...")
        print("This may take a moment...")
        
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