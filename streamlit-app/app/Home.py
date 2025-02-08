print("Running updated version")
import streamlit as st
from langchain_openai.chat_models import ChatOpenAI
import os
from dotenv import load_dotenv
from pathlib import Path

# Get the directory containing this file
current_dir = Path(__file__).parent
# Go up one level to the project root
project_root = current_dir.parent
# Load .env from project root
env_path = project_root / '.env'
print(f"Looking for .env at: {env_path}")
print(f"File exists: {env_path.exists()}")
load_dotenv(env_path)

# Load environment variables and print for debugging
print(f"API Key: {os.getenv('OPENAI_API_KEY')}")
print(f"Base URL: {os.getenv('OPENAI_BASE_URL')}")

st.title("VECTOR APP v3")

# Get API key and base URL from environment variables
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")

def generate_response(input_text):
    model = ChatOpenAI(
        temperature=0.7,
        api_key=api_key,
        base_url=base_url
    )
    st.info(model.invoke(input_text))

with st.form("my_form"):
    text = st.text_area(
        "Enter text:",
        "What are the three key pieces of advice for learning how to code?",
    )
    submitted = st.form_submit_button("Submit")
    if not api_key:
        st.warning("OpenAI API key not found in environment variables!", icon="⚠")
    if submitted and api_key:
        generate_response(text)