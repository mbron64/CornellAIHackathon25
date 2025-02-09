print("🚀 Starting AI Style Learning & Writing Assistant")
import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv
from pathlib import Path
from agents.document_processor import DocumentProcessor
from agents.vector_store import VectorStoreManager
from agents.style_analyzer import StyleAnalyzer
from agents.text_humanizer import TextHumanizer
from agents.llm_manager import LLMManager
import traceback
from langchain.chains import LLMChain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.retrieval import create_retrieval_chain
from langgraph.graph import StateGraph, START, END
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from typing import List, Dict, Any, Sequence, Union, Optional
from langchain_core.documents import Document
import json
import pandas as pd
import time

# Load environment variables and secrets
def load_secrets():
    """Load secrets from Streamlit secrets or environment variables"""
    secrets = {}
    
    # First try loading from .env file
    load_dotenv()
    
    # Define required secrets
    required_secrets = {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "OPENAI_BASE_URL": os.getenv("OPENAI_BASE_URL", "https://api.ai.it.cornell.edu/"),
        "PINECONE_API_KEY": os.getenv("PINECONE_API_KEY"),
        "PINECONE_ENV": os.getenv("PINECONE_ENV", "us-east1-gcp"),
        "PINECONE_INDEX_NAME": os.getenv("PINECONE_INDEX_NAME", "cornell-ai-hackathon-2025"),
        "AI_HUMANIZER_EMAIL": os.getenv("AI_HUMANIZER_EMAIL"),
        "AI_HUMANIZER_PASSWORD": os.getenv("AI_HUMANIZER_PASSWORD")
    }
    
    # Try to get values from Streamlit secrets if available
    try:
        for key in required_secrets:
            # Try Streamlit secrets first, then fall back to environment variable
            secrets[key] = st.secrets.get(key, required_secrets[key])
    except Exception:
        # If Streamlit secrets fail, use the environment variables we loaded
        secrets = required_secrets
    
    # Set environment variables and validate
    missing_secrets = []
    for key, value in secrets.items():
        if value:
            os.environ[key] = value
        else:
            missing_secrets.append(key)
    
    if missing_secrets:
        st.error("⚠️ Missing required secrets:")
        for key in missing_secrets:
            st.error(f"- {key}")
        st.info("Please make sure these secrets are set in .streamlit/secrets.toml or as environment variables.")
        st.stop()

# Load secrets at startup
load_secrets()

# Display welcome message and app description
st.title("@lpha.mail")
st.markdown("""
Welcome to your personal writing style assistant! This AI system will:
- 📚 Learn from your sample documents
- 🎯 Analyze your unique writing style
- ✍️ Generate new content matching your style
""")

# Add document selection section
st.markdown("### 📄 Document Selection")

# Document selection and training initialization
col1, col2 = st.columns(2)
with col1:
    if st.button("🔄 Use Default Documents", type="secondary"):
        st.session_state.clear()
        st.session_state["use_default_docs"] = True
        st.session_state["documents_selected"] = True

with col2:
    if st.button("📚 Use Uploaded Documents", type="secondary"):
        st.session_state.clear()
        st.session_state["show_uploader"] = True

# Show file uploader only after clicking "Use Uploaded Documents"
if st.session_state.get("show_uploader"):
    uploaded_files = st.file_uploader(
        "📚 Upload Documents",
        accept_multiple_files=True,
        type=['pdf', 'txt', 'docx'],
        label_visibility="collapsed",
        key="document_uploader"
    )
    
    if uploaded_files:
        if not st.session_state.get("run_system"):
            st.success(f"✅ {len(uploaded_files)} document(s) uploaded successfully!")
        st.session_state.uploaded_documents = uploaded_files
        st.session_state["use_uploaded_docs"] = True
        st.session_state["documents_selected"] = True

# Add run system button
if st.session_state.get("documents_selected"):
    if st.button("🚀 Run", type="primary"):
        st.session_state["run_system"] = True
        st.rerun()

def create_document(content: Union[str, Dict, Any], metadata: Optional[Dict] = None) -> Document:
    """Create a Document object with proper error handling."""
    try:
        if isinstance(content, Document):
            return content
        elif isinstance(content, dict):
            page_content = content.get("page_content", content.get("content", str(content)))
            doc_metadata = content.get("metadata", metadata or {})
            return Document(page_content=page_content, metadata=doc_metadata)
        else:
            return Document(page_content=str(content), metadata=metadata or {})
    except Exception as e:
        raise ValueError(f"Failed to create document from content: {content}. Error: {str(e)}")

def format_docs(docs: Sequence[Any]) -> List[Document]:
    """Format documents while preserving Document structure."""
    try:
        # Convert all inputs to Documents and maintain the Document structure
        return [create_document(doc) for doc in docs]
    except Exception as e:
        raise ValueError(f"Document formatting failed: {str(e)}")

@st.cache_resource
def initialize_system():
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    def update_progress(text, value):
        progress_text.text(f"⏳ {text}")
        progress_bar.progress(value)

    try:
        update_progress("Checking environment variables...", 0.1)
        required_vars = ["OPENAI_API_KEY", "OPENAI_BASE_URL", "PINECONE_API_KEY", 
                        "AI_HUMANIZER_EMAIL", "AI_HUMANIZER_PASSWORD"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

        update_progress("Initializing document processor...", 0.2)
        docs_path = Path(__file__).parent.parent.parent / 'back-end' / 'email_generator' / 'samples'
        
        llm = ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            model="anthropic.claude-3-haiku"
        )
        doc_processor = DocumentProcessor(llm=llm, docs_path=docs_path)

        update_progress("Setting up vector store...", 0.3)
        vector_manager = VectorStoreManager(api_key=os.getenv("PINECONE_API_KEY"))
        index_name = "cornell-ai-hackathon-2025"
        vector_manager.create_index(index_name)

        update_progress("Initializing embeddings...", 0.4)
        embeddings = OpenAIEmbeddings(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
            model="amazon.titan-text-embeddings.v2"
        )

        update_progress("Loading and processing documents...", 0.5)
        # Check which documents to use
        if st.session_state.get("use_uploaded_docs", False) and "uploaded_documents" in st.session_state:
            update_progress("Processing uploaded documents...", 0.5)
            texts = doc_processor.process_uploaded_files(st.session_state.uploaded_documents)
            if not texts:
                st.error("No content could be extracted from uploaded documents.")
                st.stop()
        else:
            if not docs_path.exists():
                raise ValueError(f"Documents path not found: {docs_path}")
            texts = doc_processor.load_and_split()
            
        if not texts:
            raise ValueError("No documents found to process")

        update_progress("Creating vector store...", 0.6)
        vectorstore = vector_manager.create_vector_store(
            texts, embeddings, index_name, "email_samples"
        )
        
        update_progress("Initializing style analyzer...", 0.7)
        style_analyzer = StyleAnalyzer(llm)
        
        update_progress("Initializing text humanizer...", 0.75)
        text_humanizer = TextHumanizer()
        
        update_progress("Learning writing style from samples...", 0.8)
        style_patterns = style_analyzer.learn_style_from_samples([doc.page_content for doc in texts])
        
        update_progress("Creating style-aware prompt...", 0.9)
        style_prompt = style_analyzer.get_style_prompt()
        
        update_progress("Initializing QA chain...", 0.95)
        
        # Create the base chain for handling documents
        doc_chain = create_stuff_documents_chain(
            llm=llm,
            prompt=style_prompt,
            document_variable_name="context"
        )
        
        # Create the retrieval chain with proper input/output mapping
        retrieval_chain = (
            {
                "context": (
                    itemgetter("input") 
                    | vectorstore.as_retriever(
                        namespace="email_samples",
                        search_kwargs={"k": 5}
                    )
                    | RunnableLambda(format_docs)
                ),
                "query": itemgetter("input"),
                "style_guide": itemgetter("style_guide")
            }
            | doc_chain
        )

        # Create a wrapper function to include style guide in the chain
        def qa_chain_with_style(query: str):
            # Capture chain operation details
            chain_details = {
                "query": query,
                "timestamp": st.session_state.get("chain_calls", 0) + 1
            }
            st.session_state["chain_calls"] = chain_details["timestamp"]
            
            try:
                # Invoke chain with proper input structure and error handling
                response = retrieval_chain.invoke({
                    "input": query,
                    "style_guide": style_patterns["style_guide"]
                })
                
                # Handle both string and dictionary responses
                if isinstance(response, str):
                    answer = response
                    source_documents = []
                else:
                    answer = response.get("answer", response)
                    source_documents = response.get("source_documents", [])
                
                # Add response details
                chain_details.update({
                    "response_length": len(answer),
                    "num_source_docs": len(source_documents),
                })
                
                # Store chain operation details
                if "chain_history" not in st.session_state:
                    st.session_state.chain_history = []
                st.session_state.chain_history.append(chain_details)
                
                return {
                    "answer": answer,
                    "source_documents": source_documents
                }
                
            except Exception as e:
                st.error(f"Chain execution error: {str(e)}")
                st.error(f"Query: {query}")
                st.error(f"Chain details: {json.dumps(chain_details, indent=2)}")
                raise

        update_progress("✅ System initialized successfully!", 1.0)
        progress_text.empty()
        return qa_chain_with_style, style_analyzer, style_patterns, text_humanizer

    except Exception as e:
        error_msg = f"❌ Error during initialization: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        progress_text.error(error_msg)
        raise e

# Initialize system with error handling
if st.session_state.get("run_system"):
    try:
        qa_chain_with_style, style_analyzer, style_patterns, text_humanizer = initialize_system()
    except Exception as e:
        st.error("Failed to initialize system. Please check the error message above.")
        st.stop()

    # Initialize session state
    if "chain_calls" not in st.session_state:
        st.session_state.chain_calls = 0
    if "chain_history" not in st.session_state:
        st.session_state.chain_history = []

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display style guide
    with st.expander("View Learned Writing Style Guide"):
        st.markdown(style_patterns["style_guide"])

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("What would you like me to write about?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            try:
                # Show loading message while generating
                with st.spinner("✍️ Crafting in your style"):
                    # Generate response using the wrapped QA chain
                    response = qa_chain_with_style(prompt)
                    response_text = response["answer"]
                    source_docs = response.get("source_documents", [])
                
                # Automatically humanize the response
                with st.spinner("🎨 Making the text more natural..."):
                    response_text = text_humanizer.humanize(response_text)
                
                # Display the final response
                st.markdown(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
                
                # Add style analysis
                with st.expander("View Style Analysis"):
                    if source_docs:
                        # Calculate style similarity
                        similarity = style_analyzer.calculate_style_similarity(
                            response_text,
                            source_docs[0].page_content
                        )
                        st.markdown(f"### Style Similarity Score: {similarity:.2%}")
                    
                    # Show style dimensions with meaningful labels
                    st.markdown("""
                    ### Writing Style Analysis (PCA)
                    
                    This analysis breaks down the writing style from your sample documents into key characteristics:
                    - Each dimension represents a distinct aspect of the writing style
                    - The percentages show how important each aspect is to the overall style
                    - The chart shows how strongly each characteristic appears in the text
                    """)
                    
                    # Create a DataFrame with full labels, using newlines for wrapping
                    style_characteristics = {
                        0: "Academic Formality\n(word choice, technical terms)",
                        1: "Personal Voice\n(first-person, emotional tone)",
                        2: "Structural Complexity\n(sentence length, organization)",
                        3: "Argumentative Style\n(evidence use, logical flow)",
                        4: "Descriptive Detail\n(specificity, examples)",
                        5: "Narrative Flow\n(transitions, pacing)",
                        6: "Technical Depth\n(subject terminology)",
                        7: "Rhetorical Devices\n(metaphors, analogies)",
                        8: "Engagement Level\n(reader interaction)",
                        9: "Analytical Depth\n(critical analysis)"
                    }
                    
                    dimensions_df = pd.DataFrame(
                        style_patterns["style_dimensions"],
                        columns=[f"{style_characteristics[i]}\n({var:.1%})" 
                                for i, var in enumerate(style_patterns["style_variations"]["variance_explained"])]
                    )
                    
                    # Add index labels for text segments
                    dimensions_df.index = [f"Segment {i+1}" for i in range(len(dimensions_df))]
                    
                    # Display the chart with increased height and axis labels
                    st.markdown("#### Style Characteristics Across Text Segments")
                    chart = st.line_chart(
                        dimensions_df,
                        height=500,
                        use_container_width=True
                    )
                    
                    # Add axis labels explanation
                    st.caption("""
                    **X-axis**: Text segments from the analyzed documents
                    **Y-axis**: Strength of each style characteristic (higher values indicate stronger presence)
                    """)
                    
                    # Show detailed interpretations for top dimensions
                    st.markdown("### Top Style Characteristics Explained")
                    components = style_patterns["style_variations"]["principal_components"]
                    for i, component in enumerate(components[:5]):
                        var_explained = style_patterns["style_variations"]["variance_explained"][i]
                        st.markdown(f"""
                        **{style_characteristics[i].replace('\n', ' ')}** ({var_explained:.1%} of style variation):
                        - High values: More formal/complex/detailed writing in this aspect
                        - Low values: More casual/simple/direct writing in this aspect
                        - This characteristic helps match the sample documents' style in terms of {style_characteristics[i].lower().replace('\n', ' ')}
                        """)
                    
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
                st.error(traceback.format_exc())
else:
    if st.session_state.get("documents_selected"):
        st.info("Click 'Run' to start processing your selected documents 👆")
    else:
        st.info("Select a document source above to begin 👆")