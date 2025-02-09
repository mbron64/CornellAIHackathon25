print("Running updated version")
import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv
from pathlib import Path
from agents.document_processor import DocumentProcessor
from agents.vector_store import VectorStoreManager
from agents.style_analyzer import StyleAnalyzer
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
from IPython.display import Image, display

# Load environment variables
load_dotenv()

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

st.title("AI Style Learning & Writing Assistant")

# Create detailed workflow visualization using LangGraph
def create_workflow_graph():
    # Define the state type
    class State(dict):
        messages: list
        style_guide: str
        context: str

    # Create the graph
    workflow = StateGraph(State)
    
    # Add nodes
    workflow.add_node("input", lambda x: {"messages": ["User Query"]})
    workflow.add_node("retriever", lambda x: {"messages": ["Document Retrieval"]})
    workflow.add_node("vectorstore", lambda x: {"messages": ["Vector Search"]})
    workflow.add_node("style_analyzer", lambda x: {"messages": ["Style Analysis"]})
    workflow.add_node("llm", lambda x: {"messages": ["Response Generation"]})
    workflow.add_node("style_guide_node", lambda x: {"messages": ["Style Guide Creation"]})
    workflow.add_node("style_prompt", lambda x: {"messages": ["Style-Aware Prompting"]})
    workflow.add_node("output", lambda x: {"messages": ["Style-Matched Response"]})
    
    # Add edges
    workflow.add_edge(START, "input")
    workflow.add_edge("input", "retriever")
    workflow.add_edge("retriever", "vectorstore")
    workflow.add_edge("vectorstore", "style_analyzer")
    workflow.add_edge("style_analyzer", "style_guide_node")
    workflow.add_edge("style_guide_node", "style_prompt")
    workflow.add_edge("style_prompt", "llm")
    workflow.add_edge("llm", "output")
    workflow.add_edge("output", END)
    
    return workflow.compile()

# Display workflow visualization
with st.expander("View LangChain Workflow", expanded=True):
    # Get the graph and convert to Mermaid syntax
    workflow = create_workflow_graph()
    mermaid_syntax = workflow.get_graph().draw_mermaid()
    
    # Generate and display the workflow image
    try:
        graph_image = workflow.get_graph().draw_mermaid_png()
        st.image(graph_image, caption="LangChain Workflow Visualization", use_column_width=True)
    except Exception as e:
        # Fallback to Mermaid syntax if image generation fails
        st.markdown(f"""
        ```mermaid
        {mermaid_syntax}
        ```
        """)
        st.warning("Fallback to Mermaid diagram due to image generation error.")
    
    st.markdown("""
    ### Workflow Components
    
    #### Document Processing
    - **Sample Documents**: Source materials for style learning
    - **Document Chunks**: Split documents for processing
    - **Titan Embeddings**: Convert text to vectors
    
    #### Style Learning
    - **Style Analyzer**: Extract writing patterns
    - **Style Guide**: Comprehensive style rules
    - **Style-Aware Prompt**: Template for response generation
    
    #### Generation
    - **Document Retriever**: Find relevant content
    - **Vector Store**: Semantic search database
    - **Claude 3 Haiku**: Generate styled responses
    """)

@st.cache_resource
def initialize_system():
    progress_text = st.empty()
    progress_bar = st.progress(0)
    
    def update_progress(text, value):
        progress_text.text(f"⏳ {text}")
        progress_bar.progress(value)

    try:
        update_progress("Checking environment variables...", 0.1)
        required_vars = ["OPENAI_API_KEY", "OPENAI_BASE_URL", "PINECONE_API_KEY"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

        update_progress("Initializing document processor...", 0.2)
        docs_path = Path(__file__).parent.parent.parent / 'back-end' / 'email_generator' / 'samples'
        if not docs_path.exists():
            raise ValueError(f"Documents path not found: {docs_path}")

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
        texts = doc_processor.load_and_split()
        if not texts:
            raise ValueError("No documents found to process")

        update_progress("Creating vector store...", 0.6)
        vectorstore = vector_manager.create_vector_store(
            texts, embeddings, index_name, "email_samples"
        )
        
        update_progress("Initializing style analyzer...", 0.7)
        style_analyzer = StyleAnalyzer(llm)
        
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
        return qa_chain_with_style, style_analyzer, style_patterns

    except Exception as e:
        error_msg = f"❌ Error during initialization: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        progress_text.error(error_msg)
        raise e

# Initialize system with error handling
try:
    with st.spinner('Initializing system...'):
        qa_chain_with_style, style_analyzer, style_patterns = initialize_system()
        st.success("✅ System initialized successfully!")
except Exception as e:
    st.error("Failed to initialize system. Please check the error message above.")
    st.stop()

# Initialize session state
if "chain_calls" not in st.session_state:
    st.session_state.chain_calls = 0
if "chain_history" not in st.session_state:
    st.session_state.chain_history = []

# Display system metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Queries", st.session_state.chain_calls)
with col2:
    avg_response_len = 0
    if st.session_state.chain_history:
        avg_response_len = sum(c["response_length"] for c in st.session_state.chain_history) / len(st.session_state.chain_history)
    st.metric("Avg Response Length", f"{avg_response_len:.0f} chars")
with col3:
    if st.session_state.chain_history:
        avg_docs = sum(c["num_source_docs"] for c in st.session_state.chain_history) / len(st.session_state.chain_history)
        st.metric("Avg Source Docs", f"{avg_docs:.1f}")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display style guide
with st.expander("View Learned Writing Style Guide"):
    st.markdown(style_patterns["style_guide"])

# Display chain history
with st.expander("View Chain Operation History"):
    if st.session_state.chain_history:
        st.json(st.session_state.chain_history)
    else:
        st.info("No chain operations recorded yet.")

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
            with st.spinner('Generating style-matched response...'):
                # Generate response using the wrapped QA chain
                response = qa_chain_with_style(prompt)
                response_text = response["answer"]
                source_docs = response.get("source_documents", [])
                
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
                    
                    # Display the chart with increased height for wrapped labels
                    st.line_chart(dimensions_df, height=500)
                    
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
                    
                    # Show chain operation details
                    st.markdown("### Chain Operation Details")
                    st.json(st.session_state.chain_history[-1])
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            st.error(traceback.format_exc())