print("Running updated version")
import streamlit as st
from langchain_openai.chat_models import ChatOpenAI

st.title("VECTOR APP v2")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate and display assistant response
    if not openai_api_key.startswith("sk-"):
        st.warning("Please enter your OpenAI API key!", icon="⚠")
    else:
        with st.chat_message("assistant"):
            model = ChatOpenAI(temperature=0.7, api_key=openai_api_key)
            response = model.invoke(prompt)
            st.markdown(response.content)
            st.session_state.messages.append({"role": "assistant", "content": response.content})