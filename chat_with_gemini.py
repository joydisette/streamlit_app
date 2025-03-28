import streamlit as st
from utils import provide_response, augmented_rag_response
import os
from dotenv import load_dotenv

load_dotenv()

project_id = os.getenv("PROJECT_ID")
model_location = os.getenv("MODEL_LOCATION")
model_id = os.getenv("MODEL_ID") 
datastore_id = os.getenv("DATASTORE_ID")
datastore_location = os.getenv("DATASTORE_LOCATION")
temperature = os.getenv("TEMPERATURE")

# Configure the page
st.set_page_config(
    page_title="Retirement Plan Assistant",
    page_icon="��",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# App title
st.title("Government and Public Employees Retirement Plan Assistant")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about retirement plans..."):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = provide_response(
                project_id=project_id,
                model_location=model_location,
                model_id=model_id,
                augmented_rag_response=augmented_rag_response,
                messages=st.session_state.messages,
                temperature=temperature,
                datastore_id=datastore_id,
                datastore_location=datastore_location
            )
            st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

# Add some CSS to improve the appearance
st.markdown("""
    <style>
    .stChat {
        padding: 20px;
    }
    .stChatMessage {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
    }
    </style>
    """, unsafe_allow_html=True) 