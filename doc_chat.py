import streamlit as st
import os
from openai import OpenAI
from tempfile import NamedTemporaryFile
import uuid

# Set page configuration
st.set_page_config(page_title="Document Chat", layout="wide")

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "vector_store_id" not in st.session_state:
    st.session_state.vector_store_id = None
    
if "file_ids" not in st.session_state:
    st.session_state.file_ids = []

# Set OpenAI API key
api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
    client = OpenAI(api_key=api_key)
else:
    st.sidebar.warning("Please enter your OpenAI API key to continue")
    client = None

# App title and description
st.title("Document Chat")
st.write("Upload documents and chat with their content. The assistant will only answer questions based on the uploaded documents.")

# Function to upload file to OpenAI
def upload_file_to_openai(uploaded_file):
    with NamedTemporaryFile(delete=False, suffix=f"_{uploaded_file.name}") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
    
    try:
        with open(tmp_file_path, "rb") as file_content:
            response = client.files.create(file=file_content, purpose="assistants")
        os.remove(tmp_file_path)
        return response.id
    except Exception as e:
        st.error(f"Error uploading file to OpenAI: {e}")
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)
        return None

# Function to create vector store
def create_vector_store():
    try:
        vector_store = client.vector_stores.create(name=f"knowledge_base_{uuid.uuid4()}")
        return vector_store.id
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None

# Function to add file to vector store
def add_file_to_vector_store(vector_store_id, file_id):
    try:
        result = client.vector_stores.files.create(
            vector_store_id=vector_store_id, file_id=file_id
        )
        return result
    except Exception as e:
        st.error(f"Error adding file to vector store: {e}")
        return None

# Sidebar for file upload
with st.sidebar:
    st.header("Upload Documents")
    uploaded_files = st.file_uploader("Choose files", accept_multiple_files=True, type=["pdf", "docx", "txt", "csv"])
    
    if uploaded_files and client:
        if st.button("Process Documents"):
            # Create a new vector store
            with st.spinner("Creating knowledge base..."):
                vector_store_id = create_vector_store()
                
                if vector_store_id:
                    st.session_state.vector_store_id = vector_store_id
                    st.session_state.file_ids = []
                    
                    # Upload each file and add to vector store
                    for uploaded_file in uploaded_files:
                        with st.spinner(f"Processing {uploaded_file.name}..."):
                            file_id = upload_file_to_openai(uploaded_file)
                            if file_id:
                                st.session_state.file_ids.append(file_id)
                                result = add_file_to_vector_store(vector_store_id, file_id)
                                if result:
                                    st.success(f"Added {uploaded_file.name} to knowledge base")
                                else:
                                    st.error(f"Failed to add {uploaded_file.name} to knowledge base")
                    
                    # Clear chat history when new documents are uploaded
                    st.session_state.messages = []
                    st.success(f"Successfully processed {len(st.session_state.file_ids)} document(s)")
                    st.info("You can now start chatting with your documents!")

    # Display current document status
    if st.session_state.file_ids:
        st.subheader("Active Documents")
        st.write(f"Number of documents: {len(st.session_state.file_ids)}")
        
        if st.button("Clear All Documents"):
            st.session_state.vector_store_id = None
            st.session_state.file_ids = []
            st.session_state.messages = []
            st.success("All documents cleared")
            st.rerun()

# Display chat interface
st.header("Chat")

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Get user input
if prompt := st.chat_input("Ask a question about your documents"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message in chat
    with st.chat_message("user"):
        st.write(prompt)
    
    # Display assistant response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        if not client:
            message_placeholder.write("Please enter your OpenAI API key in the sidebar.")
        elif not st.session_state.vector_store_id:
            message_placeholder.write("Please upload and process documents first.")
        else:
            try:
                with st.spinner("Thinking..."):
                    # Query the model with file search tool
                    # Create messages array with system message
                    messages = [
                        {
                            "role": "system", 
                            "content": """You are a helpful assistant that only answers questions based on the 
                            documents provided. If the question cannot be answered using the documents or is outside 
                            their scope, respond with "I don't know" or "I cannot answer this question based on the 
                            documents provided." Do not use any knowledge outside of the provided documents."""
                        },
                        {"role": "user", "content": prompt}
                    ]
                    
                    response = client.responses.create(
                        model="gpt-4o-mini",
                        input=messages,
                        tools=[{
                            "type": "file_search", 
                            "vector_store_ids": [st.session_state.vector_store_id]
                        }],
                        # tool_choice="auto"
                    )
                    
                    # Display the response
                    assistant_response = response.output_text
                    message_placeholder.write(assistant_response)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
            
            except Exception as e:
                message_placeholder.error(f"Error: {str(e)}")

# Add some helpful information at the bottom of the page
st.divider()
st.caption("""
This app uses OpenAI's API to process documents and answer questions. 
Your documents are uploaded to OpenAI's servers for processing.
The assistant will only answer questions based on the content of the uploaded documents.
""")