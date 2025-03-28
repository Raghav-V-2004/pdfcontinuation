import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import tempfile

# Load environment variables
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')

# Streamlit Page Configuration and Custom CSS for Styling
st.set_page_config(page_title="AI Chatbot", layout="wide")
st.markdown(
    """
    <style>
        body {
            background-color: #f9f9f9;
            color: #333;
        }
        .chat-bubble {
            padding: 10px 15px;
            margin: 10px 0;
            border-radius: 20px;
            max-width: 80%;
        }
        .user-bubble {
            background-color: #2E8B57;
            color: white;
            text-align: right;
            margin-left: auto;
        }
        .ai-bubble {
            background-color: #e0e0e0;
            color: #333;
        }
        .chat-box {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
            max-height: 400px;
            overflow-y: auto;
        }
        h2 {
            text-align: center;
            color: #0078D4;
        }
        .submit-button {
            background-color: #0078D4;
            color: white;
            border-radius: 20px;
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
        }
        .submit-button:hover {
            background-color: #0056A6;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize ChatGroq model
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

# Define the LLM prompt template
prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Questions: {input}
    """
)

# Check if the response is irrelevant
def is_out_of_context(answer):
    out_of_context_keywords = ["I don’t know", "not sure", "out of context", "invalid", "There is no mention", "no mention"]
    return any(keyword in answer.lower() for keyword in out_of_context_keywords)

# Create vector database from uploaded PDF
def create_vector_db_out_of_the_uploaded_pdf_file(pdf_file):
    if "vector_store" not in st.session_state:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(pdf_file.read())
            pdf_file_path = temp_file.name

        st.session_state.embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-small-en-v1.5', model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': True})
        
        st.session_state.loader = PyPDFLoader(pdf_file_path)
        st.session_state.text_document_from_pdf = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        
        st.session_state.final_document_chunks = st.session_state.text_splitter.split_documents(st.session_state.text_document_from_pdf)
        st.session_state.vector_store = FAISS.from_documents(st.session_state.final_document_chunks, st.session_state.embeddings)

# Handle PDF upload
pdf_input_from_user = st.file_uploader("Please Upload your PDF file", type=['pdf'])

if pdf_input_from_user is not None:
    if st.button("Create Vector Database", key="vector_btn"):
        create_vector_db_out_of_the_uploaded_pdf_file(pdf_input_from_user)
        st.success("PDF file is processed and vector database created!")

# Initialize chat history in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # List of (user_input, ai_response)

# Chat Interface - Main Section
if "vector_store" in st.session_state:
    st.markdown("<h2>Chat with the PDF</h2>", unsafe_allow_html=True)

    # Display chat history in a styled chat box
    chat_container = st.container()
    with chat_container:
        st.markdown("<div class='chat-box'>", unsafe_allow_html=True)
        for user_input, ai_response in st.session_state.chat_history:
            st.markdown(f"<div class='chat-bubble user-bubble'>{user_input}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='chat-bubble ai-bubble'>{ai_response}</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Input box for user question
    user_prompt = st.text_input("Enter Your Question for the uploaded PDF")

    if st.button('Submit Prompt', key="submit_btn"):
        if user_prompt:
            document_chain = create_stuff_documents_chain(llm, prompt)
            retriever = st.session_state.vector_store.as_retriever()
            retrieval_chain = create_retrieval_chain(retriever, document_chain)

            response = retrieval_chain.invoke({'input': user_prompt})

            if is_out_of_context(response['answer']):
                ai_response = "Sorry, I didn’t understand your question. Do you want to connect with a live agent?"
            else:
                ai_response = response['answer']

            # Update session state with the latest chat interaction
            st.session_state.chat_history.append((user_prompt, ai_response))

            # Re-render the page to display updated chat history
            st.experimental_rerun()
        else:
            st.error('Please write your prompt.')
