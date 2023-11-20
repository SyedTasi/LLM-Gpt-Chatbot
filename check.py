import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
import fitz  # PyMuPDF
import pickle

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_document = fitz.open(pdf)
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                text += page.get_text()
        except Exception as e:
            # Handle the exception (e.g., print an error message)
            print(f"Error extracting text from PDF: {str(e)}")
            # Optionally, you can return an error message to the user
            return "Error extracting text from PDF."

    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


# Constants
VECTOR_STORE_FILE = "finance_report_vectorstore"

# def preprocess_finance_report():
#     # Check if the vector store file exists
#     if os.path.isfile(VECTOR_STORE_FILE):
#         st.session_state.vectorstore = FAISS.load(VECTOR_STORE_FILE)
#         return

#     # Process and create vector store for the finance report PDF
#     pdf_path = "C:/Users/syedt/OneDrive/Documents/project/LLM-Chatbot/BMW-AG-Financial-Statements-2022-en.pdf"  # Replace with the actual path to your finance report PDF
#     pdf_text = get_pdf_text([pdf_path])
#     text_chunks = get_text_chunks(pdf_text)
#     vectorstore = get_vectorstore(text_chunks)

#     # Save the vector store to a file for future use
#     #vectorstore.save(VECTOR_STORE_FILE)
#     st.session_state.vectorstore = vectorstore

def preprocess_finance_report():
    # Check if the vector store file exists
    if os.path.isfile(VECTOR_STORE_FILE):
        # Load the vector store from the file
        with open(VECTOR_STORE_FILE, 'rb') as file:
            st.session_state.vectorstore = pickle.load(file)
        return

    # Process and create vector store for the finance report PDF
    #pdf_path = "C:/Users/syedt/OneDrive/Documents/project/LLM-Chatbot/BMW-AG-Financial-Statements-2022-en.pdf"   # Replace with the actual path to your finance report PDF
    # pdf_path = "C:/Users/syedt/OneDrive/Documents/Internship Documents/Syed, Tasleem Offer Letter Aug-Dec.pdf"   # Replace with the actual path to your finance report PDF
    # pdf_text = get_pdf_text([pdf_path])
    # text_chunks = get_text_chunks(pdf_text)
    # vectorstore = get_vectorstore(text_chunks)

    # # Save the vector store to a file for future use
    # with open(VECTOR_STORE_FILE, 'wb') as file:
    #     pickle.dump(vectorstore, file)
    
    # st.session_state.vectorstore = vectorstore


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

# def main():
#     load_dotenv()
#     st.set_page_config(page_title="Chat with Finance Report",
#                        page_icon=":books:")
#     st.write(css, unsafe_allow_html=True)

#     if "conversation" not in st.session_state:
#         st.session_state.conversation = None
#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = None

#     st.header("Chat with Finance Report :books:")
#     user_question = st.text_input("Ask a question about the finance report:")
#     if user_question:
#         handle_userinput(user_question)

#     with st.sidebar:
#         st.subheader("Finance Report")
#         preprocess_finance_report()  # Load or preprocess the finance report


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with Finance Report",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None  # Initialize as None or create as needed
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with Finance Report :books:")
    user_question = st.text_input("Ask a question about the finance report:")
    if user_question:
        # Create or retrieve the conversation when needed
        if st.session_state.conversation is None:
            st.session_state.conversation = get_conversation_chain(st.session_state.vectorstore)
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Finance Report")
        preprocess_finance_report()  # Load or preprocess the finance report

if __name__ == '__main__':
    main()
