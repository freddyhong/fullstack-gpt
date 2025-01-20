from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.storage import LocalFileStore
import os
st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓",
)

st.title("QuizGPT")

@st.cache_resource(show_spinner="Splitting file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )

    split_tup = os.path.splitext(file_path)
    file_extension = split_tup[1]
    if file_extension == '.pdf':
        loader = PyPDFLoader(file_path)
    elif file_extension == '.docx':
        loader = Docx2txtLoader(file_path)
    else:
        loader = TextLoader(file_path)

    docs = loader.load_and_split(text_splitter=splitter)
    return docs

with st.sidebar:
    choice = st.selectbox(
        "Choose what you want to use.",
        (
            "File",
            "Wikipedia Article",
        ),
    )
    if choice == "File":
        file = st.file_uploader("Upload a .docx, .pdf, or .txt file.", type=["docx", "pdf", "txt"])
        if file:
            docs = split_file(file)

        else:
            topic = st.text_input("Enter a topic to search for.")
            if topic:
                retriever = WikipediaRetriever(top_k_results=2)
                with st.spinner("Searching Wikipedia..."):
                    docs = retriever.get_relevant_documents(topic)