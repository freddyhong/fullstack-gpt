import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.document_loaders import TextLoader

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📄",
)

if "messages" not in st.session_state:
    st.session_state["messages"] = []

@st.cache_resource(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")

    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = TextLoader(file_path)

    docs = loader.load_and_split(text_splitter=splitter)

    embeddings = OpenAIEmbeddings()

    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)

    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    retriever = vectorstore.as_retriever()
    return retriever

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        st.session_state["messages"].append({"message": message, "role": role})

def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)

st.title("DocumentGPT Home")

st.markdown(
    """
    Welcome to DocumentGPT!
    Use this chatbot to ask questions about your file!
    Upload your file on the sidebar to get started.
"""
)
with st.sidebar:
    file = st.file_uploader("Upload a .docx .pdf or .txt file", type=["pdf", "txt", "docx"])
if file:
    retriever = embed_file(file)
    send_message("File uploaded! Ask any questions!", "ai", save=False)
    paint_history()
    message = st.chat_input("Ask anything about the file")
    if message:
        send_message(message, "human")
else:
    st.session_state["messages"] = []