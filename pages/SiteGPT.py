from pathlib import Path
import streamlit as st
from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import Html2TextTransformer
import os
st.set_page_config(
    page_title="SiteGPT",
    page_icon="🌐",
)

html2tex_transformer = Html2TextTransformer()

with st.sidebar:
    url = st.text_input("Enter URL", placeholder="https://example.com")

if url:
    loader = AsyncChromiumLoader([url])
    docs = loader.load()
    st.write(docs)
    st.write("---")
    transformed = html2tex_transformer.transform_documents(docs)
    st.write(transformed)