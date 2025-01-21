from pathlib import Path
import streamlit as st
from langchain.document_loaders import SitemapLoader
from fake_useragent import UserAgent
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.storage import LocalFileStore
from bs4 import BeautifulSoup
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
import requests


choose_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Only use the pre-existing answers to address the user's query.

            Prioritize answers with the highest helpfulness score and favor the most recent responses.
            Cite sources as they are provided, without modification.
            When multiple answers share the same score, select the most informed one.
            Always respond directly based on the cited source.

            Answers: {answers}
            ---
            Examples:
                                                  
            The speed of light is 299,792,458 meters per second.

            Source: https://example.com
            """,
        ),
        ("human", "{question}"),
    ]
)


def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]

    choose_chain = choose_prompt | choose_answer_llm
    condensed = "\n\n".join(
        f"{answer['answer']} \nSource:{answer['source']} \nDate:{answer['date']} \n\n"
        for answer in answers
    )
    rtn = choose_chain({"question": question, "answers": condensed})
    return rtn

answers_prompt = ChatPromptTemplate.from_template(
    """
    Rely solely on the given context to answer the user's question. 
    
    If the context doesn't provide the necessary information, respond with "I don't know" and avoid making up any details.

    Always assign a score to the answer, ranging from 0 to 5:
        - A high score (e.g., 4-5) indicates the answer effectively addresses the question.
        - A low score (e.g., 0-2) indicates the answer fails to answer the question.

    Include the score in the response, even if it is 0.

    Context: {context}
                                                  
    Examples:
                                                  
    Question: What is the speed of light?
    Answer: The speed of light is 299,792,458 meters per second.
    Score: 5
                                                  
    Question: What is the speed of sound?
    Answer: I don't know
    Score: 0
                                                  
    Your turn!

    Question: {question}
"""
)
def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]

    answers_chain = answers_prompt | get_answer_llm
    return {
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {
                        "question": question,
                        "context": doc.page_content,
                    }
                ).content,
                "source": doc.metadata["source"],
                "date": doc.metadata["lastmod"],
            }
            for doc in docs
        ],
        "question": question,
    }

def parse_page(soup: BeautifulSoup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return str(soup.get_text()).replace("\n", "")

@st.cache_data(show_spinner="Loading Website...")
def load_website(url):
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000, chunk_overlap=200
    )
    loader = SitemapLoader(
        url,
        filter_urls=(
            [
                r"https:\/\/developers.cloudflare.com/ai-gateway.*",
                r"https:\/\/developers.cloudflare.com/vectorize.*",
                r"https:\/\/developers.cloudflare.com/workers-ai.*",
            ]
        ),
        parsing_function=parse_page,
    )
    ua = UserAgent()
    loader.headers = {"User-Agent": ua.random}
    loader.requests_kwargs = {"verify": False}
    docs = loader.load_and_split(splitter)

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    url_copy = url[:]
    cache_filename = url_copy.replace("/", "_")
    cache_filename.strip()
    cache_dir = LocalFileStore(f"./.cache/{cache_filename}/")
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings,
        cache_dir,
    )
    vector_store = FAISS.from_documents(docs, cached_embeddings)
    return vector_store.as_retriever()
    # r = requests.get(url, verify=False)
    # return r.text

class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)

if "messages" not in st.session_state:
    st.session_state["messages"] = []


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})

def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(message["message"], message["role"], save=False)
    
st.set_page_config(
    page_title="SiteGPT",
    page_icon="🌐",
)

with st.sidebar:
    openai_api_key = st.text_input("Input your OpenAI API Key")
    url = st.text_input(
        "Write down a URL",
        placeholder="https://example.com",
        value="https://developers.cloudflare.com/sitemap.xml",
        disabled=True,
    )
    st.markdown("---")
    st.write("Github: https://github.com/freddyhong/fullstack-gpt")

if not openai_api_key:
     st.markdown(
    """
    Ask questions about the content of a website.

    For now, this app only supports sitemaps and questions about the Clouldfare website"
            
    Start by writing your OpenAI API key in the sidebar.
"""
     )
     
if url:
    if ".xml" not in url:
        with st.sidebar:
            st.error("Please write down a Sitemap URL.")
    if not openai_api_key:
        st.error("Please input your OpenAI API Key on the sidebar")
    else:
        paint_history()
        get_answer_llm = ChatOpenAI(
            temperature=0.1,
            openai_api_key=openai_api_key,
        )
        choose_answer_llm = ChatOpenAI(
            temperature=0.1,
            openai_api_key=openai_api_key,
            streaming=True,
            callbacks=[ChatCallbackHandler()],
        )
        retriever = load_website(url)
        query = st.chat_input("Ask a question to the website.")
        if query:
            send_message(query, "human")
            chain = (
                {
                    "docs": retriever,
                    "question": RunnablePassthrough(),
                }
                | RunnableLambda(get_answers)
                | RunnableLambda(choose_answer)
            )

            with st.chat_message("ai"):
                chain.invoke(query)
