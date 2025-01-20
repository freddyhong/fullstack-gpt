from pathlib import Path
import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain.document_loaders import UnstructuredFileLoader
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
from langchain.text_splitter import CharacterTextSplitter
from pathlib import Path
import json
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
import os
st.set_page_config(
    page_title="QuizGPT",
    page_icon="❓",
)

function = {
    "name": "create_quiz",
    "description": "function that takes a list of questions and answers and returns a quiz",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                        },
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {
                                        "type": "string",
                                    },
                                    "correct": {
                                        "type": "boolean",
                                    },
                                },
                                "required": ["answer", "correct"],
                            },
                        },
                    },
                    "required": ["question", "answers"],
                },
            }
        },
        "required": ["questions"],
    },
}

st.title("QuizGPT")

@st.cache_resource(show_spinner="Splitting file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    Path("./.cache/quiz_files").mkdir(parents=True, exist_ok=True) # making the directory if it doesn't exist
    with open(file_path, "wb+") as f:
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

@st.cache_data(show_spinner="Making quiz...")
def run_quiz_chain(_docs, topic, q_number, level):
    chain = prompt | llm
    return chain.invoke({"context": _docs, "q_number": q_number, "level": level})


@st.cache_data(show_spinner="Making Wikipedia...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=2)
    docs = retriever.get_relevant_documents(term)
    return docs

def format_documents(docs):
    return "\n\n".join(document.page_content for document in docs)

prompt = PromptTemplate.from_template(
    """
    You are a helpful assistant that is role playing as a teacher.
                    
    Based ONLY on the following context make {q_number} questions to test the user's knowledge about the text.
    
    Each question should have 4 answers, three of them must be incorrect and one should be correct.

    The difficulty level of the problem is '{level}'.

    Context: {context}
"""
)

with st.sidebar:
    docs = None
    topic = None
    openai_api_key = st.text_input("Enter your OpenAI API Key")
    st.markdown("---")
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
            docs = wiki_search(topic)
    st.markdown("---")
    q_number = st.number_input("Number of questions", min_value=1, max_value=15, value=5)
    level = st.selectbox("Quiz Level", ("EASY", "HRAD"))
if not docs:
    st.markdown(
        """
    Welcome to QuizGPT.
                
    I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.
                
    Get started by uploading a file or searching on Wikipedia in the sidebar.
    """
    )
else:
    if not openai_api_key:
        st.error("Please input your OpenAI API Key on the sidebar")
    else:
        llm = ChatOpenAI(
        temperature=0.1,
        model="gpt-3.5-turbo-1106",
        streaming=True,
        callbacks=[StreamingStdOutCallbackHandler(),],
        openai_api_key=openai_api_key,
    ).bind(
        function_call={
                "name": "create_quiz",
            },
            functions=[
                function,
            ],
    )
    response = run_quiz_chain(docs, topic if topic else file.name, q_number, level)
    response = response.additional_kwargs["function_call"]["arguments"]

    with st.form("questions_form"):
            questions = json.loads(response)["questions"]
            question_count = len(questions)
            success_count = 0
            for idx, question in enumerate(questions):
                st.markdown(f'#### {idx+1}. {question["question"]}')
                value = st.radio(
                    "Select an option.",
                    [answer["answer"] for answer in question["answers"]],
                    index=None,
                )

                if {"answer": value, "correct": True} in question["answers"]:
                    st.success("Correct!")
                    success_count += 1
                elif value is not None:
                    st.error("Wrong!")
            if question_count == success_count:
                st.balloons()
            button = st.form_submit_button()
            if button:
                with st.expander("Results", expanded=True):
                    st.write("Your score is {}/{}.".format(success_count, question_count))
