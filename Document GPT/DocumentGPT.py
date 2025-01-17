import streamlit as st
from langchain.prompts import PromptTemplate

# .write() is a magical function. It automatically writes the object in a nice format
st.write([1,2,3,4,5]) # this prints a list in cool format
st.write(PromptTemplate) # this prints the class definition and examples
