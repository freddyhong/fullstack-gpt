import streamlit as st

st.title("Title")

with st.sidebar:
    st.title("Sidebar Title")
    st.text_input("This is a sidebar")

tab1, tab2, tab3 = st.tabs(["A", "B", "C"])

with tab1:
    st.write("Content of tab 1")

with tab2:
    st.write("Content of tab 2")

with tab3:
    st.write("Content of tab 3")