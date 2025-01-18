import streamlit as st
import time

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📄",
)
st.title("DocumentGPT Home")

if "messages" not in st.session_state:
    st.session_state["messages"] = []


def send_message(role, message, save=True):
    with st.chat_message(role):
        st.write(message)
    if save:
        st.session_state["messages"].append({"role": role, "message": message})

for message in st.session_state["messages"]:
    send_message(message["role"], message["message"], save=False)

# Example of using st.status
# with st.status("Embedding File...", expanded=True) as status:
#     time.sleep(2)
#     st.write("Getting the file")
#     time.sleep(2)
#     st.write("File embedded")
#     status.update(label="error", state="error")

message = st.chat_input("Ask AI a question")
if message:
    send_message("You", message)
    with st.spinner("Thinking..."):
        time.sleep(2)
    send_message("AI", "You said: " + message)