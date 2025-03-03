import os
import tempfile
import streamlit as st
from streamlit_chat import message
from youtubequery import YoutubeQuery

st.set_page_config(page_title="Youtube to Chatbot (Mistral)")

def display_messages():
    st.subheader("Chat")
    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        message(msg, is_user=is_user, key=str(i))
    st.session_state["thinking_spinner"] = st.empty()

def process_input():
    if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:
        user_text = st.session_state["user_input"].strip()
        with st.session_state["thinking_spinner"], st.spinner(f"Thinking"):
            query_text = st.session_state["youtubequery"].ask(user_text)

        st.session_state["messages"].append((user_text, True))
        st.session_state["messages"].append((query_text, False))
        
def ingest_input():
    if st.session_state["input_url"] and len(st.session_state["input_url"].strip()) > 0:
        url = st.session_state["input_url"].strip()
        with st.session_state["thinking_spinner"], st.spinner(f"Thinking"):
            ingest_text = st.session_state["youtubequery"].ingest(url)        

def is_huggingface_api_key_set() -> bool:
    return len(st.session_state["HUGGINGFACE_API_TOKEN"]) > 0

def main():
    if len(st.session_state) == 0:
        st.session_state["messages"] = []
        st.session_state["url"] = ""
        st.session_state["HUGGINGFACE_API_TOKEN"] = os.environ.get("HUGGINGFACE_API_TOKEN", "")
        if is_huggingface_api_key_set():
            st.session_state["youtubequery"] = YoutubeQuery(st.session_state["HUGGINGFACE_API_TOKEN"])
        else:
            st.session_state["youtubequery"] = None

    st.header("Youtube to Chatbot (Powered by Mistral)")

    if st.text_input("HuggingFace API Token", value=st.session_state["HUGGINGFACE_API_TOKEN"], key="input_HUGGINGFACE_API_TOKEN", type="password"):
        if (
            len(st.session_state["input_HUGGINGFACE_API_TOKEN"]) > 0
            and st.session_state["input_HUGGINGFACE_API_TOKEN"] != st.session_state["HUGGINGFACE_API_TOKEN"]
        ):
            st.session_state["HUGGINGFACE_API_TOKEN"] = st.session_state["input_HUGGINGFACE_API_TOKEN"]
            st.session_state["messages"] = []
            st.session_state["user_input"] = ""
            st.session_state["input_url"] = ""
            st.session_state["youtubequery"] = YoutubeQuery(st.session_state["HUGGINGFACE_API_TOKEN"])

    st.subheader("Add a url")
    st.text_input("Input url", value=st.session_state["url"], key="input_url", disabled=not is_huggingface_api_key_set(), on_change=ingest_input)

    st.session_state["ingestion_spinner"] = st.empty()

    display_messages()
    st.text_input("Message", key="user_input", disabled=not is_huggingface_api_key_set(), on_change=process_input)

    st.divider()
    st.markdown("Source code: [Github](https://github.com/Anil-matcha/Chat-Youtube)")



if __name__ == "__main__":
    main()