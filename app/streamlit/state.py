import streamlit as st
import uuid

def init_state():
    if "initialized" not in st.session_state:
        st.session_state.initialized = False
    if "user" not in st.session_state:
        st.session_state.user = {
            "name": "",
            "principal": 0.0,
            "rate": 0.0,
            "tenure": 12,
            "current_month": 1,
            "sentiment": "neutral"
        }
    if "chat" not in st.session_state:
        st.session_state.chat = []
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if "history" not in st.session_state:
        st.session_state.history = []