import streamlit as st
from app.agent.chat import chat_ai

def chat_ui():
    st.subheader("Chat")

    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input = st.chat_input("Type your message")

    if user_input:
        st.session_state.chat.append({"role": "user", "content": user_input})

        response = chat_ai(
            name=st.session_state.user["name"],
            rate=st.session_state.user["rate"],
            principal=st.session_state.user["principal"],
            tenure=st.session_state.user["tenure"],
            current_month=st.session_state.user["current_month"],
            sentiment=st.session_state.user["sentiment"],
            user_prompt=user_input,
            thread_id=st.session_state.thread_id,
            history=[],  # you can plug your system history later
        )

        st.session_state.chat.append({"role": "assistant", "content": response})

        st.rerun()