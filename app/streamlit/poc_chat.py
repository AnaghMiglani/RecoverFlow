import streamlit as st
import uuid
from app.agent.chat import chat_ai


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


def user_form():
    with st.form("user_details"):
        name = st.text_input("Name")
        principal = st.number_input("Principal Amount (₹)", min_value=0.0, value=10000.0)
        rate = st.number_input("Interest Rate (% per annum)", min_value=0.0, value=12.0)
        tenure = st.number_input("Tenure (months)", min_value=1, value=12)
        current_month = st.number_input("Current Month", min_value=1, value=1)
        sentiment = st.selectbox("Sentiment", ["calm", "neutral", "agitated"])

        submitted = st.form_submit_button("Start Simulation")

        if submitted:
            st.session_state.user = {
                "name": name,
                "principal": principal,
                "rate": rate,
                "tenure": tenure,
                "current_month": current_month,
                "sentiment": sentiment
            }
            st.session_state.initialized = True
            st.rerun()


def simulation_panel():
    st.subheader("Simulation")
    u = st.session_state.user

    u["name"] = st.text_input("Name", u["name"])
    u["principal"] = st.number_input("Principal (₹)", value=u["principal"])
    u["rate"] = st.number_input("Rate (% per annum)", value=u["rate"])
    u["tenure"] = st.number_input("Tenure (months)", value=u["tenure"])
    u["current_month"] = st.number_input("Current Month", value=u["current_month"])
    u["sentiment"] = st.selectbox(
        "Sentiment",
        ["calm", "neutral", "agitated"],
        index=["calm", "neutral", "agitated"].index(u["sentiment"])
    )

    st.session_state.user = u


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
            thread_id=st.session_state.thread_id
        )

        st.session_state.chat.append({"role": "assistant", "content": response})

        st.rerun()


def main_layout():
    col1, col2 = st.columns([3, 1])

    with col1:
        chat_ui()

    with col2:
        simulation_panel()


def main():
    st.set_page_config(layout="wide")
    init_state()

    if not st.session_state.initialized:
        user_form()
    else:
        main_layout()


if __name__ == "__main__":
    main()