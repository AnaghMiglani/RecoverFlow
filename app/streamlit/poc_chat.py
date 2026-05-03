import streamlit as st
import uuid
from datetime import datetime, timedelta
from app.agent.chat import chat_ai
from app.tools.emi_calc.main import loan_summary
from app.agent.guardrail import guardrail_llm


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

    if "init_date" not in st.session_state:
        st.session_state.init_date = None

    if "last_emi_msg_month" not in st.session_state:
        st.session_state.last_emi_msg_month = None


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

            st.session_state.init_date = datetime.now()
            st.session_state.initialized = True
            st.rerun()


def simulation_panel():
    st.subheader("Simulation")
    u = st.session_state.user

    u["name"] = st.text_input("Name", u["name"])
    u["principal"] = st.number_input("Principal (₹)", value=u["principal"])
    u["rate"] = st.number_input("Rate (% per annum)", value=u["rate"])
    u["tenure"] = st.number_input("Tenure (months)", value=u["tenure"])

    new_month = st.number_input(
        "Current Month",
        min_value=u["current_month"],
        max_value=u["tenure"],
        value=min(u["current_month"], u["tenure"])
    )

    if new_month > u["current_month"]:
        diff = new_month - u["current_month"]
        st.session_state.init_date += timedelta(days=30 * diff)

    u["current_month"] = new_month

    st.write(f"Current Simulated Date: {st.session_state.init_date.strftime('%d-%m-%Y')}")

    u["sentiment"] = st.selectbox(
        "Sentiment",
        ["calm", "neutral", "agitated"],
        index=["calm", "neutral", "agitated"].index(u["sentiment"])
    )

    st.session_state.user = u


def send_auto_message():
    user = st.session_state.user

    data = loan_summary.invoke({
        "principal": user["principal"],
        "rate": user["rate"],
        "tenure": user["tenure"],
        "current_month": user["current_month"]
    })

    msg = f"Hello, your EMI for this month is ₹{data['emi']}. Please try to pay it. At minimum, ₹{data['minimum_payment']} will prevent your balance from increasing."

    st.session_state.chat.append({
        "role": "assistant",
        "content": msg,
        "type": "auto"
    })

    st.session_state.history.append({
        "time": user["current_month"],
        "system_message": msg
    })


def chat_ui():
    st.subheader("Chat")

    current_month = st.session_state.user["current_month"]

    if st.session_state.last_emi_msg_month != current_month:
        send_auto_message()
        st.session_state.last_emi_msg_month = current_month

    for msg in st.session_state.chat:
        if msg.get("type") == "auto":
            with st.chat_message("assistant"):
                st.markdown(
                    f"""
                    <div style="
                        background-color:#dcfce7;
                        color:#166534;
                        padding:10px;
                        border-radius:10px;
                        margin-bottom:5px;
                        font-weight:500;
                    ">
                        {msg['content']}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        else:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

    user_input = st.chat_input("Type your message")

    if user_input:
        safe = guardrail_llm(user_input)

        if safe == 0:
            st.session_state.chat.append({
                "role": "assistant",
                "content": "Your message violates policy. Please try again.",
                "type": "normal"
            })
            st.rerun()
            return

        st.session_state.chat.append({
            "role": "user",
            "content": user_input
        })

        response = chat_ai(
            name=st.session_state.user["name"],
            rate=st.session_state.user["rate"],
            principal=st.session_state.user["principal"],
            tenure=st.session_state.user["tenure"],
            current_month=st.session_state.user["current_month"],
            sentiment=st.session_state.user["sentiment"],
            user_prompt=user_input,
            thread_id=st.session_state.thread_id,
            history=st.session_state.history
        )

        st.session_state.chat.append({
            "role": "assistant",
            "content": response,
            "type": "normal"
        })

        st.rerun()


def main_layout():
    col1, col2 = st.columns([3, 1])
    with col2:
        simulation_panel()

    with col1:
        chat_ui()


def main():
    st.set_page_config(layout="wide")
    init_state()

    if not st.session_state.initialized:
        user_form()
    else:
        main_layout()


if __name__ == "__main__":
    main()