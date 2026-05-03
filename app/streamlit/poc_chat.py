import streamlit as st
import uuid
from datetime import datetime, timedelta
from app.agent.chat import chat_ai
from app.tools.emi_calc.main import loan_summary
from app.agent.guardrail import guardrail_llm
from app.agent.sentiment import sentiment_analysis


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

    if "paid_this_month" not in st.session_state:
        st.session_state.paid_this_month = False


def user_form():
    with st.form("user_details"):
        name = st.text_input("Name")
        principal = st.number_input("Principal Amount (₹)", min_value=0.0, value=10000.0)
        rate = st.number_input("Interest Rate (% per annum)", min_value=0.0, value=12.0)
        tenure = st.number_input("Tenure (months)", min_value=1, value=12)
        current_month = st.number_input("Current Month", min_value=1, value=1)

        submitted = st.form_submit_button("Start Simulation")

        if submitted:
            st.session_state.user = {
                "name": name,
                "principal": principal,
                "rate": rate,
                "tenure": tenure,
                "current_month": current_month,
                "sentiment": "neutral"
            }

            st.session_state.init_date = datetime.now()
            st.session_state.initialized = True
            st.rerun()


def compute_interest(principal, rate):
    return principal * (rate / 12 / 100)


def make_payment(amount):
    user = st.session_state.user

    interest = compute_interest(user["principal"], user["rate"])
    total_due = user["principal"] + interest

    if amount > total_due:
        return False, f"Payment cannot exceed ₹{round(total_due,2)}"

    new_principal = user["principal"] + interest - amount
    user["principal"] = max(new_principal, 0)

    st.session_state.chat.append({
        "role": "assistant",
        "content": f"Payment of ₹{amount} made. New principal is ₹{round(user['principal'],2)}.",
        "type": "payment_success"
    })

    st.session_state.paid_this_month = True

    if user["current_month"] < user["tenure"]:
        user["current_month"] += 1
        st.session_state.init_date += timedelta(days=30)

    return True, ""


def simulation_panel():
    st.subheader("Simulation")
    u = st.session_state.user

    u["name"] = st.text_input("Name", u["name"])

    st.text_input("Principal (₹)", value=u["principal"], disabled=True)
    st.text_input("Rate (% per annum)", value=u["rate"], disabled=True)
    st.text_input("Tenure (months)", value=u["tenure"], disabled=True)

    st.write(f"Current Month: {u['current_month']}")

    if st.button("Next Month"):
        if u["current_month"] < u["tenure"]:

            if not st.session_state.paid_this_month:
                interest = compute_interest(u["principal"], u["rate"])
                u["principal"] += interest

                st.session_state.chat.append({
                    "role": "assistant",
                    "content": f"Failure to pay. Principal increased to ₹{round(u['principal'],2)}.",
                    "type": "payment_fail"
                })

            u["current_month"] += 1
            st.session_state.paid_this_month = False
            st.session_state.init_date += timedelta(days=30)
            st.rerun()

    st.write(f"Current Simulated Date: {st.session_state.init_date.strftime('%d-%m-%Y')}")

    st.text_input("Sentiment", value=u["sentiment"], disabled=True)

    st.session_state.user = u


def payment_panel():
    st.subheader("Payment")

    if st.session_state.paid_this_month:
        st.success("Payment already made this month")
        return

    amount = st.number_input("Enter Payment Amount", min_value=0.0)

    if st.button("Pay"):
        success, msg = make_payment(amount)

        if not success:
            st.error(msg)
        else:
            st.rerun()


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
            color = "#dcfce7"
            text_color = "#166534"
        elif msg.get("type") == "payment_success":
            color = "#bbf7d0"
            text_color = "#166534"
        elif msg.get("type") == "payment_fail":
            color = "#fed7aa"
            text_color = "#9a3412"
        else:
            color = None

        if color:
            with st.chat_message("assistant"):
                st.markdown(
                    f"""
                    <div style="
                        background-color:{color};
                        color:{text_color};
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

        sentiment = sentiment_analysis(user_input)
        st.session_state.user["sentiment"] = sentiment

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

    with col1:
        chat_ui()

    with col2:
        simulation_panel()
        st.divider()
        payment_panel()


def main():
    st.set_page_config(layout="wide")
    init_state()

    if not st.session_state.initialized:
        user_form()
    else:
        main_layout()


if __name__ == "__main__":
    main()