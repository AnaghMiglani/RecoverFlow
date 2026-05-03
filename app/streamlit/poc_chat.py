import streamlit as st
import uuid
from datetime import datetime, timedelta
from app.agent.chat import chat_ai
from app.tools.emi_calc.main import loan_summary, principal_after_n_months
from app.agent.guardrail import guardrail_llm
from app.agent.sentiment import sentiment_analysis

# for deploying on streamlit
import os

try:
    import streamlit as st
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except:
    pass

# also for deployment
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


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

    if "last_payment_month" not in st.session_state:
        st.session_state.last_payment_month = 0

    if "loan_closed" not in st.session_state:
        st.session_state.loan_closed = False


def user_form():
    with st.form("user_details"):
        name = st.text_input("Name")
        principal = st.number_input("Principal Amount (₹)", min_value=0.0, value=10000.0)
        rate = st.number_input("Interest Rate (% per annum) upto 2 decimal places", min_value=0.0, value=12.0)
        tenure = st.number_input("Tenure (months)", min_value=1, value=12)
        # current_month = st.number_input("Current Month", min_value=1, value=1)

        submitted = st.form_submit_button("Start Simulation")

        if submitted:
            st.session_state.user = {
                "name": name,
                "principal": principal,
                "initial_principal": principal,
                "rate": rate,
                "tenure": tenure,
                "current_month": 1,
                "sentiment": "neutral"
            }

            st.session_state.init_date = datetime.now()
            st.session_state.initialized = True
            st.rerun()

def calculate_risk(user):
    initial_principal = user["initial_principal"]
    rate = user["rate"]
    tenure = user["tenure"]
    current_month = user["current_month"]
    actual = user["principal"]

    months_paid = max(0, current_month - 1)

    expected = principal_after_n_months(
        initial_principal,
        rate,
        tenure,
        months_paid
    )

    if expected == 0:
        return "LOW", 0

    deviation = ((actual - expected) / initial_principal) * 100

    deviation = max(0, deviation)
    deviation=round(deviation,2)

    if deviation >= 7:
        return "HIGH", round(deviation, 2)
    elif deviation >= 5:
        return "MEDIUM", round(deviation, 2)
    else:
        return "LOW", round(deviation, 2)

def compute_interest(principal, rate):
    return round(principal * (rate / 12 / 100),2)


def make_payment(amount):
    user = st.session_state.user

    if st.session_state.loan_closed:
        return False, "Loan already closed."

    if st.session_state.last_payment_month == user["current_month"]:
        return False, "Payment already made this month"

    interest = compute_interest(user["principal"], user["rate"])
    total_due = user["principal"] + interest
    total_due=round(total_due,2)

    if amount > total_due:
        return False, f"Payment cannot exceed ₹{round(total_due,2)}"

    new_principal = round(user["principal"] + interest - amount,2)
    user["principal"] = max(new_principal, 0)

    st.session_state.last_payment_month = user["current_month"]

    # FINAL MONTH LOGIC
    if user["current_month"] == user["tenure"]:
        if user["principal"] > 0:
            msg = f"Payment of ₹{amount} made. Remaining principal is ₹{round(user['principal'],2)}. Bank will contact you shortly."
            msg_type = "final_due"
        else:
            msg = f"Payment of ₹{amount} made. Loan fully repaid. Tenure completed."
            msg_type = "payment_success"

        st.session_state.loan_closed = True

    else:
        user["current_month"] += 1
        st.session_state.init_date += timedelta(days=30)

        msg = f"Payment of ₹{amount} made. New principal is ₹{round(user['principal'],2)}. Month advanced to {user['current_month']}."
        msg_type = "payment_success"

    st.session_state.chat.append({
        "role": "assistant",
        "content": msg,
        "type": msg_type
    })

    return True, ""


def simulation_panel():
    st.subheader("Simulation")
    u = st.session_state.user

    u["principal"]=round(u["principal"],2)

    st.text_input("Name", u["name"])
    st.text_input("Principal (₹)", value=u["principal"], disabled=True)
    st.text_input("Rate (% per annum)", value=u["rate"], disabled=True)
    st.text_input("Tenure (months)", value=u["tenure"], disabled=True)

    st.write(f"Current Month: {u['current_month']}")

    if st.button("Next Month"):
        if u["current_month"] < u["tenure"]:

            if st.session_state.last_payment_month != u["current_month"]:
                interest = compute_interest(u["principal"], u["rate"])
                new_principal = u["principal"] + interest
                st.session_state.user["principal"] = round(new_principal,2)

                st.session_state.chat.append({
                    "role": "assistant",
                    "content": f"Failure to pay. Principal increased to ₹{round(new_principal, 2)}.",
                    "type": "payment_fail"
                })

            st.session_state.user["current_month"] += 1
            st.session_state.init_date += timedelta(days=30)

            # Allow EMI for new month only
            st.session_state.last_emi_msg_month = None

            st.rerun()

    st.write(f"Current Simulated Date: {st.session_state.init_date.strftime('%d-%m-%Y')}")
    st.text_input("Sentiment", value=u["sentiment"], disabled=True)

    risk, deviation = calculate_risk(u)

    st.text_input("Risk", value=risk, disabled=True)
    st.text_input("Deviation (%)", value=deviation, disabled=True)


def payment_panel():
    st.subheader("Payment")

    user = st.session_state.user

    if st.session_state.loan_closed:
        st.warning("Loan closed. No further payments allowed.")
        return

    if st.session_state.last_payment_month == user["current_month"]:
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


def chat_ui():
    st.subheader("Chat")

    current_month = st.session_state.user["current_month"]

    if (
        st.session_state.last_emi_msg_month != current_month
        and st.session_state.last_payment_month != current_month
    ):
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
        elif msg.get("type") == "final_due":
            color = "#fecaca"
            text_color = "#7f1d1d"
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
            })
            st.rerun()
            return

        sentiment = sentiment_analysis(user_input)
        st.session_state.user["sentiment"] = sentiment

        st.session_state.chat.append({
            "role": "user",
            "content": user_input
        })

        u = st.session_state.user

        risk, deviation = calculate_risk(u)

        response = chat_ai(
            name=u["name"],
            rate=u["rate"],
            principal=u["principal"],
            tenure=u["tenure"],
            current_month=u["current_month"],
            sentiment=u["sentiment"],
            user_prompt=user_input,
            thread_id=st.session_state.thread_id,
            history=st.session_state.history,
            risk=risk
        )

        st.session_state.chat.append({
            "role": "assistant",
            "content": response,
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