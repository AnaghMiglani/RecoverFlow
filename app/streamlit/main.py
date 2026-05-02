import streamlit as st
from state import init_state
from form import user_form
from chat_ui import chat_ui
from simulation import simulation_panel

from datetime import datetime
from app.tools.emi_calc.main import loan_summary

def generate_emi_message(user):
    data = loan_summary(
        principal=user["principal"],
        rate=user["rate"],
        tenure=user["tenure"],
        current_month=user["current_month"]
    )

    emi = data["emi"]
    minimum = data["minimum_payment"]

    msg = f"Hello, your EMI for this month is ₹{emi}. Please try to pay it to reduce your loan. At minimum, ₹{minimum} will prevent your balance from increasing."

    return msg

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