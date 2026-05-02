import streamlit as st

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