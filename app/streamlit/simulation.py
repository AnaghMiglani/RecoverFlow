import streamlit as st

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