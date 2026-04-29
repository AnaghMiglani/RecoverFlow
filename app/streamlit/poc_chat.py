import streamlit as st
import uuid
from app.agent.chat import chat_ai

def init_state():
    if "initialized" not in st.session_state:
        st.session_state.initialized=False
    if "user" not in st.session_state:
        st.session_state.user={
            "name":"",
            "amount":"",
            "emi_plan":"None",
            "last_contact":"None",
            "no_response_days":0,
            "sentiment":"neutral"
        }
    if "chat" not in st.session_state:
        st.session_state.chat=[]
    if "thread_id" not in st.session_state:
        st.session_state.thread_id=str(uuid.uuid4())

def user_form():
    with st.form("user_details"):
        name=st.text_input("Name")
        amount=st.text_input("Outstanding Amount")
        emi_plan=st.text_input("EMI Plan","None")
        last_contact=st.text_input("Last Contact","None")
        no_response_days=st.number_input("Days Since Last Response",min_value=0,value=0)
        sentiment=st.selectbox("Sentiment",["calm","neutral","agitated"])
        submitted=st.form_submit_button("Start Simulation")

        if submitted:
            st.session_state.user={
                "name":name,
                "amount":amount,
                "emi_plan":emi_plan,
                "last_contact":last_contact,
                "no_response_days":no_response_days,
                "sentiment":sentiment
            }
            st.session_state.initialized=True
            st.rerun()

def simulation_panel():
    st.subheader("Simulation")
    u=st.session_state.user

    u["name"]=st.text_input("Name",u["name"])
    u["amount"]=st.text_input("Amount",u["amount"])
    u["emi_plan"]=st.text_input("EMI Plan",u["emi_plan"])
    u["last_contact"]=st.text_input("Last Contact",u["last_contact"])
    u["no_response_days"]=st.number_input("No Response Days",value=u["no_response_days"])
    u["sentiment"]=st.selectbox("Sentiment",["calm","neutral","agitated"],index=["calm","neutral","agitated"].index(u["sentiment"]))

    st.session_state.user=u

def chat_ui():
    st.subheader("Chat")

    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_input=st.chat_input("Type your message")
    if user_input:
        st.session_state.chat.append({"role":"user","content":user_input})

        response=chat_ai(
            **st.session_state.user,
            user_prompt=user_input,
            thread_id=st.session_state.thread_id
        )

        st.session_state.chat.append({"role":"assistant","content":response})

        st.rerun()

def main_layout():
    col1,col2=st.columns([3,1])

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

if __name__=="__main__":
    main()