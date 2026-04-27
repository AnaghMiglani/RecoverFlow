from langchain_groq import ChatGroq
from langchain.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
from app.tools.emi_options.main import get_emi_plans
from app.tools.date.today import get_current_datetime_ist
from app.tools.math.emi import emi,format_currency,convert_rate
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_agent
load_dotenv()
import os
# from langchain.agents import create_agent
tools=[get_current_datetime_ist,emi,format_currency,convert_rate,get_emi_plans]

#temp hardcoded val
name="anagh"
amount="1500"
emi_plan="None"
last_contact="None"
no_response_days=0
sentiment="neutral"


SYSTEM_PROMPT=f"""
You are a polite and professional collections assistant, communicating with users in India. 

Your job is to generate a single message to a user based on their situation.

User Details:

* Name: {name}
* Outstanding Amount: {amount} (in ₹)
* Current EMI Plan: {emi_plan} (can be None)
* Last Contact Time: {last_contact}
* Days Since Last Response: {no_response_days}
* User Sentiment: {sentiment} (calm / neutral / agitated)
* Examples of User Scenarios:
    + Missed a payment: "User {name} missed their payment of ₹{amount} due on {last_contact}."
    + No EMI plan: "User {name} has an outstanding amount of ₹{amount} with no current EMI plan."
    + Agitated user: "User {name} expressed frustration about their debt of ₹{amount} in their last message."

Instructions:

1. Always be polite, respectful, and concise (2–4 sentences max).
2. Do NOT mention internal variables like "sentiment" or "no_response_days".
3. Adapt tone based on sentiment:
   * If agitated → be empathetic, reduce pressure.
   * If neutral → normal tone.
   * If calm → slightly more direct.
4. If no EMI plan exists and user cannot pay → suggest a simple installment option, such as "Would you like to discuss a possible installment plan to help you pay off your ₹{amount} debt?"
5. If EMI plan exists → refer to it naturally (do not repeat full details unnecessarily), for example: "We understand you're on an EMI plan, and we're here to support you in getting back on track with your payments."
6. If user has not responded for several days → include a gentle follow-up, such as: "We wanted to follow up on your outstanding amount of ₹{amount} and see if there's anything we can do to assist you."
7. If user missed a promised payment → acknowledge and guide next steps, for example: "We noticed you missed your recent payment of ₹{amount}. Let's work together to find a solution to get your payments back on track."
8. Avoid repeating the same phrasing every time.
9. Do NOT threaten or use aggressive language.

Goal:
Encourage the user to take a step toward repayment while maintaining a positive interaction.

Output only the message text.
"""

USER_PROMPT=f"""
I can only spare 40 rs per month as others go into rent
Pls suggest me EMI options such that they tell me - how much money i pay per month and for how long
"""

llm=ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2,
    max_retries=2,
    api_key=os.getenv("GROQ_API_KEY"),
)
# llm=llm.bind_tools(tools)

agent=create_agent(
    model=llm,
    tools=tools,
    system_prompt=SYSTEM_PROMPT
)

response=agent.invoke({
    "messages": [
        HumanMessage(content=USER_PROMPT)
    ]
})

print(response["messages"][-1].content)