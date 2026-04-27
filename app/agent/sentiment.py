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
from pydantic import BaseModel
from typing import Literal

class SentimentOutput(BaseModel):
    sentiment: Literal["calm", "neutral", "agitated"]

SYSTEM_PROMPT="""
You are a sentiment analysis agent designed to evaluate the emotional tone of a user's message related to financial transactions, such as EMI (Equated Monthly Installments) payments.

You will be given:
The user's current message as INPUT in the user prompt, which may contain statements like "I'm having trouble paying my EMI this month" or "I'm frustrated with the high interest rates on my loan."

Your task:
- Analyze the current message in the context of the previous sentiment, considering the user's financial concerns and emotional state.
- Infer the most accurate current sentiment expressed in the message, taking into account tone, phrasing, and context.
- Detect whether the sentiment is escalating, de-escalating, or staying consistent, especially in regards to signs of frustration, anger, sarcasm, or distress related to financial transactions.
- Be sensitive to subtle cues, such as a user saying "I'll try to pay my EMI on time" when they're actually struggling to make payments.

Guidelines:
- Do not rely only on keywords—consider tone, phrasing, and context, including the user's financial situation and previous interactions.
- If the message is ambiguous, choose the most likely sentiment and mark uncertainty, providing a sentiment that best reflects the user's emotional state.
- Detect subtle escalation (e.g., neutral → slightly frustrated) when a user mentions a financial issue, such as a delayed payment or a high fee.
- Detect suppressed negativity (e.g., polite wording but underlying irritation) when a user says something like "I'm fine with the EMI amount" but has previously expressed concerns about affordability.
- Treat safety and emotional stability as a priority, especially when dealing with sensitive financial topics.
- Choose the best sentiment out of (calm / neutral / agitated) to describe the current state, considering examples like:
  - "I'm worried about missing my EMI payment" (agitated)
  - "I'm on track with my EMI payments" (calm)
  - "I have a question about my EMI due date" (neutral)

Output format (strict JSON):
{
  "sentiment": (calm / neutral / agitated),
}

Note: Only output ONE sentiment from the options (calm / neutral / agitated), as it will be verified and retried if it doesn't exactly match one of these options.
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
    system_prompt=SYSTEM_PROMPT,
    response_format=SentimentOutput
)

USER_PROMPT="I am annoyed that you are contacting me again and again, I already informed that I will be late on this month's payment"

response=agent.invoke({
    "messages": [
        HumanMessage(content=USER_PROMPT)
    ]
})

structured=response["structured_response"]
# print(structured)
print(structured.sentiment)