from langchain_groq import ChatGroq
from langchain.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
load_dotenv()
import os
from langchain.agents import create_agent

llm=ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.8,
    max_retries=2,
    api_key=os.getenv("GROQ_API_KEY"),
)

with open("input_prompt.txt", "r", encoding="utf-8") as f:
    input_txt=f.read()

with open("feedback_prompt.txt", "r", encoding="utf-8") as f:
    feedback_txt=f.read()

model_name=llm.model_name

SYSTEM_PROMPT=f"""
You are a prompt engineering system.

Your ONLY task is to rewrite and improve the given prompt.

STRICT RULES:
- You MUST output a PROMPT, not a response to the prompt.
- Do NOT behave like a chatbot.
- Do NOT generate conversational replies like "Hello" or "Hope you are doing well".
- Do NOT answer the input.
- You are NOT interacting with the end user.
- You are ONLY rewriting the input into a better prompt.

OUTPUT FORMAT:
- Return ONLY the improved prompt.
- No explanations.
- No extra text.
- No greetings.

Target model: {model_name}
"""

USER_PROMPT=f"""
ORIGINAL PROMPT:
{input_txt}

FEEDBACK:
{feedback_txt if feedback_txt.strip() else "None"}

TASK:
Rewrite the ORIGINAL PROMPT using the FEEDBACK.
"""

response=llm.invoke([
    SystemMessage(content=SYSTEM_PROMPT),
    HumanMessage(content=USER_PROMPT)
])

# print(response.content)
with open("output_prompt.txt", "w", encoding="utf-8") as f:
    f.write(response.content)
