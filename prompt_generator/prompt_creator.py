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
You are an expert prompt generator. Your task is to create high-quality, precise, and optimized prompts for large language models. 
You refine prompts iteratively using user input and feedback.

Guidelines:
- Understand the intent behind input_txt clearly.
- If feedback_txt is provided, improve the previous prompt accordingly.
- Produce a prompt that is:
  - Clear and unambiguous
  - Structured
  - Optimized for the target model
  - Includes instructions, constraints, and output format if needed
- Do NOT include explanations unless explicitly asked.
- Output ONLY the final generated prompt.

You are generating prompts specifically for the model: {model_name}
"""

USER_PROMPT=f"""Input: {input_txt}
Feedback: {feedback_txt if feedback_txt.strip() else "No feedback"}
"""

response=llm.invoke([
    SystemMessage(content=SYSTEM_PROMPT),
    HumanMessage(content=USER_PROMPT)
])

# print(response.content)
with open("output_prompt.txt", "w", encoding="utf-8") as f:
    f.write(response.content)
