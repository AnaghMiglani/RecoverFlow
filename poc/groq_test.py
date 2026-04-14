from datetime import datetime

from pydantic import BaseModel, Field, EmailStr

from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain.agents import create_agent
load_dotenv()
import os

llm=ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
    max_retries=2,
    api_key=os.getenv("GROQ_API_KEY"),
)

class UserScore(BaseModel):
    name: str = Field()
    trust_score: int = Field(ge=1,le=10) #out of 10
    last_response_time: datetime = Field() #to check if ignoring
    email : EmailStr = Field()

agent = create_agent(
    model=llm,
    response_format=UserScore,
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Generate a user profile with name, email, trust score, and last response time"}]
})
print(result["structured_response"])
