from datetime import datetime

from pydantic import BaseModel, Field, EmailStr
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
import os

class UserScore(BaseModel):
    name: str = Field()
    trust_score: int = Field(ge=1,le=10) #out of 10
    last_response_time: datetime = Field() #to check if ignoring
    email : EmailStr = Field()

llm = ChatOpenAI(
    model="openrouter/free",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPEN_ROUTER_API"),
)
agent = create_agent(
    model=llm,
    response_format=UserScore,
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Generate a user profile with name, email, trust score, and last response time"}]
})
print(result["structured_response"])