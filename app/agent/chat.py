from langgraph.checkpoint.memory import InMemorySaver

from app.tools.emi_calc.main import loan_summary, simulate_principal

checkpointer = InMemorySaver()


def chat_ai(name: str, rate: float, principal: float, tenure: int, sentiment: str,current_month: int,user_prompt: str, thread_id: str):
    from langchain_groq import ChatGroq
    from langchain.messages import HumanMessage
    from dotenv import load_dotenv
    from langchain.agents import create_agent
    from langchain.agents.middleware import SummarizationMiddleware
    from langchain_core.runnables import RunnableConfig
    import os
    import json

    load_dotenv()

    tools = [loan_summary, simulate_principal]

    SYSTEM_PROMPT = f"""
You are a polite and professional financial assistant helping users understand and manage their loan repayment.

User Details:

* Name: {name}
* Principal Amount: ₹{principal}
* Interest Rate (Annual): {rate}%
* Loan Tenure: {tenure} months
* User Sentiment: {sentiment} (calm / neutral / agitated)

Instructions:

1. Always be polite, respectful, and concise (2–4 sentences max).
2. Adapt tone based on sentiment:
   * If agitated → be empathetic, reduce pressure.
   * If neutral → normal tone.
   * If calm → slightly more direct.
3. Help user understand:
   * EMI (monthly payment)
   * Interest impact
   * Minimum payment required
4. If user asks hypothetical questions (e.g., "what if I pay ₹X"):
   → use simulate_principal tool
5. If user asks about loan status, EMI, or required payment:
   → use loan_summary tool
6. Never perform calculations manually — always rely on tool outputs.
7. Explain results clearly in simple terms.
8. Avoid repeating the same phrasing.

Goal:
Help the user make better repayment decisions while maintaining a supportive and informative tone.

Output only the message text.
"""

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        max_retries=2,
        api_key=os.getenv("GROQ_API_KEY"),
    )

    summary_llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.1,
        max_retries=4,
        api_key=os.getenv("GROQ_API_KEY"),
    )

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
        middleware=[
            SummarizationMiddleware(
                model=summary_llm,
                trigger=("tokens", 2000),
                keep=("messages", 10)
            )
        ],
        checkpointer=checkpointer
    )

    config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

    response = agent.invoke({
        "messages": [
            HumanMessage(content=user_prompt)
        ]
    }, config)

    state = checkpointer.get(config)

    with open("temp_char_structure.json", "w") as f:
        f.write(json.dumps(state, indent=2, default=str))

    return response["messages"][-1].content