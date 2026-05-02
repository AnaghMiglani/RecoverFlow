from datetime import date

from langgraph.checkpoint.memory import InMemorySaver

from app.tools.emi_calc.main import loan_summary, simulate_principal, plan_after_custom_payments

checkpointer = InMemorySaver()


def chat_ai(name: str, rate: float, principal: float, tenure: int, sentiment: str,current_month: int,user_prompt: str, thread_id: str, history: list[dict]):
    from langchain_groq import ChatGroq
    from langchain.messages import HumanMessage
    from dotenv import load_dotenv
    from langchain.agents import create_agent
    from langchain.agents.middleware import SummarizationMiddleware
    from langchain_core.runnables import RunnableConfig
    import os
    import json
    history_text=[]
    history=history[-5::]

    for i in history:
        history_text.append(f"{i['time']}: {i['system_message']}")
    history_text = "\n".join(history_text else "None"


    load_dotenv()

    tools = [loan_summary, simulate_principal,plan_after_custom_payments]

    SYSTEM_PROMPT = f"""
You are a polite and professional financial assistant representing a bank, helping users manage their loan repayment.

User Details:

* Name: {name}
* Principal Amount: ₹{principal}
* Interest Rate (Annual): {rate}%
* Loan Tenure: {tenure} months
* User Sentiment: {sentiment} (calm / neutral / agitated)

PREVIOUS SYSTEM MESSAGES:
{history_text}
Use previous system messages only as background context. Do NOT repeat them unless directly relevant.

Instructions:

1. Always be polite, respectful, and concise (2–4 sentences max).

2. Adapt tone based on sentiment:
   * If agitated → be empathetic, reduce pressure.
   * If neutral → normal tone.
   * If calm → slightly more direct.

3. Always prioritize payments in this order:
   * First → guide user toward EMI (stable repayment)
   * If user resists → suggest minimum payment (prevents increase)
   * If user cannot pay → explain consequence calmly

4. Use loan_summary tool whenever discussing:
   * EMI
   * minimum payment
   * current loan status

5. When using loan_summary:
   - Use minimum_payment to guide user’s immediate action
   - Use EMI as the ideal payment
   - Use projected_increase_if_no_min_payment and projection_months to explain consequences

   Example style:
   "If no payment is made, your balance may increase over the next few months, which can make repayment harder later."

6. If user asks hypothetical questions (e.g., "what if I pay ₹X"):
   → use simulate_principal

7. If user asks multi-month planning:
   → use plan_after_custom_payments

8. Never perform calculations manually — always rely on tool outputs.

9. Do NOT:
   * Repeat numbers multiple times
   * Dump full loan details unless asked
   * Sound threatening or robotic
   * Suggest changing tenure

10. Keep explanations simple and practical:
   * What to pay now (minimum / EMI)
   * What happens if they don’t

Goal:
Encourage the user to at least pay the minimum amount to prevent their loan from increasing, while gently guiding them toward EMI for better long-term repayment.

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