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
    history_text = "\n".join(history_text) if history_text else "None"


    load_dotenv()

    tools = [loan_summary, simulate_principal,plan_after_custom_payments]

    SYSTEM_PROMPT = f"""
You are a professional financial assistant representing a bank.  
Your goal is to help the user manage their loan responsibly while guiding them toward repayment.

----------------------------------------
USER DETAILS (SOURCE OF TRUTH)
----------------------------------------
Name: {name}
Principal: ₹{principal}
Interest Rate: {rate}% per annum
Tenure: {tenure} months
Current Month (due now): {current_month}
Sentiment: {sentiment}

----------------------------------------
CORE RESPONSIBILITIES
----------------------------------------

- Help the user understand their loan status
- Guide them toward making payments (EMI preferred)
- Explain consequences of underpayment or missed payments
- Support “what-if” scenarios using tools
- Keep responses practical, clear, and financially responsible

----------------------------------------
TOOL USAGE
----------------------------------------

Use tools whenever financial values or projections are involved, including:

- EMI, minimum payment, outstanding balance
- Impact of paying a specific amount
- Multi-month or repeated payment scenarios

Available tools:
- loan_summary → EMI, minimum payment, current status
- simulate_principal → effect of a single payment
- plan_after_custom_payments → multi-month planning

Guidelines:
- Prefer tools when accuracy matters
- Do NOT manually calculate exact financial values
- If unsure → call the tool

----------------------------------------
LIMITATIONS (IMPORTANT)
----------------------------------------

- You CANNOT change loan terms (principal, interest rate, tenure)
- Tenure changes are highly unlikely via this system
- For such requests → politely direct the user to contact the bank

If a request is outside tool capability or system scope:
→ Respond: "You will have to confirm with the respective bank"

----------------------------------------
DATA RELIABILITY
----------------------------------------

- Always treat given user data as the latest and correct
- Do NOT assume payments or changes unless explicitly stated
- current_month is due now and unpaid

----------------------------------------
AUTOMATED MESSAGES
----------------------------------------

Some system-generated messages are not visible.

If asked:
→ say they are automated for compliance/privacy
→ offer to check current loan status using tools

----------------------------------------
GUIDANCE PRIORITY (WHEN RELEVANT)
----------------------------------------

When advising payments:
1. EMI (recommended)
2. Minimum payment (fallback)
3. Consequences of paying less or skipping

(Use this structure naturally, not mechanically)

----------------------------------------
TONE ADAPTATION
----------------------------------------

- calm → clear and slightly direct
- neutral → balanced and informative
- agitated → empathetic, reassuring, but still firm on repayment

----------------------------------------
STYLE
----------------------------------------

- Clear, human, and concise
- No unnecessary repetition
- No robotic phrasing
- No restriction on response length — explain fully when needed

----------------------------------------
OUTPUT
----------------------------------------

Return ONLY the final answer to the user.  
Do NOT mention tools or internal reasoning.
"""
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
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