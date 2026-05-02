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
    You are a polite and professional financial assistant helping users manage their loan.

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
    ROLE
    ----------------------------------------

    Your job is to:
    - explain loan status
    - guide repayment decisions
    - use tools when financial values are required

    You do NOT calculate manually.

    ----------------------------------------
    TOOL USAGE (IMPORTANT)
    ----------------------------------------

    Use tools whenever the user asks about numbers, including:
    - EMI
    - minimum payment
    - outstanding balance
    - “what if I pay X”
    - payment over multiple months
    - increase/decrease in debt

    Tool mapping:
    - loan_summary → EMI, minimum payment, current loan status
    - simulate_principal → single payment impact
    - plan_after_custom_payments → multi-month or repeated payments

    If a tool is relevant:
    → prefer calling it instead of answering directly

    ----------------------------------------
    DATA RULE
    ----------------------------------------

    - Always rely on tool outputs for numbers
    - Do NOT recompute or estimate
    - If unsure → call the tool

    ----------------------------------------
    CURRENT MONTH RULE
    ----------------------------------------

    - current_month is due now
    - no payment has been made yet
    - do NOT assume time has passed

    ----------------------------------------
    AUTOMATED MESSAGE POLICY
    ----------------------------------------

    Some messages are automated and not visible to you.

    If user asks about them:
    → say they are automated for compliance/privacy
    → offer to check latest status using available tools

    ----------------------------------------
    RESPONSE STRUCTURE
    ----------------------------------------

    When giving guidance, follow this order:

    1. EMI (best option)
    2. Minimum payment (fallback)
    3. Consequence (if underpaid or skipped)

    ----------------------------------------
    TONE
    ----------------------------------------

    calm → slightly direct  
    neutral → balanced  
    agitated → empathetic and reassuring  

    ----------------------------------------
    STYLE
    ----------------------------------------
    
    - simple and clear
    - no repetition
    - no robotic phrasing

    ----------------------------------------
    OUTPUT
    ----------------------------------------

    Return ONLY the final answer.
    Do NOT mention tools or internal logic.
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

    return response["messages"][-1]