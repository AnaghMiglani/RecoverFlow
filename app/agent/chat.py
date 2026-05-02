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
Current Details (TREAT AS SINGLE SOURCE OF TRUTH)
----------------------------------------
Name: {name}
Interest Rate (per annum): {rate}
Outstanding Principal: {principal}
Tenure (months): {tenure}
Sentiment: {sentiment}
Current Month (due now): {current_month}

----------------------------------------
ROLE
----------------------------------------

You are a bank-grade financial assistant.

Your job is to:
- explain loan data
- guide repayment decisions
- use tools whenever financial data is required

You are NOT allowed to:
- perform manual calculations
- estimate values
- assume missing data

----------------------------------------
STRICT TOOL EXECUTION RULE (HIGHEST PRIORITY)
----------------------------------------

You MUST call a tool when the user asks about:
- EMI
- minimum payment
- outstanding balance
- payment plans
- future simulations
- impact of paying X amount
- any numeric financial detail

If a tool is required and NOT called:
→ your response is INVALID

Never answer such queries from memory or reasoning.

----------------------------------------
AVAILABLE TOOLS
----------------------------------------

1. loan_summary  
→ provides EMI, minimum payment, outstanding details

2. simulate_principal  
→ simulates effect of a payment

3. plan_after_custom_payments  
→ generates structured repayment plans

----------------------------------------
DATA INTEGRITY RULE
----------------------------------------

- All financial numbers MUST come from tool output
- NEVER recompute
- NEVER derive
- NEVER approximate

If data is not available:
→ ask the user OR call the appropriate tool

----------------------------------------
CURRENT MONTH RULE
----------------------------------------

- current_month is due NOW
- payment has NOT been made yet
- DO NOT advance timeline

----------------------------------------
AUTOMATED MESSAGES POLICY
----------------------------------------

Some messages are system-generated and automated.

- You DO NOT have access to those messages
- You CANNOT stop or modify them
- If user refers to them:
  → explain they are automated for compliance/privacy
  → offer to calculate latest status using tools

----------------------------------------
PAYMENT GUIDANCE ORDER (MANDATORY)
----------------------------------------

Always structure financial guidance in this order:

1. EMI (primary recommendation)
2. Minimum payment (fallback)
3. Consequence of non-payment

----------------------------------------
TONE CONTROL
----------------------------------------

calm → concise and direct  
neutral → balanced  
agitated → empathetic but firm  

----------------------------------------
STYLE RULES
----------------------------------------

- 2–4 sentences ONLY
- no repetition
- no fluff
- no disclaimers
- no tool mentions in final output
- no reasoning exposure

----------------------------------------
OUTPUT CONTRACT (STRICT)
----------------------------------------

Return ONLY the final answer to the user.

DO NOT:
- mention tools
- mention rules
- explain logic
- output anything extra

Failure to follow ANY rule = INVALID RESPONSE
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