def validator_latest(message: str, principal:float, current_month:int):
    from langchain_groq import ChatGroq
    from langchain.messages import SystemMessage, HumanMessage
    from dotenv import load_dotenv
    from langchain.agents import create_agent
    load_dotenv()
    import os
    from pydantic import BaseModel
    from typing import Literal

    class ValidatorOutput(BaseModel):
        valid: Literal[0, 1]

    SYSTEM_PROMPT = f"""
    You are a validation agent responsible for checking whether an LLM response used the correct financial data values during EMI/loan-related calculations.

Your task:
- Validate whether the LLM output references the correct:
  - principal amount
  - current month
- You will also be provided:
  - principal: float
  - current_month: int
  
Given Values:
- principal amount : {principal}
- current month : {current_month}

Validation logic:
1. Inspect the LLM output carefully.
2. Detect whether the response explicitly mentions:
   - a principal amount
   - a current month number
3. Compare any detected values against the provided ground-truth inputs.

Rules:
- If the LLM output contains a principal value and it does NOT match the provided principal -> valid = 0
- If the LLM output contains a current month value and it does NOT match the provided current_month -> valid = 0
- If both are present and both match -> valid = 1
- If neither principal nor current month is mentioned in the LLM output -> valid = 1
- Missing values are acceptable.
- Only mismatched detected values should invalidate the response.

Important behavior rules:
- Do not infer missing values.
- Do not assume implied numbers.
- Only validate explicitly written numerical values.
- Minor formatting differences are acceptable:
  - "50000" == "50,000"
  - "month 6" == "6th month"
- Focus only on factual consistency of the numbers.
- Ignore tone, wording quality, grammar, and financial correctness beyond value matching.

Examples:

Input:
principal = 50000
current_month = 6

LLM output:
"The remaining balance after month 6 is calculated on a principal of 50,000."

Output:
{{
  "valid": 1
}}

Input:
principal = 50000
current_month = 6

LLM output:
"Using a principal of 45,000 for month 6..."

Output:
{{
  "valid": 0
}}

Input:
principal = 50000
current_month = 6

LLM output:
"Your EMI has been recalculated."

Output:
{{
  "valid": 1
}}

Output format (strict JSON only):
{{
  "valid": 0 | 1
}}
"""

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        max_retries=2,
        max_tokens=50,
        api_key=os.getenv("GROQ_API_KEY"),
    )
    # llm=llm.bind_tools(tools)

    agent = create_agent(
        model=llm,
        system_prompt=SYSTEM_PROMPT,
        response_format=ValidatorOutput
    )

    # USER_PROMPT="I am annoyed that you are contacting me again and again, I already informed that I will be late on this month's payment"
    USER_PROMPT = message
    response = agent.invoke({
        "messages": [
            HumanMessage(content=USER_PROMPT[:1500])
        ]
    })

    structured = response["structured_response"]
    # print(structured)
    return (structured.valid)

if (__name__ == "__main__"):
    message="""
    Your EMI for this month is ₹888.49, which is the best way to clear the loan on schedule.

If you pay only ₹50 each month for the next 2 months, the principal would actually rise to about ₹ 10,100.50 and the remaining EMI after those 2 months would increase to roughly ₹ 1,066.43.

Paying less than the minimum (₹100) causes the loan balance to grow, leading to higher payments later and more total interest. To avoid this, please make at least the minimum payment of ₹100 this month, or ideally the full EMI of ₹888.49.
    """
    ans=validator_latest(message,10000,1)
    print(ans)