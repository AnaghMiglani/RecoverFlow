def sentiment_analysis(message:str):
    from langchain_groq import ChatGroq
    from langchain.messages import SystemMessage, HumanMessage
    from dotenv import load_dotenv
    from langchain.agents import create_agent
    load_dotenv()
    import os
    from pydantic import BaseModel
    from typing import Literal

    class SentimentOutput(BaseModel):
        sentiment: Literal["calm", "neutral", "agitated"]

    SYSTEM_PROMPT="""
    You are a sentiment analysis agent designed to evaluate the emotional tone and intent of a user's message related to financial transactions, especially EMI (Equated Monthly Installments), loan payments, dues, penalties, and repayment discussions.

Your task:
- Analyze the emotional tone, intent, and behavioral attitude in the user's message.
- Focus primarily on the wording, phrasing, intent, and emotional intensity of the current message.
- Detect whether the user is:
  - cooperative but struggling,
  - emotionally distressed,
  - frustrated or confrontational,
  - calm and compliant,
  - or simply asking informational questions.

Important interpretation rules:
- Statements expressing inability or financial hardship are NOT automatically negative or agitated.
  - Example:
    - "I won't be able to pay this month" -> likely neutral
    - "I'm trying to arrange the EMI somehow" -> calm

- Statements expressing refusal, hostility, defiance, threats, or aggression should be treated as agitated.
  - Example:
    - "I won't pay this month"
    - "Stop calling me about the EMI"
    - "Do whatever you want, I'm not paying"

- Distinguish inability from unwillingness:
  - inability = neutral/calm depending on tone
  - unwillingness/refusal = agitated

- Detect subtle frustration even when phrased politely.
  - Example:
    - "I guess I'll somehow manage the EMI again..." -> possibly agitated if emotionally strained

- Do not rely only on keywords like "can't pay", "loan", or "EMI".
- Consider:
  - emotional intensity,
  - cooperation,
  - sarcasm,
  - passive aggression,
  - resignation,
  - distress,
  - escalation patterns.

Sentiment categories:
- calm:
  - cooperative, composed, solution-oriented, stable
  - examples:
    - "I'll pay it next week."
    - "I'm arranging the amount."
    - "Can we extend the due date?"

- neutral:
  - factual, emotionally low-intensity, uncertain, or hardship-related without aggression
  - examples:
    - "I won't be able to pay this month."
    - "The EMI is delayed."
    - "I lost my job recently."

- agitated:
  - frustrated, angry, confrontational, threatening, hostile, sarcastic, or emotionally escalated
  - examples:
    - "I won't pay this EMI."
    - "This loan is a scam."
    - "Stop harassing me."
    - "Do whatever you want."

Output format (strict JSON only):
{
  "sentiment": "calm | neutral | agitated"
}
"""

    llm=ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.1,
        max_retries=2,
        max_tokens=50,
        api_key=os.getenv("GROQ_API_KEY"),
    )
    # llm=llm.bind_tools(tools)

    agent=create_agent(
        model=llm,
        system_prompt=SYSTEM_PROMPT,
        response_format=SentimentOutput
    )

    # USER_PROMPT="I am annoyed that you are contacting me again and again, I already informed that I will be late on this month's payment"
    USER_PROMPT=message
    response=agent.invoke({
        "messages": [
            HumanMessage(content=USER_PROMPT[:1500])
        ]
    })

    structured=response["structured_response"]
    # print(structured)
    return(structured.sentiment)

if (__name__=="__main__"):
    message="""
    I’m honestly getting quite frustrated with how this whole EMI situation is being handled. I understand that payments need to be made on time and I’ve never intentionally tried to delay anything, but this month has been unusually difficult for me financially due to some unexpected medical expenses in my family. I had already informed your support team a few days ago that I might be late by a week or so, yet I’m still receiving repeated reminders and calls which is making the situation more stressful than it already is.
    
    It feels like there’s no consideration for genuine cases where customers are actually trying to cooperate but just need a little flexibility. I’m not refusing to pay, and I fully intend to clear the EMI as soon as my cash flow stabilizes, but the constant notifications are honestly starting to feel a bit overwhelming and unnecessary given that I’ve already communicated my situation.
    
    Also, I tried checking if there are any options available to temporarily reduce the EMI amount or restructure the payment for this month, but I couldn’t find anything clearly explained in the app or the messages I’ve received. If there are such options, I would really appreciate some guidance instead of just automated reminders that don’t take context into account.
    
    At this point, I just want to understand what flexibility I actually have and whether there’s a way to handle this without negatively impacting my credit score, because that’s another thing I’m worried about. I’m trying to manage things responsibly, but the lack of clear communication and the repeated follow-ups are making it harder to stay calm about the whole situation.
    
    Please let me know what can be done here, and if possible, I’d appreciate fewer reminders for the next few days while I sort this out.
    """
    print(sentiment_analysis(message))