from langchain.tools import tool
from datetime import datetime
from zoneinfo import ZoneInfo

@tool
def get_current_datetime_ist() -> str:
    """
    Returns current date and time in Indian Standard Time (IST).
    Use this tool whenever the query involves time-sensitive information,
    such as current time, date, or real-time context-dependent decisions.
    """
    ist = ZoneInfo("Asia/Kolkata")
    now = datetime.now(ist)
    return now.strftime("%Y-%m-%d %H:%M:%S IST")