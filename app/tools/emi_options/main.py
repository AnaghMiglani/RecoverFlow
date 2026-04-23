from langchain.tools import tool
EMI_PLANS = [
        {"tenure_months": 1,  "interest_pa": "0%"},
        {"tenure_months": 2,  "interest_pa": "2%"},
        {"tenure_months": 3,  "interest_pa": "5%"},
        {"tenure_months": 6,  "interest_pa": "8%"},
        {"tenure_months": 9,  "interest_pa": "10%"},
        {"tenure_months": 12, "interest_pa": "12%"},
        {"tenure_months": 16, "interest_pa": "14%"},
        {"tenure_months": 20, "interest_pa": "15%"},
        {"tenure_months": 24, "interest_pa": "16%"},
    ]
@tool
def get_emi_plans():
    """Get all current available EMI plans.
    With tenure_months as well as interest_per_annum (interest_pa)
    """
    return  EMI_PLANS