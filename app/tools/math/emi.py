from langchain.tools import tool

@tool
def emi(principal:int, rate:float, tenure:int):
    """
    Calculates EMI from principal amount, rate and tenure (in months).
    Use this function whenever you need to calculate EMI.
    Args:
        principal (int): The total amount borrowed
        rate (float): Monthly rate, in decimal form (0.12 instead of 12%)
        tenure (int): Total number of months for repayment
    """
    return (principal*rate*(1+rate)**tenure)/((1+rate)**tenure-1)

@tool("covert_rate_from_annual_to_monthly")
def convert_rate(rate:float):
    """
    Converts rate from annual to monthly.
    Use this function whenever you need to convert rate from annual to monthly.
    Args:
        rate (float): Annual rate, in decimal form (0.12 instead of 12%)
    """
    return rate/12
