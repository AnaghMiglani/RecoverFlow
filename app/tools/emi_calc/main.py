import math

from langchain_core.tools import tool


def monthly_rate(rate):
    return rate / (12 * 100)


def emi(principal, rate, tenure):
    r = monthly_rate(rate)
    if r == 0:
        return principal / tenure
    return principal * r * (1 + r)**tenure / ((1 + r)**tenure - 1)


def interest_for_month(principal, rate):
    return principal * monthly_rate(rate)


def principal_after_n_months(principal, rate, tenure, months_paid):
    r = monthly_rate(rate)
    p = principal

    for i in range(months_paid):
        remaining_tenure = tenure - i
        if remaining_tenure <= 0:
            break

        e = emi(p, rate, remaining_tenure)
        interest = p * r
        principal_paid = e - interest
        p = round(p - principal_paid, 2)

    return max(p, 0)


def projection_if_no_min_payment(principal, rate, months):
    r = monthly_rate(rate)
    p = principal

    for _ in range(months):
        p = round(p + p * r, 2)

    return p


@tool
def loan_summary(principal: float, rate:float, tenure:int, current_month:int) -> dict:
    """
        PURPOSE:
        --------
        Compute a complete snapshot of a loan at a given month, including EMI,
        interest due for the current month, minimum payment required to avoid
        increase in principal, remaining principal, and projected increase if
        no minimum payment is made.

        DEFINITIONS (VERY IMPORTANT):
        ----------------------------
        principal:
            The original loan amount borrowed by the user at the start of the loan.
            Unit: currency (e.g., INR, USD)
            Example: 100000

        rate:
            Annual interest rate expressed as a percentage.
            This is NOT monthly. It is converted internally to monthly.
            Example: 12 means 12% per annum.

        tenure:
            Total duration of the loan in months.
            Example: 12 means 1 year, 24 means 2 years.

        current_month:
            The month number in the loan lifecycle for which the summary is required.
            Starts from 1.
            Example:
                1 → first month
                6 → sixth month

        INTERNAL INTERPRETATION:
        ------------------------
        - Interest is applied monthly.
        - EMI (Equated Monthly Installment) is dynamically recalculated based on remaining principal and remaining tenure.
        - Each EMI consists of:
            interest component + principal component
        - Interest is always paid first, then principal.

        LOGIC FLOW:
        -----------
        1. Clamp current_month so it does not exceed tenure.
        2. Compute remaining principal after (current_month - 1) EMIs.
        3. Compute interest for current month using remaining principal.
        4. Compute remaining tenure.
        5. Compute EMI using remaining principal, rate, and remaining tenure.
        6. If final month → full payment required (principal + interest).
        7. Project future principal growth assuming NO minimum payment is made.
           Projection is limited to min(remaining_tenure, 6 months).
        8. Calculate increase in principal due to non-payment.

        RETURN VALUES (STRUCTURED OUTPUT):
        ----------------------------------
        The function returns a dictionary with:

        month_number:
            The evaluated month in the loan cycle.

        emi:
            Dynamically adjusted EMI required to repay the loan fully within remaining tenure.

        interest_this_month:
            Interest charged for the current month on remaining principal.

        minimum_payment:
            Minimum amount required to prevent principal from increasing.
            NOTE:
                - Equal to interest_this_month for normal months.
                - Equal to full outstanding (principal + interest) in final month.

        remaining_principal:
            Outstanding loan amount after previous payments.

        projected_increase_if_no_min_payment:
            Increase in principal if user does NOT pay even the minimum amount.
            Calculated over projection_months duration.

        projection_months:
            Number of months used for projection (≤ 6).

        IMPORTANT FINANCIAL RULES:
        --------------------------
        - If payment < minimum_payment:
            principal will INCREASE.
        - If payment == minimum_payment:
            principal remains unchanged (except final month where full payment is required).
        - If payment > minimum_payment:
            principal decreases.

        LLM USAGE INSTRUCTIONS (CRITICAL):
        ---------------------------------
        Use this function when:
            - User asks about loan status
            - User asks about EMI
            - User asks "how much should I pay"
            - User asks about interest or outstanding balance

        Do NOT use this function when:
            - User asks hypothetical "what if I pay X"
              (use simulate_principal instead)

        INTERPRETATION RULES FOR LLM:
        -----------------------------
        - Treat all numeric outputs as ground truth.
        - Do NOT recompute values.
        - Use "minimum_payment" to guide user advice.
        - If projected_increase_if_no_min_payment > 0:
            warn user about loan growth.

        EXAMPLE OUTPUT:
        ---------------
        {
            "month_number": 3,
            "emi": 8560.45,
            "interest_this_month": 4200.12,
            "minimum_payment": 4200.12,
            "remaining_principal": 98200.50,
            "projected_increase_if_no_min_payment": 1250.75,
            "projection_months": 6
        }

        SUMMARY:
        --------
        This function provides a deterministic financial snapshot.
        It MUST be used as the single source of truth for loan calculations.
        LLM should only explain the results, not modify or reinterpret them.
    """
    current_month = min(current_month, tenure)

    current_principal = principal

    remaining_tenure = tenure - current_month + 1

    monthly_interest = interest_for_month(current_principal, rate)

    emi_value = emi(current_principal, rate, remaining_tenure)

    if remaining_tenure == 1:
        minimum_payment = current_principal + monthly_interest
        emi_value = minimum_payment
    else:
        minimum_payment = monthly_interest

    projection_months = min(remaining_tenure, math.ceil(remaining_tenure/20),6)

    projected_principal = projection_if_no_min_payment(
        current_principal, rate, projection_months
    )

    increase = projected_principal - current_principal

    return {
        "month_number": current_month,
        "emi": round(emi_value, 2),
        "interest_this_month": round(monthly_interest, 2),
        "minimum_payment": round(minimum_payment, 2),
        "remaining_principal": round(current_principal, 2),
        "projected_increase_if_no_min_payment": round(increase, 2),
        "projection_months": projection_months
    }


@tool
def simulate_principal(principal:float, rate:float, payment:float) -> dict:
    """
        PURPOSE:
        --------
        Simulate how the loan principal changes if a user makes a specific payment
        in the current month. This is a hypothetical calculation and does NOT modify
        actual loan state.

        DEFINITIONS (VERY IMPORTANT):
        ----------------------------
        principal:
            Current outstanding loan amount before making any payment this month.
            Unit: currency (e.g., INR, USD)
            Example: 100000

        rate:
            Annual interest rate expressed as a percentage.
            This is converted internally to a monthly rate.
            Example: 12 means 12% per annum.

        payment:
            Amount the user intends to pay in the current month.
            This is a hypothetical value provided by the user.
            Example: 1000, 5000, etc.

        INTERNAL INTERPRETATION:
        ------------------------
        - Interest is applied monthly.
        - Monthly interest = principal * (rate / 12 / 100)
        - Payment is applied in the following order:
            1. Interest is paid first
            2. Remaining amount (if any) reduces principal

        LOGIC FLOW:
        -----------
        1. Compute monthly interest on current principal.
        2. Compare payment with interest:
            - If payment >= interest:
                principal decreases by (payment - interest)
            - If payment < interest:
                unpaid interest is added to principal
        3. Ensure principal never goes below zero.
        4. Compute net change in principal.

        RETURN VALUES (STRUCTURED OUTPUT):
        ----------------------------------
        The function returns a dictionary with:

        old_principal:
            Principal before applying payment.

        new_principal:
            Principal after applying payment.

        change:
            Difference between new and old principal.
            Interpretation:
                positive  → principal increased (bad)
                zero      → no change
                negative  → principal reduced (good)

        IMPORTANT FINANCIAL RULES:
        --------------------------
        - If payment < interest:
            principal increases (loan grows)
        - If payment == interest:
            principal remains unchanged
        - If payment > interest:
            principal decreases

        LLM USAGE INSTRUCTIONS (CRITICAL):
        ---------------------------------
        Use this function when:
            - User asks hypothetical questions like:
                "What if I pay ₹X?"
                "If I pay 5000 this month, what happens?"
                "Will my loan reduce if I pay this amount?"

        Do NOT use this function when:
            - User asks for actual loan status
              (use loan_summary instead)

        INTERPRETATION RULES FOR LLM:
        -----------------------------
        - Treat returned values as exact and authoritative.
        - Do NOT recompute interest or principal.
        - Use "change" to explain outcome:
            if change > 0 → warn user about increase
            if change < 0 → highlight reduction
        - Clearly state that this is a simulation.

        EXAMPLE OUTPUT:
        ---------------
        {
            "old_principal": 100000,
            "new_principal": 103000,
            "change": 3000
        }

        SUMMARY:
        --------
        This function is used for hypothetical "what-if" analysis.
        It provides a deterministic and reliable estimate of how a specific
        payment affects the loan principal for the current month.
        The LLM must only explain the result and must not alter calculations.
    """
    interest = principal * (rate / (12 * 100))

    if payment >= interest:
        new_principal = round(principal - (payment - interest),2)
    else:
        new_principal = round(principal + (interest - payment),2)


    new_principal = max(new_principal, 0)
    change = round(new_principal - principal,2)

    return {
        "old_principal": round(principal, 2),
        "new_principal": round(new_principal, 2),
        "change": round(change, 2)
    }


@tool
def plan_after_custom_payments(principal:float, rate:float, tenure:int, current_month:int, payment:float, months:int):
    """
    PURPOSE:
    --------
    Simulate a repayment strategy where the user pays a fixed custom amount
    for a specified number of months, and then compute the required EMI for
    the remaining tenure.

    This function models a two-phase repayment:
        Phase 1 → custom payments (user-defined)
        Phase 2 → recalculated EMI for remaining months

    DEFINITIONS (VERY IMPORTANT):
    ----------------------------
    principal:
        Original loan amount taken by the user.
        Unit: currency (e.g., INR, USD)

    rate:
        Annual interest rate in percentage.
        Example: 12 means 12% per annum.

    tenure:
        Total loan duration in months.

    current_month:
        Current month in loan lifecycle (starting from 1).

    payment:
        Fixed amount user plans to pay every month during the custom period.

    months:
        Number of months user will follow this custom payment plan.

    INTERNAL INTERPRETATION:
    ------------------------
    - Interest is applied monthly.
    - Monthly interest = principal * (rate / 12 / 100)
    - Each payment is applied:
        1. Interest is paid first
        2. Remaining amount reduces principal
    - If payment < interest:
        unpaid interest increases principal
    - If payment is large:
        loan may close early

    LOGIC FLOW:
    -----------
    1. Compute current principal based on loan progress.
    2. Simulate "months" number of payments:
        - Apply interest
        - Apply payment
        - Update principal
        - Stop early if principal becomes zero
    3. Track how many months were actually used.
    4. Compute remaining tenure.
    5. Recalculate EMI for remaining principal and months.
    6. Detect if loan closed early.

    RETURN VALUES (STRUCTURED OUTPUT):
    ----------------------------------
    principal_after_custom_period:
        Remaining principal after applying custom payments.

    remaining_months:
        Months left in the loan after custom period.

    new_emi:
        EMI required to finish the loan in remaining months.
        Will be 0 if loan is already closed.

    loan_closed_early:
        Boolean indicating whether loan was fully repaid during custom period.

    months_to_close:
        Number of months required to fully close loan (only if closed early).

    closed_in_month_number:
        Absolute loan month in which loan gets closed.

    IMPORTANT FINANCIAL RULES:
    --------------------------
    - If payment < interest:
        loan grows (negative amortization)
    - If payment = interest:
        principal stays same
    - If payment > interest:
        principal reduces
    - If payment is very high:
        loan may close before planned months

    LLM USAGE INSTRUCTIONS (CRITICAL):
    ---------------------------------
    Use this function when:
        - User asks:
            "If I pay ₹X for Y months..."
            "I can only pay this amount for next few months..."
            "What happens if I follow this plan?"

    Do NOT use this function when:
        - User asks about single payment impact
          (use simulate_principal)
        - User asks about current loan state
          (use loan_summary)

    INTERPRETATION RULES FOR LLM:
    -----------------------------
    - Treat outputs as exact values.
    - Do NOT recompute EMI or principal.
    - If loan_closed_early = True:
        clearly tell user loan ends early.
    - If new_emi increases:
        warn user that low payments now increase burden later.
    - Always explain both short-term and long-term impact.

    EXAMPLE OUTPUT:
    ---------------
    {
        "principal_after_custom_period": 85000,
        "remaining_months": 9,
        "new_emi": 9800,
        "loan_closed_early": False,
        "months_to_close": None,
        "closed_in_month_number": None
    }

    SUMMARY:
    --------
    This function enables multi-month repayment planning and helps evaluate
    trade-offs between short-term affordability and long-term cost.
    It is a deterministic financial simulator and must be treated as
    the source of truth for such scenarios.
    """

    r = rate / (12 * 100)

    p = principal_after_n_months(principal, rate, tenure, current_month - 1)

    months_used = 0

    for i in range(months):
        if p <= 0:
            break

        interest = p * r

        if payment >= interest:
            p = round(p - (payment - interest), 2)
        else:
            p = round(p + (interest - payment), 2)

        p = max(p, 0)
        months_used += 1

    remaining_months = tenure - (current_month - 1) - months_used

    if remaining_months <= 0 or p == 0:
        new_emi = 0
    elif remaining_months == 1:
        new_emi = p + (p * r)
    else:
        new_emi = emi(p, rate, remaining_months)

    loan_closed = p == 0

    return {
        "principal_after_custom_period": round(p, 2),
        "remaining_months": remaining_months,
        "new_emi": round(new_emi, 2),
        "loan_closed_early": loan_closed,
        "months_to_close": months_used if loan_closed else None,
        "closed_in_month_number": (current_month + months_used - 1) if loan_closed else None
    }