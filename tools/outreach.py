BUDGET_ALLOCATIONS = {"SMB": 500, "MIDMARKET": 1000, "ENTERPRISE": 5000}


def get_segments():
    """
    Returns the list of budget segments.
    """
    return list(BUDGET_ALLOCATIONS.keys())


def get_budget_allocations():
    """
    Returns the budget allocations for each segment.
    """
    return BUDGET_ALLOCATIONS


def set_budget_allocations(segment, amount):
    """
    Sets the budget allocation for the given segment.
    """
    if segment not in BUDGET_ALLOCATIONS:
        raise ValueError(f"Invalid key: {segment}")
    BUDGET_ALLOCATIONS[segment] = amount
