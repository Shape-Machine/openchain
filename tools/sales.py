import random

SALES_TARGETS = {"SMB": 100, "MIDMARKET": 500, "ENTERPRISE": 1000}


def get_segments():
    """
    Returns the list of sales segments.
    """
    return list(SALES_TARGETS.keys())


def get_sales_targets():
    """
    Returns the sales targets for each segment.
    """
    return SALES_TARGETS


def get_sales_actuals(segment):
    """
    Returns the sales actuals for the given segment.
    """
    random_min = 0
    random_max = 0

    if segment == "SMB":
        random_min = 50
        random_max = 150
    elif segment == "MIDMARKET":
        random_min = 350
        random_max = 750
    elif segment == "ENTERPRISE":
        random_min = 700
        random_max = 1300
    else:
        raise ValueError(f"Invalid segment: {segment}")

    return random.randint(random_min, random_max)
