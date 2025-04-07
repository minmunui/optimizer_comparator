def zero_formatting(number: int) -> str:
    """
    Formats a number with leading zeros to ensure it is always 3 digits long.
    """
    return str(number).zfill(3)