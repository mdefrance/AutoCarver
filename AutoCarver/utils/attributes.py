""" function to get boolean attributes from kwargs """


def get_attribute(kwargs: dict, attribute: str, default_value: any = None) -> any:
    """returns value from kwargs"""

    # getting attribute value
    value = kwargs.get(attribute)

    # attributing default value if not found
    if value is None:
        value = default_value

    return value


def get_bool_attribute(kwargs: dict, attribute: str, default_value: bool) -> bool:
    """returns value from kwargs whilst checking for bool type"""

    # getting attribute value
    value = get_attribute(kwargs, attribute, default_value)

    # checking for correct type
    if not isinstance(value, bool):
        raise ValueError(f"{attribute} should be a bool, not {type(value)}")

    return value
