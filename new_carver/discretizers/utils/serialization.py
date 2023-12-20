"""Sets of helper functions for BaseDiscretizer JSON serialization"""

from json import dumps, loads
from typing import Any, Union

from numpy import floating, inf, integer, isfinite

from .grouped_list import GroupedList


def convert_value_to_base_type(value: Any) -> Union[str, float, int]:
    """Converts a value to python's base types (str, int, float) for JSON serialization.

    Parameters
    ----------
    value : Any
        A value to serialize

    Returns
    -------
    Union[str, float, int]
        Serialized value
    """

    output = value  # str/float/int value
    if not isinstance(value, str) and not isfinite(value):  # numpy.inf value
        output = "numpy.inf"
    elif isinstance(value, integer):  # np.int value
        output = int(value)
    elif isinstance(value, floating):  # np.float value
        output = float(value)

    return output


def convert_value_to_numpy_type(value: Union[str, float, int]) -> Any:
    """Converts a list or a dict of lists values to numpy types for JSON deserialization.

    Parameters
    ----------
    value : Union[str, float, int]
        A value to deserialize

    Returns
    -------
    Any
        Deserialized value
    """
    output = value  # str/float/int value
    if value == "numpy.inf":  # numpy.inf value
        output = inf

    return output


def convert_values_to_base_types(
    iterable: Union[list[Any], dict[str, list[Any]]]
) -> Union[list[Union[str, int, float]], dict[str, list[Union[str, int, float]]]]:
    """Converts a list or a dict of lists values to python's base types (str, int, float) for JSON serialization.

    Parameters
    ----------
    iterable : Union[list[Any], dict[str, list[Any]]]
        List or dict of lists of values to serialize

    Returns
    -------
    Union[list[Union[str, int, float]], dict[str, list[Union[str, int, float]]]]
        List or dict of lists of values serialized
    """
    # list input
    output = None
    if isinstance(iterable, list):
        output = [convert_value_to_base_type(value) for value in iterable]
    # dict input
    elif isinstance(iterable, dict):
        output = {
            convert_value_to_base_type(key): convert_values_to_base_types(values)
            for key, values in iterable.items()
        }

    return output


def convert_values_to_numpy_types(
    iterable: Union[list[Union[str, int, float]], dict[str, list[Union[str, int, float]]]]
) -> Union[list[Any], dict[str, list[Any]]]:
    """Converts a list or a dict of lists values to numpy types for JSON deserialization.

    Parameters
    ----------
    iterable : Union[list[Union[str, int, float]], dict[str, list[Union[str, int, float]]]]
        List or dict of lists of values to deserialize

    Returns
    -------
    Union[list[Any], dict[str, list[Any]]]
        List or dict of lists of values deserialized
    """
    # list input
    output = None
    if isinstance(iterable, list):
        output = [convert_value_to_numpy_type(value) for value in iterable]
    # dict input
    elif isinstance(iterable, dict):
        output = {
            convert_value_to_numpy_type(key): convert_values_to_numpy_types(values)
            for key, values in iterable.items()
        }

    return output


def json_serialize_values_orders(values_orders: dict[str, GroupedList]) -> str:
    """JSON serializes a values_orders

    Parameters
    ----------
    values_orders : dict[str: GroupedList]
        values_orders to serialize

    Returns
    -------
    str
        JSON serialized values_orders
    """

    # converting values_orders to a json
    json_serialized_values_orders = {
        feature: {
            "order": convert_values_to_base_types(order),
            "content": convert_values_to_base_types(order.content),
        }
        for feature, order in values_orders.items()
    }

    return dumps(json_serialized_values_orders)


def json_deserialize_values_orders(json_serialized_values_orders: str) -> dict[str, GroupedList]:
    """JSON serializes a values_orders

    Parameters
    ----------
    json_serialized_values_orders : str
        JSON serialized values_orders

    Returns
    -------
    dict[str: GroupedList]
        values_orders deserialized
    """
    # converting to dict
    json_deserialized_values_orders = loads(json_serialized_values_orders)

    # converting to numpy type
    json_deserialized_values_orders = convert_values_to_numpy_types(json_deserialized_values_orders)

    # converting to GroupedList
    values_orders: dict[str, GroupedList] = {}
    for feature, content in json_deserialized_values_orders.items():
        # getting content from serialized dict
        feature_content: GroupedList = {}
        for value in content["order"]:
            content_key = value
            # float and int dict keys (converted in string by json.dumps)
            if not isinstance(value, str) and isfinite(value):
                content_key = str(value)
            feature_content.update({value: content["content"][content_key]})
        # saving up built GroupedList
        values_orders.update({feature: GroupedList(feature_content)})

    return values_orders
