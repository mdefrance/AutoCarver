"""Sets of helper functions for BaseDiscretizer JSON serialization"""

from json import dumps, loads
from typing import Any, Union

from numpy import floating, inf, integer, isfinite

from ...config import Constants
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
        output = Constants.INF
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
    if value == Constants.INF:  # numpy.inf value
        output = inf

    return output


def convert_values_to_base_types(
    iterable: Union[list[Any], dict[str, list[Any]]]
) -> Union[list[Union[str, int, float]], dict[str, list[Union[str, int, float]]]]:
    """Converts a list or a dict of lists values to python's base types (str, int, float)
    for JSON serialization.

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


def json_serialize_feature(feature: dict) -> str:
    """JSON serializes a feature's dict"""

    # serializing values
    values = feature.get("values")
    feature.update({"values": convert_values_to_base_types(values)})

    # serializing content
    content = feature.get("content")
    feature.update({"content": dumps(convert_values_to_base_types(content))})

    return feature


def json_serialize_combination(combination: dict) -> str:
    """JSON serializes a combination

    Parameters
    ----------
    combination : dict
        combination to serialize

    Returns
    -------
    dict
        JSON serialized combination
    """

    # converting combination to a json
    json_serialized_combination = {key: value for key, value in combination.items()}

    # converting combination values to a json
    json_serialized_combination.update(
        {
            "combination": [
                [convert_value_to_base_type(value) for value in modality]
                for modality in json_serialized_combination["combination"]
            ]
        }
    )

    return json_serialized_combination


def json_deserialize_content(json_serialized_feature: dict) -> dict:
    """JSON deserializes a content

    Parameters
    ----------
    json_serialized_content : str
        JSON serialized content

    Returns
    -------
    dict[str: GroupedList]
        content deserialized
    """

    # reading values and content from json
    values = json_serialized_feature.get("values")
    content = json_serialized_feature.get("content")

    # converting to numpy types
    values = convert_values_to_numpy_types(values)
    content = convert_values_to_numpy_types(loads(content))

    # cehcking for values
    if values is not None:
        # fixing json dumping with string keys
        for value in values:
            content_key = value

            # float and int dict keys (converted in string by json.dumps)
            if not isinstance(value, str) and isfinite(value):
                content_key = str(value)

            # updating
            content.update({value: content.pop(content_key)})

        # converting to grouped list
        return GroupedList(content)
