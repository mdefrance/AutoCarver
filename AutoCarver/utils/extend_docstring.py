""" tools for docstrings"""


class extend_docstring:
    """Used to extend a Child's method docstring with the Parent's method docstring"""

    def __init__(self, method):
        self.doc = method.__doc__

    def __call__(self, function):
        if self.doc is not None:
            doc = function.__doc__
            function.__doc__ = self.doc
            if doc is not None:
                function.__doc__ = doc + function.__doc__
        return function
