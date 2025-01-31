""" tools for docstrings"""


class extend_docstring:
    """Used to extend a Child's method docstring with the Parent's method docstring"""

    def __init__(self, method, append=True):
        self.doc = method.__doc__
        self.append = append

    def __call__(self, function):
        if self.doc is not None:
            doc = function.__doc__
            function.__doc__ = self.doc
            if doc is not None:
                if self.append:
                    function.__doc__ = function.__doc__ + doc
                else:
                    function.__doc__ = doc + function.__doc__
        return function
