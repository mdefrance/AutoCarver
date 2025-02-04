""" tools for docstrings"""

from typing import Callable


class extend_docstring:
    """Used to extend a Child's method docstring with the Parent's method docstring"""

    def __init__(self, method: Callable, append: bool = True, exclude: list[str] = None) -> None:
        self.doc = method.__doc__
        self.append = append
        self.exclude = exclude or []

    @staticmethod
    def split_sections(docstring: str) -> dict[str, list[str]]:
        """Split docstring into sections by double newlines"""
        # splitting parameters
        parameters = docstring.split("\n\n")

        # removing section names
        sections = {}
        current_section = ""
        section_params = []
        current_param = ""
        for param in parameters:
            # checking for section name
            if "------" in param:
                # storing current section
                if current_param:
                    section_params.append(current_param)
                    sections[current_section] = section_params
                    current_param = ""
                # resetting current section
                current_section = param
                section_params = []

            # it is a parameter
            else:
                # checking for parameter name
                if " : " in param:
                    # storing current param if any
                    if current_param:
                        section_params.append(current_param)

                    # resetting current param
                    current_param = param

                # adding to current param
                else:
                    if current_param:
                        current_param = "\n\n".join([current_param, param])
                    else:
                        current_param = param
        # storing last section
        if current_param:
            section_params.append(current_param)
            sections[current_section] = section_params

        return sections

    def filter_sections(self, sections: dict[str, list[str]]) -> str:
        """Filter sections based on parameters to exclude"""

        # removing parameters to exclude and converting back to docstring
        docstring = ""
        for section, section_params in sections.items():
            # adding section name
            docstring += section + "\n\n"

            # iterating over section parameters
            for param in section_params:
                # checking if parameter should be excluded
                if not any(exclude + " :" in param for exclude in self.exclude):
                    # adding parameters
                    docstring += param + "\n\n"

        return docstring

    def process_docstring(self) -> str:
        """Process the docstring to exclude specified parameters"""
        if not self.doc:
            return ""
        sections = self.split_sections(self.doc)
        filtered_sections = self.filter_sections(sections)
        return filtered_sections

    def __call__(self, function: Callable) -> Callable:
        """Extend the docstring of the function with the one from the method"""
        if self.doc is not None:
            self.doc = self.process_docstring()
            doc = function.__doc__
            function.__doc__ = self.doc
            if doc is not None:
                if self.append:
                    function.__doc__ = function.__doc__ + doc
                else:
                    function.__doc__ = doc + function.__doc__
        return function
