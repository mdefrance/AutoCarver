""" checks for installed extra dependencies"""


def has_idisplay() -> bool:
    """Returns True if the IPython.display module is available"""

    # trying to import extra dependencies
    try:
        from IPython.display import display_html

        _ = display_html
    except ImportError:
        _has_idisplay = False
    else:
        # trying to import other dependencies needed
        try:
            import ipykernel
            import jinja2
            import matplotlib

            _, _, _ = ipykernel, jinja2, matplotlib
        except ImportError:
            _has_idisplay = False
            print(
                "WARNING: IPython.display is available, but other dependencies are missing. "
                "Please install them using pip install AutoCarver[jupyter]."
            )
        else:
            _has_idisplay = True

    return _has_idisplay
