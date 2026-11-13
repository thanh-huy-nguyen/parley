"""Parley: a speech-directed segmentation mask editor. Research prototype, not a medical device.

from parley import ParleyEditor
editor = ParleyEditor(mode="base")
mask = editor.begin_image("image.png", "mask.png")
result = editor.edit(text="pull the top in a little")
"""

__version__ = "0.1.0"
__all__ = ["ParleyEditor", "EditResult", "describe_program", "Sam2ClickEditor", "__version__"]


def __getattr__(name):
    if name in ("ParleyEditor", "EditResult", "describe_program"):
        from . import editor

        return getattr(editor, name)
    if name == "Sam2ClickEditor":
        from .sam2_click import Sam2ClickEditor

        return Sam2ClickEditor
    raise AttributeError(name)
