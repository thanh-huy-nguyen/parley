"""Exceptions and warnings raised by the package."""


class ParleyError(Exception):
    """Base class of every error the package raises on purpose."""


class InputError(ParleyError, ValueError):
    """An image, mask, click or audio input that cannot be interpreted."""


class ImageTooLargeError(InputError):
    """An image or mask above the pixel limit in `parley.io.MAX_PIXELS`."""


class WeightsNotFoundError(ParleyError, FileNotFoundError):
    """No weights file at any of the lookup locations."""


class WeightsHashMismatchError(ParleyError):
    """A weights file whose SHA-256 differs from the released value."""


class UnsupportedCheckpointError(ParleyError):
    """A checkpoint whose architecture this package does not implement."""


class SessionStateError(ParleyError):
    """A session state file that cannot be used with this editor."""


class ParleyInputWarning(UserWarning):
    """An input that is accepted but deserves attention. Use `warnings.simplefilter("error", ParleyInputWarning)` for
    strict checking.
    """
