__version__ = "0.1.2"

from ._sample_data import make_sample_data
from ._widget import OMAAS, example_magic_widget
from ._writer import write_multiple, write_single_image

__all__ = (
    "write_single_image",
    "write_multiple",
    "make_sample_data",
    "OMAAS",
    "example_magic_widget",
)
