__version__ = "1.0.2"

from ._sample_data import make_sif_sample_data, make_folder_sample_data, make_folder_sample_data_dual
from ._widget import OMAAS, example_magic_widget
from ._writer import write_multiple, write_single_image
from ._reader import napari_get_reader

__all__ = (
    "write_single_image",
    "write_multiple",
    "make_sif_sample_data",
    "make_folder_sample_data",
    "make_folder_sample_data_dual",
    "OMAAS",
    "example_magic_widget",
    "napari_get_reader",
)
