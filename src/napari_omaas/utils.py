from qtpy import QtCore, QtGui
# import numpy as np


# functions

def invert_signal(
    image: "napari.types.ImageData")-> "napari.types.LayerDataTuple":
    """Invert signal fluorescence values. This is usefull to properly visulaize
    AP signals from inverted traces.

    Parameters
    ----------
    image : np.ndarray
        The image to be filtered.

    Returns
    -------
    inverted_signal : np.ndarray
        The image with inverted fluorescence values
    """
    data = image.active.data
    processed_data = data.max(axis = 0) - data
    layer_data  = (
        processed_data,
        {

        },
        "image"
        
    )
    print(f'computing "invert_signal" to image {image.active}')
    # print (f'computing "invert_signal" to image colormap='magma' ndim: {image.active.data.ndim}')
    # return(inverted_data, dict(name= "lalala"), "image") 
    # return(layer_data)
    return(processed_data)


def local_normal_fun(
    image: "napari.types.ImageData")-> "napari.types.LayerDataTuple":
    """Invert signal fluorescence values. This is usefull to properly visulaize
    AP signals from inverted traces.

    Parameters
    ----------
    image : np.ndarray
        The image to be filtered.

    Returns
    -------
    inverted_signal : np.ndarray
        The image with inverted fluorescence values
    """
    data = image.active.data

    processed_data = (data - data.min(axis = 0)) / data.max(axis = 0)
    print(f'computing "local_normal_fun" to image {image.active}')

    return(processed_data)
