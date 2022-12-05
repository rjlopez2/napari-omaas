from qtpy import QtCore, QtGui
import numpy as np


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

    processed_data = np.nan_to_num((data - data.min(axis = 0)) / data.max(axis = 0), nan=0.0)
    print(f'computing "local_normal_fun" to image {image.active}')

    return(processed_data)

def split_channels_fun(
    image: "napari.types.ImageData")-> "napari.types.LayerDataTuple":
    """Split the stack every other images. 
    This is needed when doing Calcium and Voltage membrane recording.
    
    Parameters
    ----------
    image : np.ndarray
        The image to be splitted.

    Returns
    -------
    inverted_signal : np.ndarray
        two images for Calcim and Voltage signals respectively?"""
    
    data = image.active.data
    ch_1 = data[::2,:,:]
    ch_2 = data[1::2,:,:]
    print(f'applying "split_channels" to image {image.active}')
    return [ch_1, ch_2]
