"""
This module is an example of a barebones numpy reader plugin for napari.

It implements the Reader specification, but your plugin may choose to
implement multiple readers or even other plugin contributions. see:
https://napari.org/stable/plugins/guides.html?#readers
"""
import sif_parser
import numpy as np
import os
from glob import glob 
from warnings import warn
# import napari_omaas as o

SUPPORTED_IMAGES = ".sif", ".SIF", ".sifx", ".SIFX"

def napari_get_reader(path):
    """A basic implementation of a Reader contribution.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    function or None
        If the path is a recognized format, return a function that accepts the
        same path or list of paths, and returns a list of layer data tuples.
    """
    if isinstance(path, list):
        # reader plugins may be handed single path, or a list of paths.
        # if it is a list, it is assumed to be an image stack...
        # so we are only going to look at the first file.
        path = path[0]
    
    if isinstance(path, str) and os.path.isdir(path):
        sifx_file = glob(os.path.join(path, "*.sifx"))

        if len(sifx_file) > 0:
            # sifx_file[0].endswith(".sifx"):
            return reader_function
        else:
            warn(" No file found in current directory with extension '*.sifx'")
            return None

        
    # if we know we cannot read the file, we immediately return None.
    if not any(path.lower().endswith(ext) for ext in SUPPORTED_IMAGES):
        return None
    


    # otherwise we return the *function* that can read ``path``.
    return reader_function

def reader_function(path):
    """Take a path or list of paths and return a list of LayerData tuples.

    Readers are expected to return data as a list of tuples, where each tuple
    is (data, [add_kwargs, [layer_type]]), "add_kwargs" and "layer_type" are
    both optional.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    layer_data : list of tuples
        A list of LayerData tuples where each tuple in the list contains
        (data, metadata, layer_type), where data is a numpy array, metadata is
        a dict of keyword arguments for the corresponding viewer.add_* method
        in napari, and layer_type is a lower-case string naming the type of
        layer. Both "meta", and "layer_type" are optional. napari will
        default to layer_type=="image" if not provided
    """
    # handle both a string and a list of strings
    # paths = [path] if isinstance(path, str) else path
    # load all files into array
    # arrays = [np.load(_path) for _path in paths]
    # stack arrays into single array

    if os.path.isdir(path):
        # is_multithreading = o.fast_loading.isChecked()
        data, info = sif_parser.np_spool_open(path, multithreading=False)
        
    elif  os.path.isfile(path):
        data, info = sif_parser.np_open(path)
    else:
        return warn(f"The path or file porvide is not support or not valid")

    data = np.flip(data, axis=(1))
    metadata = {key: val for key, val in info.items() if (not key.startswith("timestamp") and (not key.startswith("tile"))) }
    metadata['CurrentFileSource'] = path

    # reorder array so it's the same order dimention represented as in the matlab app
    # metadata = get_custome_metadata_func(info)
    # skip first two frames to avoid peak artifact on first frame?
    # if stack contain more than 3 images will remove the first 2 because of artefact
    # if not metadata["NumberOfFrames"] is None and metadata["NumberOfFrames"] >= 3:
    #     data = data[2:,...]
 
  
    # optional kwargs for the corresponding viewer.add_* method
    add_kwargs = {
        "colormap" : "turbo",
        # "gamma" : 0.15,
        "metadata": metadata
    }

    layer_type = "image"  # optional, default is "image"
    return [(data, add_kwargs, layer_type)]
