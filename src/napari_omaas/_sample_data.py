"""
This module is an example of a barebones sample data provider for napari.

It implements the "sample data" specification.
see: https://napari.org/stable/plugins/guides.html?#sample-data

Replace code below according to your needs.
"""
from __future__ import annotations

import numpy
from napari_sif_reader import napari_get_reader

import os
import sys

THIS_DIR = os.path.dirname(__file__)
# sys.path.append(THIS_DIR + '/../../img/' )
DATA_DIR = THIS_DIR + '/../../img/'

# sample_img_dir = THIS_DIR + '/../../img/' 


def make_sample_data():

    """Generates an image"""
    path = DATA_DIR + '11h-27m-32s.sif'
    sif_reader_fun = napari_get_reader(path)
    img_layer = sif_reader_fun(path)
    img_layer[0][1]["name"] = 'TestImg_11h-27m-32s'

    return img_layer