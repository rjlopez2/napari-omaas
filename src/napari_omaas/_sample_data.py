"""
This module is an example of a barebones sample data provider for napari.

It implements the "sample data" specification.
see: https://napari.org/stable/plugins/guides.html?#sample-data

Replace code below according to your needs.
"""
from __future__ import annotations

# import numpy
# from napari_sif_reader import napari_get_reader

# import os
# import sys
# import requests, zipfile, io, shutil

# THIS_DIR = os.path.dirname(__file__)
# sys.path.append(THIS_DIR + '/../../img/' )

# DATA_DIR = THIS_DIR + '/../../img/'
# filename = '11h-27m-32s.sif'
# file_ext = [".zip"]
    

# # sample_img_dir = THIS_DIR + '/../../img/' 

# ext_link = 'https://physiologie.unibe.ch/~odening/group/data/zipped/4viewpanoramicstackimage.zip' # institute link for the "4 view panoramic stack image" dataset

# if not os.path.exists(DATA_DIR + filename):
#     r = requests.get(ext_link)
#     z = zipfile.ZipFile(io.BytesIO(r.content))
#     z.extractall(DATA_DIR)
#     z_file_path = os.path.abspath(os.path.join(DATA_DIR, z.infolist()[0].filename))
#     os.rename(z_file_path, DATA_DIR + os.path.basename(z.infolist()[0].filename))
#     shutil.rmtree(DATA_DIR + "Users")

# for ext in file_ext:
#     if not os.path.exists(DATA_DIR + ext):
        
        
        
        



def make_sample_data():

    """Generates an image"""
    path = DATA_DIR + '11h-27m-32s.sif'
    sif_reader_fun = napari_get_reader(path)
    # img_layer = sif_reader_fun(path)
    img_layer = sif_reader_fun(path)
    img_layer[0][1]["name"] = 'TestImg_11h-27m-32s'

    return img_layer