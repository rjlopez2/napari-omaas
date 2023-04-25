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
import requests, zipfile, io, shutil

THIS_DIR = os.path.dirname(__file__)
sys.path.append(THIS_DIR + '/../../img/' )

DATA_DIR = THIS_DIR + '/../../img/'
filename = '11h-27m-32s.sif'
file_ext = [".zip"]
    

# sample_img_dir = THIS_DIR + '/../../img/' 

# ext_link = "/Users/rubencito/Library/CloudStorage/OneDrive-UniversitaetBern/Bern/Odening_lab/OM/napari-omaas_things/data_samples/4view_stacks/11h-27m-32s.sif"
# ext_link = "https://unibe365-my.sharepoint.com/:u:/g/personal/ruben_lopez_unibe_ch/EYzTZlFA06VFvwHH_qBsaE4BBMYZ3Z3GA7ZiMOuNlhb_cw?e=1mW9gt" #original
# ext_link = 'https://unibe365-my.sharepoint.com/:u:/g/personal/ruben_lopez_unibe_ch/EX3rR1ZlV-FFqU4lIRP9xdoBHp0rHjNSMBAFoJb4L7OfEQ?e=bAPjDN' # zipped
# ext_link = 'https://unibe365-my.sharepoint.com/:u:/r/personal/ruben_lopez_unibe_ch/Documents/Bern/Odening_lab/OM/napari-omaas_things/data_samples/4view_stacks/11h-27m-32s.zip'
ext_link = 'https://www.dropbox.com/s/9g1m4ptvl6waevv/11h-27m-32s.zip?dl=1' # dropbox link

if not os.path.exists(DATA_DIR + filename):
    r = requests.get(ext_link)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall(DATA_DIR)
    z_file_path = os.path.abspath(os.path.join(DATA_DIR, z.infolist()[0].filename))
    os.rename(z_file_path, DATA_DIR + os.path.basename(z.infolist()[0].filename))
    shutil.rmtree(DATA_DIR + "Users")

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