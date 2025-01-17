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
        
        
        
        



# def make_sample_data():

#     """Generates an image"""
#     path = DATA_DIR + '11h-27m-32s.sif'
#     sif_reader_fun = napari_get_reader(path)
#     # img_layer = sif_reader_fun(path)
#     img_layer = sif_reader_fun(path)
#     img_layer[0][1]["name"] = 'TestImg_11h-27m-32s'

#     return img_layer


# import os
# import requests
# import zipfile
# import tempfile
# from pathlib import Path
# from napari_omaas._reader import reader_sif_function

# # Registry of sample datasets: keys, URLs, and filenames
# SAMPLE_DATA_REGISTRY = {
#     "heart_sample_1": {
#         "url": "https://physiologie.unibe.ch/~odening/group/data/4viewpanoramicstackimage.zip",
#         "filename": "4viewpanoramicstackimage.sif",
#     },
#     "heart_sample_2": {
#         "url": "https://physiologie.unibe.ch/~odening/group/data/zipped/4viewpanoramicstackimage.zip",
#         "filename": "sample_data2.sif",
#     },
#     "heart_sample_3": {
#         "url": "https://physiologie.unibe.ch/~odening/group/data/zipped/4viewpanoramicstackimage.zip",
#         "filename": "sample_data3.sif",
#     },
#     "heart_sample_4": {
#         "url": "https://physiologie.unibe.ch/~odening/group/data/zipped/4viewpanoramicstackimage.zip",
#         "filename": "sample_data4.sif",
#     },
#     "heart_sample_5": {
#         "url": "https://physiologie.unibe.ch/~odening/group/data/zipped/4viewpanoramicstackimage.zip",
#         "filename": "sample_data5.sif",
#     },
# }

# # Local cache path for efficiency
# CACHE_DIR = Path(tempfile.gettempdir()) / "napari_omaas_sample_data"
# CACHE_DIR.mkdir(parents=True, exist_ok=True)


# def download_and_extract_sample_data(key):
#     """
#     Downloads and extracts the sample .sif data for a given key if not already present.

#     Parameters
#     ----------
#     key : str
#         The key of the dataset to download.

#     Returns
#     -------
#     str
#         Path to the extracted .sif file.
#     """
#     if key not in SAMPLE_DATA_REGISTRY:
#         raise ValueError(f"Invalid key: {key}. Valid keys are {list(SAMPLE_DATA_REGISTRY.keys())}")

#     # Get the dataset information
#     dataset_info = SAMPLE_DATA_REGISTRY[key]
#     url = dataset_info["url"]
#     sif_filename = dataset_info["filename"]

#     # Define cache file paths
#     zip_file_path = CACHE_DIR / f"{key}.zip"
#     sif_file_path = CACHE_DIR / sif_filename

#     # Check if the .sif file already exists
#     if sif_file_path.exists():
#         print(f"Sample data already exists: {sif_file_path}")
#         return str(sif_file_path)

#     # Download the zip file if it doesn't exist
#     if not zip_file_path.exists():
#         print(f"Downloading sample data '{key}' from {url}...")
#         response = requests.get(url, stream=True)
#         response.raise_for_status()
#         with open(zip_file_path, "wb") as f:
#             for chunk in response.iter_content(chunk_size=8192):
#                 f.write(chunk)

#     # Extract the zip file
#     print(f"Extracting sample data '{key}' to {CACHE_DIR}...")
#     with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
#         zip_ref.extractall(CACHE_DIR)

#     # Check for the .sif file
#     if sif_file_path.exists():
#         return str(sif_file_path)
#     else:
#         raise FileNotFoundError(f"Expected .sif file '{sif_filename}' not found in the zip archive.")


# def make_sample_data(*args, **kwargs):
#     """
#     Generates sample data for Napari.

#     Parameters
#     ----------
#     key : str
#         The key of the dataset to load.
#     """
#     key = kwargs.get('key')
#     if not key:
#         raise ValueError("The 'key' parameter is required but was not provided.")
#     # Proceed to load the sample data based on the key
#     if key == "heart_sample_1":
#         sif_file_path = download_and_extract_sample_data(key)
#         return reader_sif_function(sif_file_path)
#     else:
#         raise ValueError(f"Unknown sample data key: {key}")


# import os
# import zipfile
# import requests
# import sif_parser
# import numpy as np
# from pathlib import Path
# from tempfile import TemporaryDirectory
# from napari.types import LayerData
# from napari.utils import progress  # Import napari's progress module

# TEMP_DIR = TemporaryDirectory()
# TEMP_PATH = Path(TEMP_DIR.name)
# SAMPLE_ZIP_FILE = TEMP_PATH / "sample.zip"
# SAMPLE_SIF_FILE = TEMP_PATH / "sample.sif"

# SAMPLE_DATA_URL = "https://physiologie.unibe.ch/~odening/group/data/4viewpanoramicstackimage.zip"  # Replace with your actual URL
# # SAMPLE_DATA_DIR = Path.home() / ".napari_omaas_sample_data"
# SAMPLE_DATA_FILE = TEMP_PATH / "4viewpanoramicstackimage.sif"

# def download_sample_data_heart_1():
#     """Download the sample data if it doesn't already exist."""
#     SAMPLE_DATA_DIR.mkdir(parents=True, exist_ok=True)
#     zip_path = SAMPLE_DATA_DIR / "sample.zip"

#     if not SAMPLE_DATA_FILE.exists():
#         print("Downloading sample data...")
#         response = requests.get(SAMPLE_DATA_URL, stream=True)
#         response.raise_for_status()
        
#         total_size = int(response.headers.get("content-length", 0))
#         with open(zip_path, "wb") as f, progress(total=total_size, desc="Downloading") as pbar:
#             for chunk in response.iter_content(chunk_size=8192):
#                 f.write(chunk)
#                 pbar.update(len(chunk))

#         # Extract the zip file
#         with zipfile.ZipFile(zip_path, "r") as z:
#             z.extractall(SAMPLE_DATA_DIR)
#         os.remove(zip_path)

#     return SAMPLE_DATA_FILE

# def make_sample_data():
#     """
#     Return sample data for napari.

#     Returns
#     -------
#     data : list of LayerData tuples
#     """
#     sif_path = download_sample_data_heart_1()
#     data, info = sif_parser.np_open(sif_path)

#     metadata = {key: val for key, val in info.items() if not key.startswith("timestamp")}
#     metadata['source'] = str(sif_path)

#     add_kwargs = {"colormap": "turbo", 
#                   "metadata": metadata, 
#                   "name": sif_path.name.split('.')[0]} # need to exploitaly set the name here otherwise get not assigned by default.
#     return [(data, add_kwargs, "image")]


import zipfile
import os
import requests
from pathlib import Path
from napari.types import LayerData
from napari.utils import progress
import tempfile  # For creating a temporary directory
import sif_parser  # Assuming files in the folder are .sif files or compatible with this plugin

# Define constants for the sample data URLs
SIF_SAMPLE_URL = "https://physiologie.unibe.ch/~odening/group/data/4viewpanoramicstackimage.zip"  # Existing .sif dataset
FOLDER_SAMPLE_URL = "https://physiologie.unibe.ch/~odening/group/data/single_illumination_spool_data_sample.zip"  # New dataset (folder inside .zip)

# Use a temporary directory for the session
SESSION_TEMP_DIR = Path(tempfile.gettempdir()) / "napari_omaas_sample_data"
SESSION_TEMP_DIR.mkdir(exist_ok=True)

SIF_ZIP_FILE = SESSION_TEMP_DIR / "sif_sample.zip"
FOLDER_ZIP_FILE = SESSION_TEMP_DIR / "folder_sample.zip"


def download_file(url, dest_path):
    """Download a file from a URL to a specific path with progress tracking."""
    print(f"Downloading file from {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    with open(dest_path, "wb") as f, progress(total=total_size, desc="Downloading") as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))

    print(f"File downloaded to {dest_path}")


def extract_zip_to_temp(zip_path):
    """
    Extract a .zip file into a session temporary directory.

    Parameters
    ----------
    zip_path : Path
        Path to the .zip file.

    Returns
    -------
    Path
        Path to the extracted folder.
    """
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as z:
        extracted_folder = SESSION_TEMP_DIR / zip_path.stem
        z.extractall(extracted_folder)
    print(f"Extracted to {extracted_folder}")
    return extracted_folder


def ensure_sample_data(zip_url, zip_path):
    """Ensure the sample data is downloaded and extracted."""
    if not zip_path.exists():
        download_file(zip_url, zip_path)
    extracted_path = extract_zip_to_temp(zip_path)
    return extracted_path


def make_sif_sample_data():
    """
    Create a sample dataset for a single .sif file.

    Returns
    -------
    data : list of LayerData tuples
    """
    sif_folder = ensure_sample_data(SIF_SAMPLE_URL, SIF_ZIP_FILE)
    sif_files = list(sif_folder.glob("*.sif"))
    if not sif_files:
        raise FileNotFoundError("No .sif files found in the sample dataset.")

    sif_path = sif_files[0] # assume there are only one file in the zipped file.
    data, info = sif_parser.np_open(sif_path)

    metadata = {key: val for key, val in info.items() if not key.startswith("timestamp")}
    metadata["source"] = str(sif_path)

    add_kwargs = {
        "colormap": "turbo",
        "metadata": metadata,
        "name": sif_path.stem,
    }
    return [(data, add_kwargs, "image")]


def make_folder_sample_data():
    """
    Create a sample dataset for a folder that contains a spool dataset.

    Returns
    -------
    data : list of LayerData tuples
    """
    folder_path = ensure_sample_data(FOLDER_SAMPLE_URL, FOLDER_ZIP_FILE)
    spool_folder = os.listdir(folder_path)
    if not spool_folder:
        raise FileNotFoundError("No folder found in the ziped dataset.")
    folder_path = folder_path / spool_folder[0] # assume there are only one file in the zipped file.

    data, info = sif_parser.np_spool_open(str(folder_path))

    metadata = {key: val for key, val in info.items() if not key.startswith("timestamp")}
    metadata["source"] = str(spool_folder)

    add_kwargs = {
        "colormap": "turbo",
        "metadata": metadata,
        "name": folder_path.stem,
    }
    return [(data, add_kwargs, "image")]
