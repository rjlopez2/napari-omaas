"""
This module is an example of a barebones sample data provider for napari.

It implements the "sample data" specification.
see: https://napari.org/stable/plugins/guides.html?#sample-data

Replace code below according to your needs.
"""
from __future__ import annotations

import zipfile
import os
import requests
from pathlib import Path
from napari.types import LayerData
from napari.utils import progress
import tempfile  # For creating a temporary directory
from napari_omaas._reader import reader_sif_function

# Define constants for the sample data URLs
SIF_SAMPLE_URL = "https://physiologie.unibe.ch/~odening/group/data/4viewpanoramicstackimage.zip"  # Existing .sif dataset
FOLDER_SAMPLE_URL = "https://physiologie.unibe.ch/~odening/group/data/single_illumination_spool_data_sample.zip"  # New dataset (folder inside .zip)
FOLDER_SAMPLE_URL_DUAL = "https://physiologie.unibe.ch/~odening/group/data/dual_illumination_spool_data_sample.zip"  # New dataset (folder inside .zip)

# Use a temporary directory for the session
SESSION_TEMP_DIR = Path(tempfile.gettempdir()) / "napari_omaas_sample_data"
SESSION_TEMP_DIR.mkdir(exist_ok=True)

SIF_ZIP_FILE = SESSION_TEMP_DIR / "sif_sample_freiburg.zip"
FOLDER_ZIP_FILE = SESSION_TEMP_DIR / "samplefile_20241110_12h-20m-25.zip"
FOLDER_ZIP_FILE_DUAL = SESSION_TEMP_DIR / "folder_sample_dual.zip"


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
    Extract a .zip file into a session temporary directory, 
    only if it hasn't already been unzipped.

    Parameters
    ----------
    zip_path : Path
        Path to the .zip file.

    Returns
    -------
    Path
        Path to the extracted folder.
    """
    
    extracted_folder = SESSION_TEMP_DIR / zip_path.stem
    # Check if the folder already exists
    if extracted_folder.exists():
        print(f"Folder already exists: {extracted_folder}")
        return extracted_folder
    
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as z:
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
    
    image = reader_sif_function(str(sif_path))
    data, add_kwargs, layer_type = image[0]
    add_kwargs["name"] = sif_path.stem

    return [(data, add_kwargs, layer_type)]


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

    image = reader_sif_function(str(folder_path))
    data, add_kwargs, layer_type = image[0]
    add_kwargs["name"] = folder_path.stem

    return [(data, add_kwargs, layer_type)]


def make_folder_sample_data_dual():
    """
    Create a sample dataset for a folder that contains a spool dataset.

    Returns
    -------
    data : list of LayerData tuples
    """
    folder_path = ensure_sample_data(FOLDER_SAMPLE_URL_DUAL, FOLDER_ZIP_FILE_DUAL)
    spool_folder = os.listdir(folder_path)
    if not spool_folder:
        raise FileNotFoundError("No folder found in the ziped dataset.")
    
    folder_path = folder_path / spool_folder[0] # assume there are only one file in the zipped file.

    image = reader_sif_function(str(folder_path))
    data, add_kwargs, layer_type = image[0]
    add_kwargs["name"] = folder_path.stem

    return [(data, add_kwargs, layer_type)]



