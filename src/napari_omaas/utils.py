from qtpy import QtCore, QtGui
import numpy as np
from skimage.filters import gaussian, threshold_triangle, median, rank
from skimage.morphology import disk
from skimage.registration import optical_flow_ilk
from skimage.transform import warp
from skimage import morphology
from skimage import segmentation
import warnings
from napari.layers import Image

# from numba import jit, prange
from scipy import signal, ndimage
# functions

from napari.types import ImageData, ShapesData
# def detect_spots(
#     image: "napari.types.ImageData",
#     high_pass_sigma: float = 2,
#     spot_threshold: float = 0.01,
#     blob_sigma: float = 2
# ) -> "napari.types.LayerDataTuple":

def invert_signal(
    image: "napari.types.ImageData"
    )-> "napari.types.LayerDataTuple":

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
    # processed_data = data.max(axis = 0) - data
    # processed_data = np.nanmax(data, axis=0) - data
    # processed_data = paralele_inv_signal(data)
    
    print(f'computing "invert_signal" to image {image.active}')
    # print (f'computing "invert_signal" to image colormap='magma' ndim: {image.active.data.ndim}')
    # return(inverted_data, dict(name= "lalala"), "image") 
    # return(layer_data)
    # norm_data = np.nanmax(data, axis=0) - data

    # layer_data  = (
    #     norm_data,
    #     {
    #         'name': 'My Image', 
    #         'colormap': 'red'
    #     },
    #     "image"
        
    # )
    # return layer_data
    return np.nanmax(data, axis=0) - data
    # return Image(image.active.data.max(axis = 0) - image.active.data)


def local_normal_fun(
    image: "napari.types.ImageData")-> "napari.types.ImageData":

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

    print(f'computing "local_normal_fun" to image {image.active}')

    return (data - np.nanmin(data, axis = 0)) / np.nanmax(data, axis=0)

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
    ch_1, ch_2 : list 
        two np.ndarray images for Calcim and Voltage signals respectively?"""
    
    data = image.active.data
    ch_1 = data[::2,:,:]
    ch_2 = data[1::2,:,:]
    print(f'applying "split_channels" to image {image.active}')
    return [ch_1, ch_2]

def segment_heart_func( 
    image: "napari.types.ImageData",
    sigma = 2, 
    nbins = 100, 
    holes_siz = 500, 
    expa_value = 25,
    eros_value = 25)-> "napari.types.LayerDataTuple":

    """
    Subtract background from image.

    Parameters
    ----------
    image : np.ndarray
        The image to be back subtracted.
    
    sigma: int
        Magintud of gaussina filer kernel, default to 2.
    
    nbins: int
        Magnitud of bins for the threshold_triangle method. Default to 100.
    
    holes_siz: int

    expa_value = int

    eros_value = int

    
    Returns
    -------
    img_no_background : np.ndarray
       Image with backgrounf subtracted.

    """

    data = image.active.data
    
    # 1. apply a median projection on the 0 (time) axis
    projet_img = np.median(data, axis = 0, keepdims= True)
    # 2. apply gaussina filter
    projet_img = gaussian(projet_img, sigma)
    # 3. Threshold
    threshold = threshold_triangle(projet_img, nbins)
    # 4. create mask
    mask = projet_img > threshold
    # 5. remove holes
    mask = morphology.remove_small_holes(mask, holes_siz)
    # 6. remove small objects
    mask = morphology.remove_small_objects(mask, holes_siz)
    # 7. expand mask
    mask = segmentation.expand_labels(mask, expa_value)
    # 8. erode mask
    mask = morphology.erosion(mask[0, ...], footprint=disk(eros_value))
    # 9. expand to the 0 axis
    mask = np.tile(mask, (data.shape[0], 1, 1))
    
    # 10. subtracts background from original img
    # raw_img_stack_nobg = data.copy()
    # raw_img_stack_nobg[~mask] = 0
    
    print(f'applying "segment_heart_func" to image {image.active}')
    # return raw_img_stack_nobg
    return mask

def apply_gaussian_func (image: "napari.types.ImageData",
    sigma, kernel_size = 3)-> "Image":

    """
    Apply Gaussian filter to selected image.

    Parameters
    ----------
    image : np.ndarray
        The image to be back subtracted.
    
    sigma: int
        Magintud of gaussina filer kernel, default to 2.
    
    
    Returns
    -------
    out_img : np.ndarray
       Smoothed Image with Gaussian filter.

    """



    data = image.active.data
    out_img = np.empty_like(data)

    print(f'applying "apply_gaussian_func" to image {image.active}')

    for plane, img in enumerate(data):
        # out_img[plane] = gaussian(img, sigma, preserve_range = True)
        gauss_kernel1d = signal.windows.gaussian(M= kernel_size, std=sigma)
        gauss_kernel2d = gauss_kernel1d[:, None] @ gauss_kernel1d[None]
        out_img[plane] = signal.oaconvolve(img, gauss_kernel2d, mode="same")

    # return (gaussian(data, sigma))
    return out_img

def apply_median_filt_func (image: "napari.types.ImageData",
    param)-> "Image":

    """
    Apply Median filter to selected image.

    Parameters
    ----------
    image : np.ndarray
        The image to be back subtracted.
    
    footprint: ndarray 
        Magintud of Median filter.
    
    
    Returns
    -------
    out_img : np.ndarray
       Smoothed Image with Median filter.

    """
    param = int(param)
    data = image.active.data
    out_img = np.empty_like(data)
    footprint = disk(int(param))

    print(f'applying "apply_median_filt_func" to image {image.active}')

    # for plane, img in enumerate(data):
        # out_img[plane] = median(img, footprint = footprint)

    # for plane in list(range(data.shape[0])):
    #     out_img[plane, :, :] = median(data[plane, :, :], footprint = footprint)

# using numba method #
    # out_img = parallel_median(data,footprint)
    # for plane, img in enumerate(data):
    #     out_img[plane] = signal.medfilt2d(img, kernel_size = param)
    
    out_img = ndimage.median_filter(data, size = (1, param, param))
    # return (gaussian(data, sigma))
    return out_img


# @jit
# def parallel_median(image, footprint):
#     out_img = np.empty_like(image)
#     footprint = disk(int(footprint))
#     for plane in prange(image.shape[0]):
#         out_img[plane, :, :] = median(image[plane, :, :], footprint = footprint)
#     return out_img







def pick_frames_fun(
    image: "napari.types.ImageData",
    n_frames = 5,
    from_time = 0,
    to_time = 50,
    fps = 200,
    time_unit = "ms")-> "napari.types.LayerDataTuple":
    
    """
    
    Subset the stak based on selected time points.
    
    Parameters
    ----------
    data : np.ndarray
        The image to be Subsetted.
        
    n_frames = int
        Number of output frames output.
    
    from_time = float
        starting timepoint frame in ms.
    
    to_time = float
        final tinmepoint frame in ms.
    
    fps = float
        time resolution of your aquisition in Hz.
    
    time_unit = str
        "ms" or "s"
    

    Returns
    -------
    inverted_signal : np.ndarray
        two images for Calcim and Voltage signals respectively?
    
    """
    data = image.active.data

    time_fct = None
    
    
    try:
        
        if time_unit == "ms":
            time_fct = 1000
        
        elif time_fct == "s":
            time_fct = 1
    
    except ValueError:
        
        print("Time unit must be in 'ms' or 's'.")
        
    
    time_conv_factor = int(1 / fps * time_fct)
    
    
    if (to_time - from_time ) <  time_conv_factor:
        
        warnings.warn(f' number of frames requiered = {n_frames} in the time range selected {to_time - from_time} {time_unit} are beyond the time resolution allowed "fps" = {time_conv_factor} ms. Please select a larger range or less frames. Only 1 frame will be returned.')
        
        n_frames = 1

   
    from_time = int(from_time / time_conv_factor)
    to_time = int(to_time / time_conv_factor) 
    
    

    if (to_time - from_time ) <  n_frames:
        
        warnings.warn(f' number of frames requiered = {n_frames} in the time range selected {to_time - from_time} {time_unit} are beyond the time resolution allowed "fps" = {time_conv_factor} ms. Please select a larger range or less frames. Number of frames will be set to the maximum allowed: {to_time} frames.')
        
        n_frames = to_time
        

    
    # inx = int((to_time - from_time )  / n_frames) 
    # inx = np.arange(from_time, to_time, inx)
    subset = data[from_time:to_time, ...]
    indx = np.arange(0, subset.shape[0], subset.shape[0] / n_frames, dtype = int)
    
    print(f'applying "pick_frames_fun" to image {image.active}')
    return(subset[indx, ...])


def motion_correction_func(image: "napari.types.ImageData",
        foot_print_size = 10, radius_size = 7, num_warp = 20)-> "Image":

    """
    Apply Registration (motion correction) to selected image.

    Parameters
    ----------
    image : np.ndarray
        The image to be processed.
    
    foot_print_size: int
        Magintud of footprint for local normalization, default to 10.

    radius_size: int
        Magintud of footprint for local normalization, default to 7.
    
    num_warp: int
         Magintud of footprint for local normalization, default to 7.

    
    
    Returns
    -------
        registered_img : np.ndarray of type np.uint16
        Corrected Image after registration filter.

    """
    foot_print = disk(foot_print_size)
    data = image.active.data
    # imgae must be converted to integer for the method `skimage.filters.rank.minimum`
    data = data.astype(np.uint16)
    
    # out_img = np.zeros_like(my_3d_image)
    
    # apply local scaling
    scaled_img = []
    # print(type(data))
    
    for plane, img in enumerate(data):

        im_min = rank.minimum(img, footprint=foot_print)
        im_max = rank.maximum(img, footprint=foot_print)
        im_local_scaled = (img - im_min) / (im_max- im_min)
        scaled_img.append( im_local_scaled)
    
    scaled_img = np.asanyarray(scaled_img)
    
    ref_frame = scaled_img[0, ...] # take 1st frame as reference? (only works on averaged traces)
    nr, nc = ref_frame.shape
    
    # apply registration
    registered_img = []
    
    for plane2, img2 in enumerate(scaled_img):
        v, u = optical_flow_ilk(ref_frame, img2, radius = radius_size, num_warp=num_warp)
        row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc),
                                     indexing='ij')
        image1_warp = warp(data[plane2], np.array([row_coords + v, col_coords + u]),
                   mode='edge', preserve_range=True)
        registered_img.append(image1_warp)
        
    
    registered_img = np.asanyarray(registered_img, dtype=np.uint16).copy()
    
    return registered_img


def transform_to_unit16_func(image: "napari.types.ImageData")-> "Image":

    """
    Transfrom numpy array values to type: np.uint16.

    Parameters
    ----------
    image : np.ndarray
        The image to be processed.
     
    
    Returns
    -------
        image : np.ndarray of type np.uint16

    """
    
    

    return image.active.data.astype(np.uint16)



def apply_butterworth_filt_func(image: "napari.types.ImageData",
        ac_freq, cf_freq, fil_ord )-> "Image":
        
        """
        Transfrom numpy array values to type: np.uint16.

        Parameters
        ----------
        image : np.ndarray
            The image to be processed.
        
        ac_freq : float
            Acquisition time interval betwen each fram in ms.
        
        cf_freq : int
            Cutoff Frequency for butterworth low band filter.

        fil_ord : int
            Order size of the filter.
        

     
    
        Returns
        -------
            filt_image : np.ndarray with filtered data

        """
        
        normal_freq = cf_freq / ((1 /ac_freq) / 2)
        
        a, b = signal.butter(fil_ord, normal_freq, btype='low')

        print(f"Applying 'apply_butterworth_filt_func'  to image {image.active}'")
        filt_image = signal.filtfilt(a, b, image.active.data, 0)
        
        return filt_image

    
