########## GUI libraries ##########
from qtpy import QtCore, QtGui
from qtpy.QtWidgets import QHBoxLayout, QVBoxLayout, QGroupBox, QGridLayout, QCheckBox, QComboBox
from qtpy.QtCore import Qt, QAbstractTableModel, QModelIndex, QRect, QPropertyAnimation, QPoint, QEasingCurve, Property
# from qtpy.QtCore import *
from qtpy.QtGui import QColor, QPainter, QStandardItemModel, QStandardItem

########## cv libraries ##########
from skimage.filters import gaussian, threshold_triangle, median, rank, sobel
from skimage.measure import label
from skimage.filters.rank import mean_bilateral
from skimage.morphology import disk, binary_closing, remove_small_objects, closing
from skimage.registration import optical_flow_ilk
from skimage import transform, exposure, morphology, registration, segmentation
from skimage.restoration import denoise_bilateral
from skimage.measure import regionprops

########## utils ##########
import warnings
import tqdm.auto as tqdm
from time import time
import pandas as pd
import yaml
import toml
import datetime
from tifffile import imwrite

########## napari & friends ##########
from napari.layers import Image
from napari.types import ImageData, ShapesData
import sif_parser
# from numba import njit
from napari.utils import progress
from optimap.image import detect_background_threshold
from optimap import motion_compensate
from optimap.video import normalize_pixelwise_slidingwindow, normalize_pixelwise

########## scientific computing libraries##########
# from numba import jit, prange
from scipy import signal, ndimage
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter, binary_fill_holes
import numpy as np
# import cupy as cp
# from cupyx.scipy.ndimage import median_filter as cp_median_filter
# from cupyx.scipy.ndimage import gaussian_filter as cp_gaussian_filter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  import napari

def import_gpu_or_cpu():
    try:
        import cupy as cp # type: ignore
        from cupyx.scipy.ndimage import median_filter as cp_median_filter # type: ignore
        from cupyx.scipy.ndimage import gaussian_filter as cp_gaussian_filter # type: ignore
        return {
            'array': cp,
            'median_filter': cp_median_filter,
            'gaussian_filter': cp_gaussian_filter,
            'use_gpu': True
        }
    except ImportError:
        import numpy as np
        from scipy.ndimage import median_filter, gaussian_filter
        return {
            'array': np,
            'median_filter': median_filter,
            'gaussian_filter': gaussian_filter,
            'use_gpu': False
        }

backend = import_gpu_or_cpu()


# def detect_spots(
#     image: "napari.types.ImageData",
#     high_pass_sigma: float = 2,
#     spot_threshold: float = 0.01,
#     blob_sigma: float = 2
# ) -> "napari.types.LayerDataTuple":
# import cupy as cp
# import cupyx
# from cucim.skimage import registration as registration_gpu
# from cucim.skimage import transform as transform_gpu




def invert_signal(
    data: 'napari.types.ImageData'
    )-> 'napari.types.LayerDataTuple':

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

    return np.nanmax(data) - data
   

def local_normal_fun(
    image: "napari.types.ImageData")-> "napari.types.ImageData":

    """Normalize traces pixelwise along the time dimension.

    Parameters
    ----------
    image : np.ndarray
        The image to be filtered.

    Returns
    -------
    inverted_signal : np.ndarray
        The image with inverted fluorescence values
    """
    # results = (image - np.min(image, axis = 0)) / np.max(image, axis=0)
    # results = np.nan_to_num(results, nan=0)
    # return results
    return normalize_pixelwise(image)


def global_normal_fun(
    data: "napari.types.ImageData")-> "napari.types.ImageData":

    """Nomrlaize and scale to [0, 1] the siganl by the global max and min 
    previusly cliping the data between 5-95%. This helps to remove outliers on the data.
    
    source: 'https://scikit-image.org/docs/stable/auto_examples/color_exposure/plot_adapt_hist_eq_3d.html#sphx-glr-auto-examples-color-exposure-plot-adapt-hist-eq-3d-py'

    Parameters
    ----------
    image : np.ndarray
        The image to be normalized.

    Returns
    -------
    normalized_signal : np.ndarray
        The normlaized numpy array.

    """

    # Rescale image data to range [0, 1]
    im_orig = data
    # NOTE: clipping does not seem to be usefull inthis case.
    # im_orig = np.clip(data,
    #                 np.percentile(data, 2),
    #                 np.percentile(data, 98)
    #                 )
    # eps = np.finfo(im_orig.dtype).eps
    
    return (im_orig - im_orig.min()) / (im_orig.max() - im_orig.min() )



def slide_window_normalization_func(np_array, slide_window = 100):
    # NOTE: you have to check an error here.
    # Either any of the two approach bellow fix issue for dealing with float np.array
    # np_array = np_array.copy() 
    # y = np.ascontiguousarray(np_array)
    return normalize_pixelwise_slidingwindow(np.ascontiguousarray(np_array), window_size=  slide_window)



def split_channels_fun(
    data: "napari.types.ImageData")-> "napari.types.LayerDataTuple":

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
    
    ch_1 = data[::2,:,:]
    ch_2 = data[1::2,:,:]
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


def apply_gaussian_func (data: "napari.types.ImageData",
    sigma, kernel_size = 3)-> "Image":

    """
    Apply Gaussian filter to selected image.

    Parameters
    ----------
    image : np.ndarray
        The image to be back subtracted.
    
    sigma: int
        Magintud of gaussian filer kernel, default to 2.
    
    
    Returns
    -------
    out_img : np.ndarray
       Smoothed Image with Gaussian filter.

    """

    # data = image.active.data
    # out_img = np.empty_like(data)
    # gauss_kernel1d = signal.windows.gaussian(M= kernel_size, std=sigma)
    # gauss_kernel2d = gauss_kernel1d[:, None] @ gauss_kernel1d[None]

    # print(f'applying "apply_gaussian_func" to image {image.active}')

    # for plane, img in enumerate(tqdm.tqdm(data)):
    # for plane, img in enumerate(progress(data)):
        # out_img[plane] = gaussian(img, sigma, preserve_range = True)
        # out_img[plane] = signal.oaconvolve(img, gauss_kernel2d, mode="same")

    # ###########################
    # scikitimage (CPU) base function
    # ###########################  

    # return (gaussian(data, sigma))
    # return out_img
    # return gaussian_filter(data, sigma=sigma, order= 0, radius=kernel_size, # axes = (1,2))


    # data_cp = cp.asarray(data)
    # out_img = cp_gaussian_filter(data_cp, 
    #                           sigma=(0, sigma, sigma), 
    #                           order= 0,
    #                           # radius=kernel_size,
    #                           truncate=kernel_size,
    #                           #axes = (1,2)
    #                           )
    # return cp.asnumpy(out_img)

    data_cp = backend['array'].asarray(data)
    out_img = backend['gaussian_filter'](data_cp, 
                                         sigma=(0, sigma, sigma), 
                                         order=0,
                                         truncate=kernel_size)
    
    return backend['array'].asnumpy(out_img) if backend['use_gpu'] else out_img

    



def apply_median_filt_func (data: "napari.types.ImageData",
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
#     param = int(param)
#     # data = image.active.data
#     # out_img = np.empty_like(data)
#     footprint = disk(int(param))

#     # print(f'applying "apply_median_filt_func" to image {image.active}')

#     # for plane, img in enumerate(data):
#         # out_img[plane] = median(img, footprint = footprint)

#     # for plane in list(range(data.shape[0])):
#     #     out_img[plane, :, :] = median(data[plane, :, :], footprint = footprint)

# # using numba method #
#     # out_img = parallel_median(data,footprint)
#     # for plane, img in enumerate(data):
#     #     out_img[plane] = signal.medfilt2d(img, kernel_size = param)
    
#     # out_img = ndimage.median_filter(data, size = (1, param, param))

    
#     data_cp = cp.asarray(data)
#     # xp = cpx.get_array_module(data_cp, param)  # 'xp' is a standard usage in the community
#     # print("Using:", xp.__name__)
#     out_img = cp_median_filter(data_cp, size = (1, param, param))
    
#     return cp.asnumpy(out_img)



    param = int(param)
    
    # Convert data to GPU array if on GPU, else use CPU array
    data_cp = backend['array'].asarray(data)
    
    # Apply the filter using the appropriate backend
    out_img = backend['median_filter'](data_cp, size=(1, param, param))
    
    # Convert back to CPU array if needed
    return backend['array'].asnumpy(out_img) if backend['use_gpu'] else out_img


    param = int(param)
    
    # Convert data to GPU array if on GPU, else use CPU array
    data_cp = backend['array'].asarray(data)
    
    # Apply the filter using the appropriate backend
    out_img = backend['median_filter'](data_cp, size=(1, param, param))
    
    # Convert back to CPU array if needed
    return backend['array'].asnumpy(out_img) if backend['use_gpu'] else out_img


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


# def motion_correction_func(image: "napari.types.ImageData",
#         foot_print_size = 10, radius_size = 7, num_warp = 20)-> "Image":

#     """
#     Apply Registration (motion correction) to selected image.

#     Parameters
#     ----------
#     image : np.ndarray
#         The image to be processed.
    
#     foot_print_size: int
#         Magintud of footprint for local normalization, default to 10.

#     radius_size: int
#         Magintud of footprint for local normalization, default to 7.
    
#     num_warp: int
#          Magintud of footprint for local normalization, default to 7.

    
    
#     Returns
#     -------
#         registered_img : np.ndarray of type np.uint16
#         Corrected Image after registration filter.

#     """
#     foot_print = disk(foot_print_size)
#     data = image.active.data
#     # imgae must be converted to integer for the method `skimage.filters.rank.minimum`
#     data = data.astype(np.uint16)
    
#     # out_img = np.zeros_like(my_3d_image)
    
#     # apply local scaling
#     scaled_img = []
#     # print(type(data))
    
#     for plane, img in enumerate(data):

#         im_min = rank.minimum(img, footprint=foot_print)
#         im_max = rank.maximum(img, footprint=foot_print)
#         im_local_scaled = (img - im_min) / (im_max- im_min)
#         scaled_img.append( im_local_scaled)
    
#     scaled_img = np.asanyarray(scaled_img)
    
#     ref_frame = scaled_img[0, ...] # take 1st frame as reference? (only works on averaged traces)
#     nr, nc = ref_frame.shape
    
#     # apply registration
#     registered_img = []
    
#     for plane2, img2 in enumerate(scaled_img):
#         v, u = optical_flow_ilk(ref_frame, img2, radius = radius_size, num_warp=num_warp)
#         row_coords, col_coords = np.meshgrid(np.arange(nr), np.arange(nc),
#                                      indexing='ij')
#         image1_warp = warp(data[plane2], np.array([row_coords + v, col_coords + u]),
#                    mode='edge', preserve_range=True)
#         registered_img.append(image1_warp)
        
    
#     registered_img = np.asanyarray(registered_img, dtype=np.uint16).copy()
    
#     return registered_img

################# NOTE: deprecating function 15.08-2024 #################
# 
# def scaled_img_func(data, foot_print_size = 10):
#     foot_print = disk(foot_print_size)
    
#     xp = cp.get_array_module(data)
#     xpx = cupyx.scipy.get_array_module(data)
    
#     # data = xp.asarray(img)
#     print(f"Using: {xp.__name__} in scaled_img_func")
    
#     scaled_img = xp.empty_like(data, dtype= (xp.float64))
    
    
#     for plane, img in enumerate(data):
#         # if plane % 50 == 0:
#         #     print(f"normalizing plane: {plane}")
            
#         im_min = xpx.ndimage.minimum_filter(img, footprint=foot_print)
#         im_max =xpx.ndimage.maximum_filter(img, footprint=foot_print)
#         scaled_img[plane,...] = xp.divide(xp.subtract(img, im_min),  xp.subtract(im_max, im_min))
        
    
#     return scaled_img

################# NOTE: deprecating function 15.08-2024 #################
# 
# def register_img_func(data, orig_data, ref_frame = 1, radius_size = 7, num_warp = 8):
#     xp, xpx = cp.get_array_module(data), cupyx.scipy.get_array_module(data)
    
#     if type(data) == cp.ndarray:
#         device_type = "GPU"
#         register_func = registration_gpu
#         transform_func = transform_gpu
#     else:
#         device_type = "CPU"
#         register_func = registration
#         transform_func = transform
        
#     print (f'using device: {device_type}')
    
#     ref_frame_data = data[ref_frame, ...]
#     nr, nc = ref_frame_data.shape
#     registered_img = xp.empty_like(data)
#     # print(type(registered_img))
        
#     for plane, img in enumerate(data):
#         # if plane % 50 == 0:
#             # print(f"Registering plane: {plane}")
        
#         v, u = register_func.optical_flow_ilk(ref_frame_data, img, 
#                                               radius = radius_size, num_warp=num_warp)
#         row_coords, col_coords = xp.meshgrid(xp.arange(nr), xp.arange(nc),
#                                              indexing='ij')
#         registered_img[plane, ...] = transform_func.warp(orig_data[plane], xp.array([row_coords + v, col_coords + u]),
#                                                          mode='edge', preserve_range=True)
    
#     return registered_img



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



def apply_butterworth_filt_func(data: "napari.types.ImageData",
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
        
        normal_freq = cf_freq / (ac_freq / 2)
        
        a, b = signal.butter(fil_ord, normal_freq, btype='low')

        # print(f"Applying 'apply_butterworth_filt_func'  to image {image.active}'")
        filt_image = signal.filtfilt(a, b, data, 0)
        
        return filt_image



def apply_FIR_filt_func(data: "napari.types.ImageData", n_taps, cf_freq, acquisition_freq
        )-> "Image":
        
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
        
        source: https://chatgpt.com/share/9907ea1b-a997-4547-b21d-493eed893225
     
    
        Returns
        -------
            filt_image : np.ndarray with filtered data along time dimension.

        """

        # Design the FIR filter
        # Define the number of taps (filter length) and the cutoff frequency
        # num_taps = 21  # Length of the filter
        # cutoff_frequency = 0.1  # Normalized cutoff frequency (0 to 1, where 1 is Nyquist frequency)

        # Use firwin to create a low-pass FIR filter
        fir_coeff = signal.firwin(n_taps, cf_freq, window='hamming', fs = acquisition_freq)

        # Apply the FIR filter along the temporal axis (axis=0)
        filt_image = np.apply_along_axis(lambda m: signal.lfilter(fir_coeff, 1.0, m), axis=0, arr=data)
        return filt_image
        




def apply_box_filter(data: "napari.types.ImageData", kernel_size):

    # data = image.active.data
    out_img = np.empty_like(data)

    box_kernel2d = np.ones((kernel_size, kernel_size))/kernel_size**2

    for plane, img in enumerate(data):
        out_img[plane] = signal.oaconvolve(img, box_kernel2d, mode="same")

    # print(f'applying "apply_box_filter" to image {image.active}')

    return (out_img)



def apply_laplace_filter(data: "napari.types.ImageData", kernel_size, sigma):
    
    # data = image.active.data
    out_img = np.empty_like(data)

    mex_hat_kernel1d = signal.ricker(kernel_size, sigma)
    mex_hat_kernel2d = (mex_hat_kernel1d[:, None] @ mex_hat_kernel1d[None]) * -1

    for plane, img in enumerate(data):
        out_img[plane] = signal.oaconvolve(img, mex_hat_kernel2d, mode="same")

    # print(f'applying "apply_laplace_filter" to image {image.active}')

    return (out_img)



def apply_bilateral_filter(data: "napari.types.ImageData", wind_size, sigma_col, sigma_spa):

    out_img = np.empty_like(data)
    # start = time()
    for plane, img in enumerate(progress(data)):
        # out_img[plane] = mean_bilateral(data[plane], disk(disk_size), s0=5, s1=5) this requires the image to be a uint8/uint16
        out_img[plane] = denoise_bilateral(data[plane], 
                                           win_size = wind_size, 
                                           sigma_color = sigma_col, 
                                           sigma_spatial = sigma_spa, 
                                           bins = 1024)
    # end = time()
    # print(f"elapsed time: {round((end - start)/60, 1)} min")
    return (out_img)


def compute_APD_props_func(np_1Darray, curr_img_name, cycle_length_ms, diff_n = 1, rmp_method = "bcl_to_bcl", apd_perc = 75, promi = 0.18, roi_indx = 0, roi_id = None, interpolate = False, curr_file_id =None):
    
    """
        Find the DF/Dt max using 1st derivative of a given average trace.

        Parameters
        ----------
        np_1Darray : np.ndarray
            1D average trace taken form ROI.
        
        diff_n : float
            the Nth derivative to be computed. default to 1st derivative.
        
        prominence : int
            Prominence to detect peaks.

        cycle_length_ms : float
            Cycle length from current acquisition.
        

     
    
        Returns
        -------
            dft_max_df : pd.Dataframe conatining the DF/Dt max for peaks founds in signal.

        """
    
    time = np.arange(0, np_1Darray.shape[-1]) * cycle_length_ms 

    # try:
        
    AP_peaks_indx, AP_peaks_props = signal.find_peaks(signal.savgol_filter(np_1Darray, window_length=15, polyorder=2), prominence=promi) # use Solaiy filter as Callum
    # peak_left_bases_indx = AP_peaks_props["left_bases"]
    # except Exception as e:

    #     print(f"ERROR: computing APD parameters fails witht error: {repr(e)}")



    peaks_times = time[AP_peaks_indx]

    # compute first approximation of BCL based on the peaks

    bcl_list = np.diff(peaks_times) 

    # add a last bcl value for the last AP if the last AP could not have an associated BCL
    if len(bcl_list) < len(peaks_times):
        # bcl_list = [bcl_list, time(end) - times(end)]
        bcl_list = np.append(bcl_list, time[-1] - peaks_times[-1])
        # bcl_list = np.append(bcl_list, (peaks_times[-2] + peaks_times[-1]))


    APD = np.zeros_like(peaks_times)
    activation_time = np.zeros_like(peaks_times)
    repol_time = np.zeros_like(peaks_times)
    dVdtmax =  np.zeros_like(peaks_times)
    resting_V = np.zeros_like(peaks_times, dtype=np.float64) # this must be of float data type otherwose produce a tricky bug later whan asigning a uint type value
    amp_Vmax = np.zeros_like(peaks_times)

    # compute dfdt and normalize it

    dfdt = np.diff(np_1Darray, n = diff_n, prepend = np_1Darray[:diff_n])
    
    dfdt =  (dfdt - dfdt.min()) /  dfdt.max()
    # dff_filt = signal.savgol_filter(dff, filter_size, 2)
    
    indx_AP_upstroke = []
    indx_AP_peak = []
    indx_AP_end = []
    indx_AP_resting = []

    for peak in range(len(peaks_times)):
        #  Find RMP and APD
        # RMP is mean V of BCL/2 to 5ms before AP upstroke to when this value is crossed to BCL/2 (or end) 

        bcl = bcl_list[peak]
        # Find interval of AP in indices
        ini_indx = np.argmax(time >= (peaks_times[peak] - bcl / 1.5))  # finds the first True index
        ini_indx = max(ini_indx, 0)  # ensures the index is at least 0
        
        if peak + 1 == len(peaks_times):
            end_indx = np.argmax(time)
            # end_index_for_upstroke = end_indx - 1
        else:
            end_indx = np.min(np.append(np.argwhere(time >= peaks_times[peak] + bcl/1.5)[0], time.shape[-1] -2))
        
        end_index_for_upstroke = np.append(np.argwhere(time >= peaks_times[peak] + bcl/2)[0], time.shape[-1] -1).min()
            
        # Find upstroke index within the interval
        upstroke_indx = ini_indx + np.argwhere( dfdt[ini_indx:end_index_for_upstroke] >=  dfdt[ini_indx:end_index_for_upstroke].max())[0][0]

        # interpolation to find accurate activation time
        interp_points = 1000;
        if interpolate:

            delta = 10;
            if upstroke_indx - delta <= 0:
                lower_bound_interp = 1;
            else:
                lower_bound_interp = upstroke_indx - delta;
            
            if upstroke_indx + delta > time.shape[-1]:
                upper_bound_interp = time.shape[-1] -1
            else:
                upper_bound_interp = upstroke_indx + delta;
            
            time_fine_grid = np.linspace(time[lower_bound_interp], time[upper_bound_interp], interp_points)
            interpolation_f = CubicSpline(time[lower_bound_interp :  upper_bound_interp], dfdt[lower_bound_interp :  upper_bound_interp], extrapolate=True, bc_type = 'natural')
            dfdt_interpolated = interpolation_f(time_fine_grid) 
            # find new dfdt max
            dfdt_max, dfdt_max_indx = np.max(dfdt_interpolated), np.argmax(dfdt_interpolated)
            activation_time[peak] = time_fine_grid[dfdt_max_indx]
            dVdtmax[peak] = dfdt_max * cycle_length_ms
        
        if not interpolate:

            # dfdt_max, dfdt_max_indx = np.max(dfdt_interpolated), np.argmax(dfdt_interpolated)
            activation_time[peak] = time[upstroke_indx]
            dVdtmax[peak] = np.max(dfdt) * cycle_length_ms


        # compute RMP from before and after upstroke
        end_rmp_indx = upstroke_indx - np.ceil(np.divide(0.005 , cycle_length_ms)).astype(np.int64) # take mean from init_Ap indx until 5 ms before upstroke
        init_RMP = np.nanmedian(np_1Darray[ini_indx:end_rmp_indx]) # using median isntead of mean
        min_indx = np.argwhere(np_1Darray[upstroke_indx:end_indx] <= init_RMP)

        if not np.any(min_indx):
            cross_indx = np.minimum(upstroke_indx, end_indx)
        else:
            cross_indx = np.minimum(upstroke_indx + min_indx[0], end_indx)[0]
        
        end_RMP = np.nanmedian(np_1Darray[cross_indx:end_indx]) # using median instead of mean

        #  insert here a conditional to set the method for rmp computation 
        if rmp_method == "bcl_to_bcl":
            # use mean of RMP previous and after AP
            resting_V[peak] = np.mean(np.array(init_RMP, end_RMP))
            # use minimum value pre upstroke
        if rmp_method == "pre_upstroke_min":
            resting_V[peak] = np.min(np_1Darray[ini_indx:upstroke_indx])
            # use minimum value after AP
        if rmp_method == "post_AP_min":
            next_peak_index = np.minimum(np.argwhere(time >= peaks_times[peak] + bcl )[0], time.shape[-1] -1)[0]
            resting_V[peak] = np.min(np_1Darray[upstroke_indx:next_peak_index])
            # take the average of min before upstroke and min after AP
        if rmp_method == "ave_pre_post_min":
            pre_min = np.min(np_1Darray[ini_indx:upstroke_indx])
            next_peak_index = np.minimum(np.argwhere(time >= peaks_times[peak] + bcl )[0], time.shape[-1] -1)[0]
            post_min = np.min(np_1Darray[upstroke_indx:next_peak_index])
            resting_V[peak] = np.mean(np.array(pre_min, post_min))

        
        # compute APD
        V_max = np.max(np_1Darray[ini_indx:end_indx])
        amp_Vmax[peak] = V_max
        amp_V = (((100 - apd_perc) / 100) * (V_max - resting_V[peak])) + resting_V[peak]
        # Find index where the AP has recovered the given percentage (or if it didnt, take the last index)
        current_APD_segment = np_1Darray[AP_peaks_indx[peak] + 1 : end_indx]
        indx_baseline = np.argwhere(np_1Darray[ini_indx:upstroke_indx][::-1] <= resting_V[peak]) # look backwards and find the first index where the the baseline start
        resting_V_indx = upstroke_indx - indx_baseline[0][0] if indx_baseline.size > 0 else upstroke_indx - np.argwhere(np_1Darray[ini_indx:upstroke_indx][::-1] <= np.mean(np_1Darray[ini_indx:upstroke_indx]))[0][0] # uses index of baseline if exists otherwise get the average from ini_indx:upstroke_indx
        repol_index =  AP_peaks_indx[peak] + np.minimum(np.argmax(current_APD_segment <= amp_V) , current_APD_segment.shape[-1] -1)
        pre_repol_index = repol_index - 2
        # generate fine grid for interpolation in ms
        time_fine_grid = np.linspace(time[pre_repol_index], time[repol_index], interp_points)
        Vm_interpolated = np.interp(time_fine_grid, time[pre_repol_index:repol_index], np_1Darray[pre_repol_index:repol_index])
        # repol_index_interpolated = np.nanmin( np.append(np.array(time_fine_grid.size -1 ), np.argwhere(Vm_interpolated <= amp_V)))
        repol_index_interpolated = np.append(np.argwhere(Vm_interpolated <= amp_V), time_fine_grid.size -1 ).min()

        repol_time[peak] = time_fine_grid[repol_index_interpolated] #* 1000
        # APD[peak] = repol_time[peak] - activation_time[peak]
        APD[peak] = repol_time[peak] - time[resting_V_indx]

        indx_AP_upstroke.append(upstroke_indx) 
        indx_AP_peak.append(AP_peaks_indx[peak]) 
        indx_AP_end.append(repol_index )
        indx_AP_resting.append(resting_V_indx)



    AP_ids = [f'AP_{i}' for i in range(peaks_times.shape[-1])]
    if not roi_id:
        ROI_ids = [f'ROI-{roi_indx}' for i in range(peaks_times.shape[-1])]
    else:
        ROI_ids = [f'ROI-{roi_id}' for i in range(peaks_times.shape[-1])]
        
    apd_perc = [apd_perc for i in range(peaks_times.shape[-1])]
    img_name = [curr_img_name for i in range(peaks_times.shape[-1])]
    file_id = [curr_file_id for i in range(peaks_times.shape[-1])]

    rslt_dict = {
        "image_name": img_name,
        "ROI_id": ROI_ids,
        "AP_id": AP_ids,
        "APD_perc": apd_perc,
        "APD": APD,
        "AcTime_dVdtmax":dVdtmax,
        "amp_Vmax":amp_Vmax,
        "BasCycLength_bcl":bcl_list,
        "resting_V":resting_V,
        "time_at_AP_upstroke":time[indx_AP_upstroke],
        "time_at_AP_peak":time[indx_AP_peak], 
        "time_at_AP_end":time[indx_AP_end], 
        "indx_at_AP_resting":indx_AP_resting,
        "indx_at_AP_upstroke":indx_AP_upstroke,
        "indx_at_AP_peak":indx_AP_peak, 
        "indx_at_AP_end":indx_AP_end,
        "curr_file_id":file_id
        }
     
    # rslt_df = rslt_df.apply(lambda x: np.round(x * 1000, 2) if x.dtypes == "float64" else x ) # convert to ms and round values
    return (rslt_dict)

def return_spool_img_fun(path):
    data, info = sif_parser.np_spool_open(path, multithreading= True, max_workers=16)
    info['CurrentFileSource'] = path
    info = {key: val for key, val in info.items() if (not key.startswith("timestamp") and (not key.startswith("tile"))) }
    return (np.flip(data, axis=(1)), info)


def return_peaks_found_fun(promi, np_1Darray):
    AP_peaks_indx, AP_peaks_props = signal.find_peaks(signal.savgol_filter(np_1Darray, window_length=15, polyorder=2), prominence=promi) # use Solaiy filter as Callum

    return len(AP_peaks_indx)

def add_index_dim(arr1d, scale):
    """Add a dimension to a 1D array, containing scaled index values.
    
    :param arr1d: array with one dimension
    :type arr1d: np.ndarray
    :param scale: index scaling value
    :type scale: float
    :retun: 2D array with the index dim added at -1 position
    :rtype: np.ndarray
    """
    idx = np.arange(0, arr1d.size, 1) * scale
    out = np.zeros((2, arr1d.size))
    out[0] = idx
    out[1] = arr1d
    return out


def extract_ROI_time_series(img_layer, shape_layer, idx_shape, roi_mode, xscale = 1):
    """Extract the array element values inside a ROI along the first axis of a napari viewer layer.

    :param current_step: napari viewer current step
    :param layer: a napari image layer
    :param labels: 2D label array derived from a shapes layer (Shapes.to_labels())
    :param idx_shape: the index value for a given shape
    :param roi_mode: defines how to handle the values inside of the ROI -> calc mean (default), median, sum or std
    :return: shape index, ROI mean time series
    :rtype: np.ndarray
    """

    ndim = img_layer.ndim
    dshape = img_layer.data.shape

    mode_dict = dict(
        Min=np.min,
        Max=np.max,
        Mean=np.mean,
        Median=np.median,
        Sum=np.sum,
        Std=np.std,
    )
    # convert ROI label to mask
    if ndim == 3:    
        label = shape_layer.to_labels(dshape[-2:])
        mask = np.tile(label == (idx_shape + 1), (dshape[0], 1, 1))

    if mask.any():
        return add_index_dim(mode_dict[roi_mode](img_layer.data[mask].reshape(dshape[0], -1), axis=1), xscale)


def return_AP_ini_end_indx_func(my_1d_array, promi = 0.03):
    """
    This function takes a 1d array trace, compute the peaks
    and return the ini, end, and peak indexes of n numbers of peaks found.
    """
    
    AP_peaks_indx, _ = signal.find_peaks(signal.savgol_filter(my_1d_array, 
                                                                           window_length=15, 
                                                                           polyorder=2), 
                                                      prominence=promi) # use Solaiy filter as Callum
    

       # handle case when theere is ony one peak found
    if len(AP_peaks_indx) == 1:
        
        bcl_list = len(my_1d_array) - AP_peaks_indx[0] 
        end_ap_indx = len(my_1d_array)
        ini_ap_indx = AP_peaks_indx
        return ini_ap_indx, AP_peaks_indx, end_ap_indx
        # set promi to 100
    elif len(AP_peaks_indx) > 1:

        bcl_list = np.diff(AP_peaks_indx) 
        bcl_list = np.median(bcl_list).astype(np.uint16)
        half_bcl_list = np.round(bcl_list // 2 )

        end_ap_indx = AP_peaks_indx + half_bcl_list
        ini_ap_indx = AP_peaks_indx - half_bcl_list

        for indx, indx_peak in enumerate(AP_peaks_indx):
            # handeling first trace
            if ini_ap_indx[indx] < 0 :
                
                half_bcl_list = half_bcl_list + ini_ap_indx[indx]        
                ini_ap_indx = AP_peaks_indx - half_bcl_list

            # handeling last trace NOTE: no sure if this make sense, need to teste with real data
            if end_ap_indx[indx] > len(my_1d_array):
                half_bcl_list = half_bcl_list - end_ap_indx[indx]
                end_ap_indx = AP_peaks_indx - half_bcl_list
        return ini_ap_indx, AP_peaks_indx, end_ap_indx
    
    elif len(AP_peaks_indx) < 1:

        raise ValueError(f"Number of AP founds  = {len(AP_peaks_indx)}. Change the threshold, ROI or check your image")


    # return splited_arrays
    


# def split_AP_traces_and_ave_func(trace, ini_i, end_i, type = "1d", return_mean = False):
#     """
#     This function takes a 1d or 3D array, ini index, end index of ap 
#     previously computed with function 'return_AP_ini_end_indx_func' 
#     and return the splitted arrays for each AP.
#     """
#     # must check that all len are the same
#     n_peaks = len(ini_i)
    
#     splitted_traces = [[trace[ini:end, ...]] for ini, end in zip(ini_i, end_i)]
#     # take the small trace len and adjust the other traces to that
#     min_dim = np.min([trace[0].shape for trace in splitted_traces])
#     splitted_traces = [trace[0][:min_dim] for trace in splitted_traces]

#     if type == "1d":
#         splitted_traces = np.array(splitted_traces).reshape(n_peaks, -1)
    
#     elif type == "3d":
#         img_dim_x, img_dim_y = trace.shape[-2:]
#         splitted_traces = np.array(splitted_traces).reshape( n_peaks, -1, img_dim_x, img_dim_y)
    

#     if return_mean:
#         splitted_traces = np.mean(np.array([trace for trace in splitted_traces]), axis=(0))
        
    
#     return splitted_traces

def concatenate_and_padd_with_nan_2d_arrays(arrays):
    """
    Concatenates multiple 2D arrays with different sizes into one larger array.
    
    If the arrays have different shapes, they will be padded with NaN values 
    to ensure the final concatenated array has a consistent size.
    
    Parameters:
    arrays (list of np.ndarray): A list containing 2D numpy arrays to be concatenated.

    Returns:
    np.ndarray: A single 2D numpy array where all input arrays are stacked along the
                second axis (horizontally), padded with NaNs where the arrays are smaller.
    
    Example:
    >>> a1 = np.array([[1, 2], [3, 4]])
    >>> a2 = np.array([[5, 6, 7]])
    >>> a3 = np.array([[8]])
    >>> concatenate_with_nan([a1, a2, a3])
    array([[ 1.,  2., nan],
           [ 3.,  4., nan],
           [ 5.,  6.,  7.],
           [ 8., nan, nan]])
    """
    # Find the maximum shape (height and width) among all the arrays
    max_rows = max(arr.shape[0] for arr in arrays)
    max_cols = max(arr.shape[1] for arr in arrays)

    # Create a list of padded arrays
    padded_arrays = [np.pad(arr, ((0, max_rows - arr.shape[0]), 
                                  (0, max_cols - arr.shape[1])), 
                            mode='constant', constant_values=np.nan)
                     for arr in arrays]
    
    # Concatenate them along the second axis (horizontally)
    return np.hstack(padded_arrays)


def split_AP_traces_and_ave_func(trace, ini_i, end_i, type="1d", return_mean=False):
    """
    This function takes a 1D or 3D array and splits it into segments based on 
    the provided start (ini_i) and end (end_i) indices, handling cases where the 
    lengths of the split segments may differ. Optionally, it computes the mean of 
    the split segments, while ignoring NaNs (used for padding).

    Parameters:
    trace (np.array): Input 1D or 3D array to be split.
    ini_i (list or np.array): List of start indices for each segment.
    end_i (list or np.array): List of end indices for each segment.
    type (str): Indicates whether the input array is "1d" or "3d". Default is "1d".
    return_mean (bool): If True, returns the average of the split arrays. Default is False.

    Returns:
    np.array: An array containing the split traces. If return_mean is True, returns the 
    mean of the split traces, with NaNs ignored during averaging.
    """
    
    # Number of segments to be split from the trace
    n_peaks = len(ini_i)
    
    # Splitting the trace into individual segments based on ini_i and end_i
    splitted_traces = [trace[ini:end, ...] for ini, end in zip(ini_i, end_i)]
    
    # Find the maximum length of the segments to ensure consistent array dimensions
    max_len = max([segment.shape[0] for segment in splitted_traces])
    
    # Pad shorter segments with NaN values to match the maximum length
    padded_traces = [np.pad(segment, ((0, max_len - segment.shape[0]),) + ((0, 0),) * (segment.ndim - 1), 
                            mode='constant', constant_values=np.nan) for segment in splitted_traces]
    
    # Reshaping based on whether the input array is 1D or 3D
    if type == "1d":
        # Convert list of arrays into a 2D numpy array (n_peaks x max_len)
        padded_traces = np.array(padded_traces).reshape(n_peaks, -1)
    
    elif type == "3d":
        # Assumes the input trace is 3D and reshapes accordingly (n_peaks x max_len x img_dim_x x img_dim_y)
        img_dim_x, img_dim_y = trace.shape[-2:]
        padded_traces = np.array(padded_traces).reshape(n_peaks, max_len, img_dim_x, img_dim_y)

    # If return_mean is True, compute the mean across the first dimension (n_peaks), ignoring NaN values
    if return_mean:
        padded_traces = np.nanmean(padded_traces, axis=0)
        
    return padded_traces




def return_maps(image: "napari.types.ImageData", cycle_time, percentage, map_type = 0) -> "napari.types.ImageData":
    # data: "napari.types.ImageData")-> "napari.types.ImageData":
    
    """
        Find the DF/Dt max using 1st derivative of a given average trace.

        Parameters
        ----------
        image : np.ndarray
            3D stack image of a single AP. ussually teh result from 
            averageing multiple APs.
        map_type = 0 for Aactvation time maps, 2 for APD maps

        Returns
        -------
        inverted_signal : np.ndarray
            The image with inverted fluorescence values

    """

    # 1. Get gradient and normalize it
    dfdt = np.gradient(image, axis=0)
    dfdt = (dfdt - np.nanmin(dfdt)) / np.nanmax(dfdt)
    dfdt = np.nan_to_num(dfdt, nan=0)
    
    activation_times = np.full_like(image[0, ...], fill_value= np.nan,  dtype=np.float64)

    
    ini_indices = np.nanargmax(dfdt, axis=0)
    
    #  may be a faster way to compute act time. just muktiply cycle_time by the index at max dfdt.
    activation_times[ini_indices != 0] = ini_indices[ini_indices != 0]* cycle_time

    if map_type == 0:

        if cycle_time == 1:
            return activation_times
        else:
            return activation_times * 1000 # convert to ms
    
    elif map_type ==2: 

        delta = 10 # this define the number of frames before and after the ini_indices
        
        # 2. get time vector
        n_frames, y_size, x_size = image.shape
        time = np.arange(0, n_frames) * cycle_time
        end_indx = np.argmax(time)
        # end_indx = np.min(np.append(np.argwhere(time >= peaks_times[peak] + bcl/1.5)[0], time.shape[-1] -2))

        # 3 find peak of signal
        # max_v = np.max(image, axis = 0)
        max_v_indx_stack = np.argmax(image, axis = 0)
        APD = np.full_like(ini_indices, fill_value= np.nan,  dtype=np.float64)
        
        mask_repol_indx_out = np.full_like(ini_indices, fill_value= np.nan,  dtype=np.uint16)
        t_index_out = np.full_like(ini_indices, fill_value= np.nan,  dtype=np.uint16)
        
        pre5ms_indx = int(np.ceil(0.005 / cycle_time))

                    
        # activation_times[activation_times == 0] = np.nan # remove zeros
        # 4 main loop
        for y_px  in range(y_size):
            for x_px in range(x_size):
                
                t_index = ini_indices[y_px, x_px]
                # from_ini_to_end_vector = image[t_index:, y_px, x_px]
                # trace_length_from_peak = len(from_ini_to_end_vector)
                # print(max_v_indx_stack.size)
                max_v_indx = max_v_indx_stack[y_px, x_px]
                
                # if  (t_index != 0) | (np.isnan(t_index)) and (from_ini_to_end_vector.size > 0):  #assert that you have a peak
                # if ~ np.isnan(image[t_index:, y_px, x_px]).any()  :  #assert that is not nan
                if (t_index != 0):
                    # print(t_index)
                    try:
                                        
                        Vm_signal = image[:, y_px, x_px]
                        end_rmp_indx = (t_index) - pre5ms_indx # 5 ms before upstroke
                        
                        resting_V = np.nanmean(image[:end_rmp_indx, y_px, x_px]) 
                        
                        amp_V = ( ((100 - percentage)/ 100) * (image[max_v_indx, y_px, x_px] - resting_V)) + resting_V
                        # print(f"amp_V: {amp_V}")
                        
                        mask_repol_indx =  np.argwhere(image[max_v_indx:, y_px, x_px]<= amp_V)
                        # if mask_repol_indx
                        mask_repol_indx =  max_v_indx + mask_repol_indx[0].min() if mask_repol_indx.size != 0 else max_v_indx
                        mask_repol_indx_out[y_px, x_px] = mask_repol_indx
                        t_index_out[y_px, x_px] = t_index
            
                        #get APD duration
                        len_segment = image[t_index:mask_repol_indx, y_px, x_px].size
                        # print(f"len_segment: {len_segment}")
                        APD[y_px, x_px] = len_segment * cycle_time
                        # print(f"APD: {len_segment * cycle_time_in_ms}")

                            
                    except Exception as e:
                        # print(mask_repol_indx.size, np.isnan(mask_repol_indx))
                        # print("you enter the exeception and are not printing")
                        warnings.warn(f"ERROR: Computing APD parameters fails witht error: {repr(e)}.")
                        # break
                        raise e

        return (APD, mask_repol_indx_out, t_index_out, resting_V)
    
def return_index_for_map(image: "napari.types.ImageData", cycle_time, percentage, map_type = 0) -> "napari.types.ImageData":
    # data: "napari.types.ImageData")-> "napari.types.ImageData":
    
    """
        Find the DF/Dt max using 1st derivative of a given average trace.

        Parameters
        ----------
        image : np.ndarray
            3D stack image of a single AP. ussually teh result from 
            averageing multiple APs.
        map_type = 0 for Aactvation time maps, 2 for APD maps

        Returns
        -------
        inverted_signal : np.ndarray
            The image with inverted fluorescence values

    """

    # 1. Get gradient and normalize it
    dfdt = np.gradient(image, axis=0)
    dfdt = (dfdt - np.nanmin(dfdt)) / np.nanmax(dfdt)
    dfdt = np.nan_to_num(dfdt, nan=0)

    activation_times = np.full_like(image[0, ...], fill_value= np.nan,  dtype=np.float64)
    
    ini_indices = np.nanargmax(dfdt, axis=0)
    
    #  may be a faster way to compute act time. just muktiply cycle_time by the index at max dfdt.
    activation_times[ini_indices != 0] = ini_indices[ini_indices != 0] * cycle_time

    if map_type == 0:

        if cycle_time == 1:
            return activation_times
        else:
            return activation_times * 1000 # convert to ms
    
    elif map_type ==2: 

        delta = 10 # this define the number of frames before and after the ini_indices
        
        # 2. get time vector
        n_frames, y_size, x_size = image.shape
        time = np.arange(0, n_frames) * cycle_time
        end_indx = np.argmax(time)
        # end_indx = np.min(np.append(np.argwhere(time >= peaks_times[peak] + bcl/1.5)[0], time.shape[-1] -2))

        # 3 find peak of signal
        # max_v = np.max(image, axis = 0)
        max_v_indx_stack = np.argmax(image, axis = 0)
        APD = np.full_like(ini_indices, fill_value= np.nan,  dtype=np.float64)
        
        mask_repol_indx_out = np.full_like(ini_indices, fill_value= np.nan,  dtype=np.uint16)
        t_index_out = np.full_like(ini_indices, fill_value= np.nan,  dtype=np.uint16)
        
        pre5ms_indx = int(np.ceil(0.005 / cycle_time))

                    
        # activation_times[activation_times == 0] = np.nan # remove zeros
        # 4 main loop
        for y_px  in range(y_size):
            for x_px in range(x_size):
                
                t_index = ini_indices[y_px, x_px]
                # from_ini_to_end_vector = image[t_index:, y_px, x_px]
                # trace_length_from_peak = len(from_ini_to_end_vector)
                # print(max_v_indx_stack.size)
                max_v_indx = max_v_indx_stack[y_px, x_px]
                
                # if  (t_index != 0) | (np.isnan(t_index)) and (from_ini_to_end_vector.size > 0):  #assert that you have a peak
                # if ~ np.isnan(image[t_index:, y_px, x_px]).any()  :  #assert that is not nan
                if (t_index != 0):
                    # print(t_index)
                    try:
                                        
                        Vm_signal = image[:, y_px, x_px]
                        end_rmp_indx = (t_index) - pre5ms_indx # 5 ms before upstroke
                        
                        resting_V = np.nanmean(image[:end_rmp_indx, y_px, x_px]) 
                        
                        amp_V = ( ((100 - percentage)/ 100) * (image[max_v_indx, y_px, x_px] - resting_V)) + resting_V
                        # print(f"amp_V: {amp_V}")
                        
                        mask_repol_indx =  np.argwhere(image[max_v_indx:, y_px, x_px]<= amp_V)
                        # if mask_repol_indx
                        mask_repol_indx =  max_v_indx + mask_repol_indx[0].min() if mask_repol_indx.size != 0 else max_v_indx
                        mask_repol_indx_out[y_px, x_px] = mask_repol_indx
                        t_index_out[y_px, x_px] = t_index
            
                        #get APD duration
                        len_segment = image[t_index:mask_repol_indx, y_px, x_px].size
                        # print(f"len_segment: {len_segment}")
                        APD[y_px, x_px] = len_segment * cycle_time
                        # print(f"APD: {len_segment * cycle_time_in_ms}")

                            
                    except Exception as e:
                        # print(mask_repol_indx.size, np.isnan(mask_repol_indx))
                        # print("you enter the exeception and are not printing")
                        warnings.warn(f"ERROR: Computing APD parameters fails witht error: {repr(e)}.")
                        # break
                        raise e


def crop_from_shape(shape_layer_data, img_layer):
    dshape = img_layer.data.shape

    # NOTE: you need to handel case for 2d images alike 3d images
    # NOTE: handle rgb images -> no so urgent
    # NOTE: handel 2d images
    
    # label = shape_layer.to_labels(dshape[-2:])
    # get vertices from shape: top-right, top-left, botton-left, botton-right
    tl, tr, bl, br = shape_layer_data.astype(int)
    y_ini, y_end = sorted([tr[-2], br[-2]])
    x_ini, x_end = sorted([tl[-1], tr[-1]])
    ini_index = [y_ini, x_ini ]
    end_index = [y_end, x_end]
    
    # parse the negative index to the minimum value
    for index, value in enumerate(ini_index):
        if value < 0:
            ini_index[index] = 0
    # for index, value in enumerate(x_index):
    #     if value < 0:
    #         x_index[index] = 0

    # parse the values out of range to the max
    y_max, x_max = dshape[-2], dshape[-1]
    if end_index[0] > y_max:
        end_index[0] = y_max
    
    if end_index[1] > x_max:
        end_index[1] = x_max

    cropped_img = img_layer.data.copy()

    cropped_img = cropped_img[:,ini_index[0]:end_index[0],  ini_index[1]:end_index[1]]

    return cropped_img, ini_index, end_index


def bounding_box_vertices(my_labels_data, area_threshold=1000, vertical_padding=0, horizontal_padding=0):
    """
    Create bounding box vertices with optional padding for each region in my_labels_data.
    The resulting bounding boxes are sorted from left to right based on their position.
    
    Parameters:
    - my_labels_data: The labeled image data.
    # - img_data: 2d intensity image data used for regioprops.
    - area_threshold: Minimum area to include a region.
    - vertical_padding: Pixels to increase bounding box height (top and bottom).
    - horizontal_padding: Pixels to increase bounding box width (left and right).
    
    Returns:
    - A sorted list of bounding box vertices suitable for drawing shapes in Napari.
    - A list of cropped regions (both image and label) based on the bounding boxes.
    """
    # Retrieve bounding boxes from regionprops
    bounding_boxes = [region.bbox for region in regionprops(label_image=my_labels_data) if region.area > area_threshold]

    # Sort bounding boxes from left to right based on min_col (second value in bbox)
    bounding_boxes.sort(key=lambda bbox: bbox[1])  # Sort by the second value (min_col)

    # Prepare lists to store bounding box vertices, cropped images, and cropped labels
    napari_boxes = []
    cropped_labels = []
    
    for bbox in bounding_boxes:
        # (min_row, min_col, max_row, max_col)
        min_row, min_col, max_row, max_col = bbox

        # Apply padding to bounding box
        min_row = max(min_row - vertical_padding, 0)  # Ensure not negative
        max_row += vertical_padding
        min_col = max(min_col - horizontal_padding, 0)  # Ensure not negative
        max_col += horizontal_padding

        # Create a numpy array of vertices in the required format
        box_vertices = np.array([
            [min_row, min_col],  # top-left corner
            [min_row, max_col],  # top-right corner
            [max_row, max_col],  # bottom-right corner
            [max_row, min_col],  # bottom-left corner
        ])

        napari_boxes.append(box_vertices)

        # Crop the label data corresponding to this bounding box
        cropped_label = my_labels_data[min_row:max_row, min_col:max_col]
        cropped_labels.append(cropped_label)

    return napari_boxes, cropped_labels


def crop_from_bounding_boxes(img_layer, my_labels_data, rotate_directions, area_threshold=1000, vertical_padding=0, horizontal_padding=0):
    """
    Crop regions from an image based on bounding boxes from labels.

    Parameters:
    - img_layer: The image layer from which to crop.
    - my_labels_data: Labeled data to generate bounding boxes.
    - rotate_directions: list of 4 elements (strings) showing the direction to rotate the image R (right) or L (left).
    - area_threshold: Minimum area to include a region.
    - vertical_padding: Pixels to increase bounding box height (top and bottom).
    - horizontal_padding: Pixels to increase bounding box width (left and right).

    Returns:
    - A list of cropped images corresponding to each bounding box.
    """
    dshape = img_layer.data.shape
    bounding_boxes, cropped_labels = bounding_box_vertices(my_labels_data, area_threshold, vertical_padding, horizontal_padding)
    
    cropped_images = []
    
    for indx, (box, label, direction)in enumerate(zip(bounding_boxes, cropped_labels, rotate_directions)):
        # Get top-left (tl) and bottom-right (br) corners
        tl, _, br, _ = box

        # Convert to integer values
        tl = tl.astype(int)
        br = br.astype(int)

        # Crop coordinates
        y_ini, x_ini = tl
        y_end, x_end = br

        # Ensure the coordinates are within valid ranges
        if y_ini < 0: y_ini = 0
        if x_ini < 0: x_ini = 0
        if y_end > dshape[-2]: y_end = dshape[-2]
        if x_end > dshape[-1]: x_end = dshape[-1]

        # Crop the image
        cropped_img = img_layer.data[:, y_ini:y_end, x_ini:x_end]
        if indx in [0, 1, 2]:
            cropped_img = np.rot90(cropped_img, axes=(1, 2))
            cropped_labels[indx] = np.rot90(label, axes=(0, 1))
        if indx == 3:
            cropped_img = np.rot90(cropped_img, axes=(2, 1))
            cropped_labels[indx] = np.rot90(label, axes=(1, 0))
            

        cropped_images.append((cropped_img, [y_ini, x_ini], [y_end, x_end]))

    return cropped_images, cropped_labels, bounding_boxes

def arrange_cropped_images(cropped_images, arrangement='horizontal', padding_value=0):
    """
    Arrange cropped images horizontally or vertically into a new array, padding smaller images as needed.

    Parameters:
    - cropped_images: List of tuples containing (cropped_image, ini_index, end_index).
                      Each cropped_image is a numpy array of the form (channels, height, width).
    - arrangement: 'horizontal' or 'vertical' arrangement of the images.
    - padding_value: The value to use for padding (default: 0, but could be NaN or any other value).
    
    Returns:
    - A new numpy array with the images arranged horizontally or vertically with padding.
    """
    # Get the maximum height and width among all cropped images
    max_height = max([img.shape[-2] for img, _, _ in cropped_images])
    max_width = max([img.shape[-1] for img, _, _ in cropped_images])

    # Initialize a list to hold the padded images
    padded_images = []

    for img, _, _ in cropped_images:
        # Get current image dimensions
        img_height, img_width = img.shape[-2], img.shape[-1]

        # Create a new array filled with padding_value and of the max size
        padded_img = np.full((img.shape[0], max_height, max_width), fill_value=padding_value, dtype=img.dtype)

        # Place the original image in the top-left corner of the padded array
        padded_img[:, :img_height, :img_width] = img

        padded_images.append(padded_img)

    # Stack the padded images horizontally or vertically
    if arrangement == 'horizontal':
        # Concatenate along the width axis
        new_image = np.concatenate(padded_images, axis=-1)
    elif arrangement == 'vertical':
        # Concatenate along the height axis
        new_image = np.concatenate(padded_images, axis=-2)
    else:
        raise ValueError("Arrangement must be either 'horizontal' or 'vertical'")

    return new_image


def return_APD_maps(image: "napari.types.ImageData", cycle_time,  interpolate_df = False, percentage = 75):
    
    n_frames, y_size, x_size = image.shape
    
    # Find peak and resting membrane voltages
    
    max_v = np.nanmax(image, axis = 0)
    APD = np.full_like(max_v, fill_value= np.nan,  dtype=np.float64)

    for y_px  in progress(range(y_size)):
        for x_px in range(x_size):

            print("lalal")





def segment_image_triangle(np_array, 
                 ):
    """Segment an image using an intensity threshold determined via
    triangle method.

    Parameters
    ----------
    image : np.ndarray
        The image to be segmented

    Returns
    -------
    label_image : np.ndarray
        The resulting image where each detected object labeled with a unique integer.
    
    cleared : np.ndarray bool
        Mask as a boolean array of the detected segments

    selection : np.ndarray
        Image with segments and bacgorund removed.
    """
    # create copy of dataset
    # masked_image = np_array.copy()
    # n_frames = masked_image.shape[0]
    
    # select one frame
    # one_frame_img = np_array[0]

    # 1. make a elevation map
    # elevation_map = sobel(one_frame_img)

    # Global equalize histogram to enhance contrast (ok)
    
    # img_glo_eq = exposure.equalize_hist(elevation_map)
    # # Local equalize histogram to enhance contrast (not good)
    # footprint = disk(disk_s)
    # img_loc_eq = rank.equalize(elevation_map, footprint=footprint)
    
    # 2. Local equalize histogram to enhance contrast (best)
    # img_adapteq = exposure.equalize_adapthist(elevation_map, clip_limit=clip_limit_s)
    
    
    # 3. apply threshold
    # thresh = threshold_otsu(img_adapteq)
    # thresh = threshold_li(one_frame_img)
    thresh = threshold_triangle(np.nan_to_num(np_array))
    # # thresh = threshold_sauvola(one_frame_img, window_size=wind_s)
    # # thresh = threshold_niblack(one_frame_img, window_size=wind_s)

    # 4.  create mask
    # mask = one_frame_img > thresh
    mask = np_array > thresh
    # bw = closing(mask, square(square_s))

    # # remove artifacts connected to image border
    
    # 5. remove small objects from aoutside heart siluete
    # cleared = remove_small_objects(mask, small_obj_s)
    
    # # 6. fill smaall holes ininside regions objects
    # footprint=[(np.ones((small_holes_s, 1)), 1), (np.ones((1, small_holes_s)), 1)]
    # cleared = binary_closing(cleared, footprint=footprint)
    # # cleared = clear_border(bw)

    # # 7.  label image regions
    # label_image = label(cleared)

    # 8. remove background using mask
    # selection[~np.tile(cleared, (n_frames, 1, 1))] = None

    # # 9. subtract bacground from original image 
    # background = selection[np.tile(cleared, (n_frames, 1, 1))].mean()
    
    # selection = selection - background

    # return label_image
    return mask



def segment_image_GHT(image, threshold=None, return_threshold=False,
                       small_obj_s = 500,  
                #   square_s = 15,  
                  small_holes_s = 5,):
    """Create a foreground mask for an image using a threshold.

    If no threshold is given, the background threshold is detected using the GHT algorithm :cite:p:`Barron2020`.

    Parameters
    ----------
    image : np.ndarray
        Image to create background mask for.
    threshold : float or int, optional
        Background threshold, by default None
    return_threshold : bool, optional
        If True, return the threshold as well, by default False

    Returns
    -------
    mask : np.ndarray
        Background mask.
    threshold : float or int
        Background threshold, only if ``return_threshold`` is True.
    """
    
    if threshold is None:
        threshold = detect_background_threshold(image)
        # print(f"Creating mask with detected threshold {threshold}")

    mask = image > threshold
    
    if return_threshold:
        return mask, threshold
    else:
        return mask


def segement_region_based_func(array_2d, lo_t = 0.05, hi_t = 0.2, expand = None):
    egdes = sobel(array_2d)
    
    markers = np.zeros_like(array_2d, dtype=np.uint)
    foreground, background = 1, 2
    markers[array_2d < lo_t] = foreground
    markers[array_2d > hi_t] = background

    ws = segmentation.watershed(egdes, markers)
    mask = label(ws == foreground)
    mask = binary_fill_holes(mask)
    if expand:
        mask = segmentation.expand_labels(mask, distance=expand)
    
    # return markers
    # print(type(mask))
    return (mask)


def polish_mask(mask, small_obj_s = 1000, small_holes_s = 5):
    
    # Number of pixels you want to reduce from the edges
    erosion_size = 3 

    # Pad the mask using numpy.pad
    padded_mask = np.pad(mask, pad_width=erosion_size, mode='constant', constant_values=0)
    # Apply erosion with a square or custom structuring element
    eroded_mask = morphology.binary_erosion(padded_mask, disk(erosion_size))
    # Remove the padding to get the mask back to its original size
    cleared = eroded_mask[erosion_size:-erosion_size, erosion_size:-erosion_size]
    
    cleared = remove_small_objects(cleared, small_obj_s)
    
    # 6. fill smaall holes ininside regions objects
    footprint=[(np.ones((small_holes_s, 1)), 1), (np.ones((1, small_holes_s)), 1)]
    cleared = binary_closing(cleared, footprint=footprint)
    cleared = segmentation.clear_border(cleared)

    # 7.  label image regions
    label_image = label(cleared)

    return label_image

def optimap_mot_correction(np_array, c_k, pre_smooth_t, proe_smooth_s, ref_fr):
    video_warped  =  motion_compensate(video = np_array, 
                                       contrast_kernel= c_k, 
                                       ref_frame=ref_fr,
                                       presmooth_temporal=pre_smooth_t, 
                                       presmooth_spatial=proe_smooth_s)

    return video_warped


def gaussian_filter_nan(array, sigma=1, radius=3, axes=(0, 1), truncate = 4):
    
    # taken from this post https://stackoverflow.com/questions/18697532/gaussian-filtering-a-image-with-nan-in-python
    
    # sigma=2.0                  # standard deviation for Gaussian kernel
    # truncate=4.0               # truncate filter at this many sigmas
    U = array
    # U=sp.randn(10,10)          # random array...
    # U[U>2]=np.nan              # ...with NaNs for testing
    
    V=U.copy()
    V[np.isnan(U)]=0
    VV=gaussian_filter(V,sigma=sigma, radius= radius, axes = axes, truncate=truncate)
    
    W=0*U.copy()+1
    W[np.isnan(U)]=0
    WW=gaussian_filter(W,sigma=sigma, radius= radius, axes = axes, truncate=truncate)
    WW[WW==0]=np.nan
    
    return VV/WW


def convert_to_json_serializable(obj):
    if isinstance(obj, bytes):
        return obj.decode('utf-8')  # Convert bytes to string
    elif isinstance(obj, tuple):
        return list(obj)  # Convert tuple to list
    elif isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}  # Recursively convert dict
    elif isinstance(obj, list):
        return [convert_to_json_serializable(i) for i in obj]  # Recursively convert list
    return obj  # Return the object if it is already JSON serializable

# def decodeDictionary(dictionary):
#     # if type(dictionary) == dict:
#     if isinstance(dictionary, dict):
        
#         for key in dictionary.keys():
#             dictionary[key] = decodeDictionary(dictionary[key])
    
#     elif isinstance(**dictionary**, bytes):
        
#         dictionary = dictionary.decode('UTF-8') 
    
#     # elif type(**dictionary**) == bytes:
#     return dictionary


# this class helper allow to make gorup layouts easily"
class VHGroup():
    """Group box with specific layout.

    Parameters
    ----------
    name: str
        Name of the group box
    orientation: str
        'V' for vertical, 'H' for horizontal, 'G' for grid
    """

    def __init__(self, name, orientation='V'):
        self.gbox = QGroupBox(name)
        if orientation=='V':
            self.glayout = QVBoxLayout()
        elif orientation=='H':
            self.glayout = QHBoxLayout()
        elif orientation=='G':
            self.glayout = QGridLayout()
        else:
            raise Exception(f"Unknown orientation {orientation}") 

        self.gbox.setLayout(self.glayout)


class PandasModel(QAbstractTableModel):
    """A model to interface a Qt view with pandas dataframe """

    def __init__(self, dataframe: pd.DataFrame, parent=None):
        QAbstractTableModel.__init__(self, parent)
        self._dataframe = dataframe

    def rowCount(self, parent=QModelIndex()) -> int:
        """ Override method from QAbstractTableModel

        Return row count of the pandas DataFrame
        """
        if parent == QModelIndex():
            return len(self._dataframe)

        return 0

    def columnCount(self, parent=QModelIndex()) -> int:
        """Override method from QAbstractTableModel

        Return column count of the pandas DataFrame
        """
        if parent == QModelIndex():
            return len(self._dataframe.columns)
        return 0

    def data(self, index: QModelIndex, role=Qt.ItemDataRole):
        """Override method from QAbstractTableModel

        Return data cell from the pandas DataFrame
        """
        if not index.isValid():
            return None

        if role == Qt.DisplayRole:
            return str(self._dataframe.iloc[index.row(), index.column()])

        return None

    def headerData(
        self, section: int, orientation: Qt.Orientation, role: Qt.ItemDataRole
    ):
        """Override method from QAbstractTableModel

        Return dataframe index as vertical header data and columns as horizontal header data.
        """
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._dataframe.columns[section])

            if orientation == Qt.Vertical:
                return str(self._dataframe.index[section])

        return None



class ToggleButton(QCheckBox):
    def __init__(
        self,
        width=140,
        bgColor="#777",
        circleColor="#DDD",
        activeColor="#00BCff",
        animationCurve=QEasingCurve.OutBounce,
    ):
        QCheckBox.__init__(self)
        self.setFixedSize(width, 25)
        self.setCursor(Qt.PointingHandCursor)

        self._bg_color = bgColor
        self._circle_color = circleColor
        self._active_color = activeColor
        self._circle_position = 3
        self.animation = QPropertyAnimation(self, b"circle_position")

        self.animation.setEasingCurve(animationCurve)
        self.animation.setDuration(100)
        self.stateChanged.connect(self.start_transition)

    @Property(int)
    def circle_position(self):
        return self._circle_position

    @circle_position.setter
    def circle_position(self, pos):
        self._circle_position = pos
        self.update()

    def start_transition(self, value):
        self.animation.setStartValue(self.circle_position)
        if value:
            self.animation.setEndValue(self.width() - 20 )
        else:
            self.animation.setEndValue(3)
        self.animation.start()

    def hitButton(self, pos: QPoint):
        return self.contentsRect().contains(pos)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        p.setPen(Qt.NoPen)

        rect = QRect(0, 0, self.width(), self.height())

        if not self.isChecked():
            p.setBrush(QColor(self._bg_color))
            p.drawRoundedRect(
                0, 0, rect.width(), self.height(), self.height() / 2, self.height() / 2
            )

            p.setBrush(QColor(self._circle_color))
            p.drawEllipse(self._circle_position, 3, 16, 16)
        else:
            p.setBrush(QColor(self._active_color))
            p.drawRoundedRect(
                0, 0, rect.width(), self.height(), self.height() / 2, self.height() / 2
            )

            p.setBrush(QColor(self._circle_color))
            p.drawEllipse(self._circle_position, 3, 16, 16)


class MultiComboBox(QComboBox):
    """
    MultiComboBox this class help to create dropdown 
    checkable QCombobox

    _extended_summary_
    source . https://stackoverflow.com/questions/76680387/do-a-multi-selection-in-dropdown-list-in-qt-python

    Parameters
    ----------
    QComboBox : _type_
        _description_
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setEditable(True)
        self.lineEdit().setReadOnly(True)
        self.setModel(QStandardItemModel(self))

        # Connect to the dataChanged signal to update the text
        self.model().dataChanged.connect(self.updateText)

    def addItem(self, text: str, data=None):
        item = QStandardItem()
        item.setText(text)
        item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
        item.setData(Qt.CheckState.Unchecked, Qt.ItemDataRole.CheckStateRole)
        self.model().appendRow(item)

    def addItems(self, items_list: list):
        for text in items_list:
            self.addItem(text)

    def updateText(self):
        selected_items = [self.model().item(i).text() for i in range(self.model().rowCount())
                          if self.model().item(i).checkState() == Qt.CheckState.Checked]
        self.lineEdit().setText(", ".join(selected_items))

    def showPopup(self):
        super().showPopup()
        # Set the state of each item in the dropdown
        for i in range(self.model().rowCount()):
            item = self.model().item(i)
            combo_box_view = self.view()
            combo_box_view.setRowHidden(i, False)
            check_box = combo_box_view.indexWidget(item.index())
            if check_box:
                check_box.setChecked(item.checkState() == Qt.CheckState.Checked)

    def hidePopup(self):
        # Update the check state of each item based on the checkbox state
        for i in range(self.model().rowCount()):
            item = self.model().item(i)
            combo_box_view = self.view()
            check_box = combo_box_view.indexWidget(item.index())
            if check_box:
                item.setCheckState(Qt.CheckState.Checked if check_box.isChecked() else Qt.CheckState.Unchecked)
        super().hidePopup()


class TrackProcessingSteps:
    def __init__(self):
        self.steps = []
    
    def add_step(self, operation, method_name, inputs, outputs, parameters):
        steps = {
            'id': len(self.steps) + 1,
            'operation': operation,
            'method_name': method_name,
            'inputs': inputs,
            'outputs': outputs,
            'parameters': parameters,
            'timestamp': datetime.datetime.now().isoformat(),
        }
        self.steps.append(steps)

    def save_to_yaml(self, file_path):
        with open(file_path, "w") as f:
            yaml.dump({"ProcessingSteps": {"steps": self.steps}}, f, sort_keys=False)

    def save_to_toml(self, file_path):
        with open(file_path, "w") as f:
            toml.dump({"ProcessingSteps": {"steps": self.steps}}, f)

    def save_to_tiff(self, image, metadata, file_path):
        # metadata = {
        #     "ProcessingSteps": {"steps": self.steps}
        # }
        imwrite(file_path, image, photometric='minisblack', metadata=metadata)
        # imwrite(
        #     # shape=(8, 800, 600),
        #     file_path,
        #     # dtype='uint16',
        #     photometric='minisblack',
        #     # tile=(128, 128),
        #     metadata=metadata
        # )

