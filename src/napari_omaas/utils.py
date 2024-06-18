from qtpy import QtCore, QtGui
import numpy as np
from skimage.filters import gaussian, threshold_triangle, median, rank, sobel
from skimage.measure import label
from skimage.filters.rank import mean_bilateral
from skimage.morphology import disk, binary_closing, remove_small_objects, closing
from skimage.registration import optical_flow_ilk
from skimage import transform, exposure, morphology, registration, segmentation
from skimage.restoration import denoise_bilateral
import warnings
from napari.layers import Image
import sif_parser
# from numba import njit
import tqdm.auto as tqdm
from napari.utils import progress
from time import time
from optimap.image import detect_background_threshold
from optimap import motion_compensate
from optimap.video import normalize_pixelwise_slidingwindow, normalize_pixelwise

from napari_macrokit import get_macro

# from numba import jit, prange
from scipy import signal, ndimage
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter, binary_fill_holes
import cupy as cp
from cupyx.scipy.ndimage import median_filter as cp_median_filter
from cupyx.scipy.ndimage import gaussian_filter as cp_gaussian_filter
# functions

from napari.types import ImageData, ShapesData
import pandas as pd
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
from qtpy.QtWidgets import QHBoxLayout, QVBoxLayout, QGroupBox, QGridLayout, QCheckBox
from qtpy.QtCore import Qt, QAbstractTableModel, QModelIndex, QRect, QPropertyAnimation, QPoint, QEasingCurve, Property
# from qtpy.QtCore import *
from qtpy.QtGui import QColor, QPainter


# instanciate a macro object
macro = get_macro("OMAAS_analysis")

@macro.record
def invert_signal(
    data: "napari.types.ImageData"
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

    return np.nanmax(data) - data
   
@macro.record
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

@macro.record
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


@macro.record
def slide_window_normalization_func(np_array, slide_window = 20):
    # NOTE: you have to check an error here.
    return normalize_pixelwise_slidingwindow(np_array, window_size=  slide_window)


@macro.record
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

@macro.record
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

@macro.record
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

    # return (gaussian(data, sigma))
    # return out_img
    # return gaussian_filter(data, sigma=sigma, order= 0, radius=kernel_size, # axes = (1,2))
    data_cp = cp.asarray(data)
    out_img = cp_gaussian_filter(data_cp, 
                              sigma=(0, sigma, sigma), 
                              order= 0,
                              # radius=kernel_size,
                              truncate=kernel_size,
                              #axes = (1,2)
                              )
    return cp.asnumpy(out_img)


@macro.record
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
    param = int(param)
    # data = image.active.data
    # out_img = np.empty_like(data)
    footprint = disk(int(param))

    # print(f'applying "apply_median_filt_func" to image {image.active}')

    # for plane, img in enumerate(data):
        # out_img[plane] = median(img, footprint = footprint)

    # for plane in list(range(data.shape[0])):
    #     out_img[plane, :, :] = median(data[plane, :, :], footprint = footprint)

# using numba method #
    # out_img = parallel_median(data,footprint)
    # for plane, img in enumerate(data):
    #     out_img[plane] = signal.medfilt2d(img, kernel_size = param)
    
    # out_img = ndimage.median_filter(data, size = (1, param, param))

    
    data_cp = cp.asarray(data)
    # xp = cpx.get_array_module(data_cp, param)  # 'xp' is a standard usage in the community
    # print("Using:", xp.__name__)
    out_img = cp_median_filter(data_cp, size = (1, param, param))
    
    return cp.asnumpy(out_img)


@macro.record
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

@macro.record
def scaled_img_func(data, foot_print_size = 10):
    foot_print = disk(foot_print_size)
    
    xp = cp.get_array_module(data)
    xpx = cupyx.scipy.get_array_module(data)
    
    # data = xp.asarray(img)
    print(f"Using: {xp.__name__} in scaled_img_func")
    
    scaled_img = xp.empty_like(data, dtype= (xp.float64))
    
    
    for plane, img in enumerate(data):
        # if plane % 50 == 0:
        #     print(f"normalizing plane: {plane}")
            
        im_min = xpx.ndimage.minimum_filter(img, footprint=foot_print)
        im_max =xpx.ndimage.maximum_filter(img, footprint=foot_print)
        scaled_img[plane,...] = xp.divide(xp.subtract(img, im_min),  xp.subtract(im_max, im_min))
        
    
    return scaled_img

@macro.record
def register_img_func(data, orig_data, ref_frame = 1, radius_size = 7, num_warp = 8):
    xp, xpx = cp.get_array_module(data), cupyx.scipy.get_array_module(data)
    
    if type(data) == cp.ndarray:
        device_type = "GPU"
        register_func = registration_gpu
        transform_func = transform_gpu
    else:
        device_type = "CPU"
        register_func = registration
        transform_func = transform
        
    print (f'using device: {device_type}')
    
    ref_frame_data = data[ref_frame, ...]
    nr, nc = ref_frame_data.shape
    registered_img = xp.empty_like(data)
    # print(type(registered_img))
        
    for plane, img in enumerate(data):
        # if plane % 50 == 0:
            # print(f"Registering plane: {plane}")
        
        v, u = register_func.optical_flow_ilk(ref_frame_data, img, 
                                              radius = radius_size, num_warp=num_warp)
        row_coords, col_coords = xp.meshgrid(xp.arange(nr), xp.arange(nc),
                                             indexing='ij')
        registered_img[plane, ...] = transform_func.warp(orig_data[plane], xp.array([row_coords + v, col_coords + u]),
                                                         mode='edge', preserve_range=True)
    
    return registered_img


@macro.record
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


@macro.record
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


@macro.record
def apply_box_filter(data: "napari.types.ImageData", kernel_size):

    # data = image.active.data
    out_img = np.empty_like(data)

    box_kernel2d = np.ones((kernel_size, kernel_size))/kernel_size**2

    for plane, img in enumerate(data):
        out_img[plane] = signal.oaconvolve(img, box_kernel2d, mode="same")

    # print(f'applying "apply_box_filter" to image {image.active}')

    return (out_img)


@macro.record
def apply_laplace_filter(data: "napari.types.ImageData", kernel_size, sigma):
    
    # data = image.active.data
    out_img = np.empty_like(data)

    mex_hat_kernel1d = signal.ricker(kernel_size, sigma)
    mex_hat_kernel2d = (mex_hat_kernel1d[:, None] @ mex_hat_kernel1d[None]) * -1

    for plane, img in enumerate(data):
        out_img[plane] = signal.oaconvolve(img, mex_hat_kernel2d, mode="same")

    # print(f'applying "apply_laplace_filter" to image {image.active}')

    return (out_img)


@macro.record
def apply_bilateral_filter(data: "napari.types.ImageData", wind_size, sigma_col, sigma_spa):

    out_img = np.empty_like(data)
    start = time()
    for plane, img in enumerate(progress(data)):
        # out_img[plane] = mean_bilateral(data[plane], disk(disk_size), s0=5, s1=5) this requires the image to be a uint8/uint16
        out_img[plane] = denoise_bilateral(data[plane], 
                                           win_size = wind_size, 
                                           sigma_color = sigma_col, 
                                           sigma_spatial = sigma_spa, 
                                           bins = 1024)
    end = time()
    print(f"elapsed time: {round((end - start)/60, 1)} min")
    return (out_img)

@macro.record
def compute_APD_props_func(np_1Darray, curr_img_name, cycle_length_ms, diff_n = 1, rmp_method = "bcl_to_bcl", apd_perc = 75, promi = 0.18, roi_indx = 0, roi_id = None, interpolate = False):
    
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
    
    AP_ini = []
    AP_peak = []
    AP_end = [] 

    for peak in range(len(peaks_times)):
        #  Find RMP and APD
        # RMP is mean V of BCL/2 to 5ms before AP upstroke to when this value is crossed to BCL/2 (or end) 

        bcl = bcl_list[peak]
        # Find interval of AP in indices
        ini_indx = np.max(np.append(np.argwhere(time >= peaks_times[peak] - bcl/1.5)[0], 0))
        
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
            dVdtmax[peak] = dfdt[upstroke_indx] * cycle_length_ms


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
        repol_index =  AP_peaks_indx[peak] + np.minimum(np.argmax(current_APD_segment <= amp_V) , current_APD_segment.shape[-1] -1)
        pre_repol_index = repol_index - 2
        # generate fine grid for interpolation in ms
        time_fine_grid = np.linspace(time[pre_repol_index], time[repol_index], interp_points)
        Vm_interpolated = np.interp(time_fine_grid, time[pre_repol_index:repol_index], np_1Darray[pre_repol_index:repol_index])
        # repol_index_interpolated = np.nanmin( np.append(np.array(time_fine_grid.size -1 ), np.argwhere(Vm_interpolated <= amp_V)))
        repol_index_interpolated = np.append(np.argwhere(Vm_interpolated <= amp_V), time_fine_grid.size -1 ).min()

        repol_time[peak] = time_fine_grid[repol_index_interpolated] #* 1000
        APD[peak] = repol_time[peak] - activation_time[peak]

        AP_ini.append(upstroke_indx) 
        AP_peak.append(AP_peaks_indx[peak]) 
        AP_end.append(repol_index ) 



    AP_ids = [f'AP_{i}' for i in range(peaks_times.shape[-1])]
    if not roi_id:
        ROI_ids = [f'ROI-{roi_indx}' for i in range(peaks_times.shape[-1])]
    else:
        ROI_ids = [f'ROI-{roi_id}' for i in range(peaks_times.shape[-1])]
        
    apd_perc = [apd_perc for i in range(peaks_times.shape[-1])]
    img_name = [curr_img_name for i in range(peaks_times.shape[-1])]


    rslt_df = [img_name, 
                ROI_ids, 
                AP_ids, 
                apd_perc, 
                APD, 
                dVdtmax, 
                amp_Vmax, 
                bcl_list, 
                resting_V,
                time[AP_ini], 
                time[AP_peak], 
                time[AP_end], 
                AP_ini, 
                AP_peak, 
                AP_end]
     
    # rslt_df = rslt_df.apply(lambda x: np.round(x * 1000, 2) if x.dtypes == "float64" else x ) # convert to ms and round values
    return (rslt_df)

def return_spool_img_fun(path):
    data, info = sif_parser.np_spool_open(path, multithreading= True, max_workers=16)
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
    

@macro.record
def split_AP_traces_func(trace, ini_i, end_i, type = "1d", return_mean = False):
    """
    This function takes a 1d or 3D array, ini index, end index of ap 
    previously computed with function 'return_AP_ini_end_indx_func' 
    and return the splitted arrays for each AP.
    """
    # must check that all len are the same
    n_peaks = len(ini_i)
    
    splitted_traces = [[trace[ini:end, ...]] for ini, end in zip(ini_i, end_i)]
    # take the small trace len and adjust the other traces to that
    min_dim = np.min([trace[0].shape for trace in splitted_traces])
    splitted_traces = [trace[0][:min_dim] for trace in splitted_traces]

    if type == "1d":
        splitted_traces = np.array(splitted_traces).reshape(n_peaks, -1)
    
    elif type == "3d":
        img_dim_x, img_dim_y = trace.shape[-2:]
        splitted_traces = np.array(splitted_traces).reshape( n_peaks, -1, img_dim_x, img_dim_y)
    

    if return_mean:
        splitted_traces = np.mean(np.array([trace for trace in splitted_traces]), axis=(0))
        
    
    return splitted_traces


@macro.record
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

@macro.record
def crop_from_shape(shape_layer, img_layer):
    dshape = img_layer.data.shape

    # NOTE: you need to handel case for 2d images alike 3d images
    # NOTE: handle rgb images -> no so urgent
    # NOTE: handel 2d images
    
    # label = shape_layer.to_labels(dshape[-2:])
    # get vertices from shape: top-right, top-left, botton-left, botton-right
    tl, tr, bl, br = shape_layer.data[0].astype(int)
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


@macro.record
def return_APD_maps(image: "napari.types.ImageData", cycle_time,  interpolate_df = False, percentage = 75):
    
    n_frames, y_size, x_size = image.shape
    
    # Find peak and resting membrane voltages
    
    max_v = np.nanmax(image, axis = 0)
    APD = np.full_like(max_v, fill_value= np.nan,  dtype=np.float64)

    for y_px  in progress(range(y_size)):
        for x_px in range(x_size):

            print("lalal")




@macro.record
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
    thresh = threshold_triangle(np_array)
    # # thresh = threshold_sauvola(one_frame_img, window_size=wind_s)
    # # thresh = threshold_niblack(one_frame_img, window_size=wind_s)

    # 4.  create mask
    # mask = one_frame_img > thresh
    mask = np_array.mean(axis = 0) > thresh
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


@macro.record
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
        print(f"Creating mask with detected threshold {threshold}")

    mask = image.mean(axis = 0) > threshold
    
    if return_threshold:
        return mask, threshold
    else:
        return mask

@macro.record
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
    print(type(mask))
    return (mask)

@macro.record
def polish_mask(mask, small_obj_s = 500, small_holes_s = 5):

    cleared = remove_small_objects(mask, small_obj_s)
    
    # 6. fill smaall holes ininside regions objects
    footprint=[(np.ones((small_holes_s, 1)), 1), (np.ones((1, small_holes_s)), 1)]
    cleared = binary_closing(cleared, footprint=footprint)
    # cleared = clear_border(bw)

    # 7.  label image regions
    label_image = label(cleared)

    return label_image
@macro.record
def optimap_mot_correction(np_array, c_k, pre_smooth_t, proe_smooth_s, ref_fr):
    video_warped  =  motion_compensate(video = np_array, 
                                       contrast_kernel= c_k, 
                                       ref_frame=ref_fr,
                                       presmooth_temporal=pre_smooth_t, 
                                       presmooth_spatial=proe_smooth_s)

    return video_warped




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