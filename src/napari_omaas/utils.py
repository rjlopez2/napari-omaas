from qtpy import QtCore, QtGui
import numpy as np
from skimage.filters import gaussian, threshold_triangle, median, rank
from skimage.morphology import disk
from skimage.registration import optical_flow_ilk
from skimage import transform
from skimage import morphology, registration, segmentation
import warnings
from napari.layers import Image
import sif_parser
from numba import njit
import tqdm.auto as tqdm
from napari.utils import progress


from napari_macrokit import get_macro

# from numba import jit, prange
from scipy import signal, ndimage
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter
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
from qtpy.QtWidgets import QHBoxLayout, QVBoxLayout, QGroupBox, QGridLayout
from qtpy.QtCore import Qt, QAbstractTableModel, QModelIndex


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
    data: "napari.types.ImageData")-> "napari.types.ImageData":

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
    results = (data - np.min(data, axis = 0)) / np.max(data, axis=0)
    results = np.nan_to_num(results, nan=0)
    return results


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
    return gaussian_filter(data, sigma=sigma, order= 0, radius=kernel_size, axes = (1,2))

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
    out_img = np.empty_like(data)
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
    
    out_img = ndimage.median_filter(data, size = (1, param, param))
    # return (gaussian(data, sigma))
    return out_img


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
def compute_APD_props_func(np_1Darray, curr_img_name, diff_n = 1, cycle_length_ms = 0.004, rmp_method = "bcl_to_bcl", apd_perc = 75, promi = 0.18, roi_indx = 0):
    
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
    resting_V = np.zeros_like(peaks_times)
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
        delta = 10;
        if upstroke_indx - delta <= 0:
            lower_bound_interp = 1;
        else:
            lower_bound_interp = upstroke_indx - delta;
        
        if upstroke_indx + delta > time.shape[-1]:
            upper_bound_interp = time.shape[-1]
        else:
            upper_bound_interp = upstroke_indx + delta;
        
        time_fine_grid = np.linspace(time[lower_bound_interp], time[upper_bound_interp], interp_points)
        interpolation_f = CubicSpline(time[lower_bound_interp :  upper_bound_interp], dfdt[lower_bound_interp :  upper_bound_interp], extrapolate=True)
        dfdt_interpolated = interpolation_f(time_fine_grid) 
        # find new dfdt max
        dfdt_max, dfdt_max_indx = np.max(dfdt_interpolated), np.argmax(dfdt_interpolated)
        activation_time[peak] = time_fine_grid[dfdt_max_indx]
        dVdtmax[peak] = dfdt_max * cycle_length_ms

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
        # enerate fine grid for interpolation in ms
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
    ROI_ids = [f'ROI_{roi_indx}' for i in range(peaks_times.shape[-1])]
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
    data, info = sif_parser.np_spool_open(path)
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