"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
from typing import TYPE_CHECKING

from magicgui import magic_factory
from superqt import (
    QCollapsible, QLabeledSlider, QLabeledRangeSlider, 
    QRangeSlider, QDoubleRangeSlider, QLabeledDoubleRangeSlider,
)
from qtpy.QtWidgets import (
    QHBoxLayout, QPushButton, QWidget, QFileDialog, 
    QVBoxLayout, QGroupBox, QGridLayout, QTabWidget, QListWidget,
    QDoubleSpinBox, QLabel, QComboBox, QSpinBox, QLineEdit, QPlainTextEdit,
    QTreeWidget, QTreeWidgetItem, QCheckBox, QSlider, QTableView, QMessageBox, QToolButton, 
    )

from qtpy import QtWidgets, QtCore
from warnings import warn
from qtpy.QtCore import Qt
from qtpy.QtGui import QIntValidator, QDoubleValidator
from napari_matplotlib.base import BaseNapariMPLWidget
from napari.layers import Shapes, Image, Labels
from napari.components.layerlist import LayerList
from napari.utils import progress

from skimage.morphology import disk, binary_closing, remove_small_objects, closing, erosion, reconstruction

import copy
import pandas as pd

import numpy as np
from .custom_exceptions import CustomException
import sys
import re
from .utils import (
    VHGroup,
    ToggleButton,
    PandasModel,
    MultiComboBox,
    TrackProcessingSteps,
    
    invert_signal,
    local_normal_fun,
    slide_window_normalization_func,
    global_normal_fun,
    split_channels_fun,
    apply_gaussian_func,
    apply_median_filt_func,
    apply_box_filter,
    apply_laplace_filter,
    apply_bilateral_filter,
    transform_to_unit16_func,
    apply_butterworth_filt_func,
    segment_heart_func,
    pick_frames_fun,
    return_peaks_found_fun,
    compute_APD_props_func,
    return_spool_img_fun,
    extract_ROI_time_series,
    return_AP_ini_end_indx_func,
    split_AP_traces_and_ave_func,
    segment_image_triangle,
    segment_image_GHT,
    polish_mask,
    segement_region_based_func,
    optimap_mot_correction,
    crop_from_shape,
    concatenate_and_padd_with_nan_2d_arrays,
    return_maps,
    apply_FIR_filt_func,
    gaussian_filter_nan,
    # decodeDictionary,
    convert_to_json_serializable,
    bounding_box_vertices,
    crop_from_bounding_boxes, 
    arrange_cropped_images

)

import os
import sys
from pathlib import Path
import h5py
import tifffile


if TYPE_CHECKING:
    import napari


class OMAAS(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # in one of two ways:
    # 1. use a parameter called `napari_viewer`, as done here
    # 2. use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.viewer.text_overlay.visible = True
        self.viewer.text_overlay.font_size = 36


        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)

        ###############################
        ######## create tabs ##########
        ###############################

        ######## pre-processing tab ########
        self.pre_processing_tab = QWidget()
        self._pre_processing_layout = QVBoxLayout()
        self.pre_processing_tab.setLayout(self._pre_processing_layout)
        self.tabs.addTab(self.pre_processing_tab, 'Pre-processing')
       
       ######## Shapes tab ########
        self.layers_processing = QWidget()
        self._layers_processing_layout = QGridLayout()
        self.layers_processing.setLayout(self._layers_processing_layout)
        self.tabs.addTab(self.layers_processing, 'Shapes') # this tab is not making the GUI fat, it's ok!

        ######## Mot-Correction tab ########
        self.motion_correction = QWidget()
        self._motion_correction_layout = QVBoxLayout()
        self.motion_correction.setLayout(self._motion_correction_layout)
        self.tabs.addTab(self.motion_correction, 'Mot-Correction') # this tab is just ok!

        ######## Mapping tab ########
        self.mapping_processing = QWidget()
        self._mapping_processing_layout = QVBoxLayout()
        self.mapping_processing.setLayout(self._mapping_processing_layout)
        self.tabs.addTab(self.mapping_processing, 'Mapping') # this tab is just ok!

        ######## APD analysis tab ########
        self.APD_analysis = QWidget()
        self._APD_analysis_layout = QVBoxLayout()
        self.APD_analysis.setLayout(self._APD_analysis_layout)
        self.tabs.addTab(self.APD_analysis, 'APD analysis') # this one makes the GUI fat!

        ######## Settings tab ########
        self.settings = QWidget()
        self._settings_layout = QVBoxLayout()
        self.settings.setLayout(self._settings_layout)
        self.tabs.addTab(self.settings, 'Settings') # this tab is just ok!

        self.metadata_recording_steps = TrackProcessingSteps()

        #########################################
        ######## Editing indivicual tabs ########
        #########################################

        ######## Pre-processing tab ########
        ####################################
        self._pre_processing_layout.setAlignment(Qt.AlignTop)
        
        ######## pre-processing  group ########
        self.pre_processing_group = VHGroup('Pre-porcessing', orientation='G')

        ######## pre-processing btns ########

        self.apply_normalization_btn = QPushButton("Normalize")
        self.apply_normalization_btn.setToolTip(("Apply Normalization to the current Image."))
        self.pre_processing_group.glayout.addWidget(self.apply_normalization_btn, 1, 1, 1, 1)

        self.data_normalization_options = QComboBox()
        self.data_normalization_options.addItems(["Local max", "Slide window", "Global"])
        self.data_normalization_options.setToolTip(("List of normalization methods."))
        self.pre_processing_group.glayout.addWidget(self.data_normalization_options, 1, 2, 1, 1)

        self.inv_and_norm_data_btn = QPushButton("Invert + Normalize")        
        self.inv_and_norm_data_btn.setToolTip(("Invert and Apply Normalization to the current Image."))
        self.pre_processing_group.glayout.addWidget(self.inv_and_norm_data_btn, 1, 3, 1, 1)
        
        self.inv_data_btn = QPushButton("Invert signal")
        self.inv_data_btn.setToolTip(("Invert the polarity of the signal"))
        self.pre_processing_group.glayout.addWidget(self.inv_data_btn , 2, 1, 1, 1)

        self.slide_wind_n = QSpinBox()
        self.slide_wind_n.setToolTip(("Windows size for slide window normalization method."))
        # self.slide_wind_n.setSingleStep(1)
        self.slide_wind_n.setValue(100)
        self.slide_wind_n.setMaximum(10000000)
        self.pre_processing_group.glayout.addWidget(self.slide_wind_n , 2, 2, 1, 1)


        # self.splt_chann_label = QLabel("Split Channels")
        # self.pre_processing_group.glayout.addWidget(self.splt_chann_label, 3, 6, 1, 1)
        self.split_chann_btn = QPushButton("Split Channels")
        self.split_chann_btn.setToolTip(("Split the current Image stack when using dual illumination."))
        self.pre_processing_group.glayout.addWidget(self.split_chann_btn, 2, 3, 1, 1)

        # self.glob_norm_data_btn = QPushButton("Normalize (global)")
        # self.pre_processing_group.glayout.addWidget(self.glob_norm_data_btn, 2, 2, 1, 1)
 
        self.compute_ratio_btn = QPushButton("compute ratio")
        self.compute_ratio_btn.setToolTip(("Compute Ratio of two images with identical dimensions. By default uses Ch0/Ch1 from the images selector"))
        self.pre_processing_group.glayout.addWidget(self.compute_ratio_btn, 2, 5, 1, 1)

        self.Ch0_ratio = QComboBox()
        self.pre_processing_group.glayout.addWidget(self.Ch0_ratio, 1, 4, 1, 1)
        self.Ch1_ratio = QComboBox()
        self.pre_processing_group.glayout.addWidget(self.Ch1_ratio, 2, 4, 1, 1)


        self.is_ratio_inverted = QCheckBox("Invert ratio")
        self.is_ratio_inverted.setToolTip(("By default Ch0/Ch1 is computed. Tick this checkbox and ratio will be computed inverted (Ch1/Ch0)"))
        self.is_ratio_inverted.setChecked(False)
        self.pre_processing_group.glayout.addWidget(self.is_ratio_inverted, 1, 5, 1, 1)
        ######## Filters group ########
        # QCollapsible creates a collapse container for inner widgets
       
        # magicgui doesn't need to be used as a decorator, you can call it
        # repeatedly to create new widgets:
        # new_widget = magicgui(do_something)
        # if you're mixing Qt and magicgui, you need to use the "native"
        # attribute in magicgui to access the QWidget





        self.filter_group = VHGroup('Filter Image', orientation='G')
        # self._pre_processing_layout.addWidget(self.filter_group.gbox)

        self._collapse_filter_group = QCollapsible('Filters', self)
        self._collapse_filter_group.addWidget(self.filter_group.gbox)


        ####### temporal filter subgroup #######     
        self.temp_filter_group = VHGroup('Temporal Filters', orientation='G')
        self.filter_group.glayout.addWidget(self.temp_filter_group.gbox, 1, 0)

        ######## temporal Filters btns ########
        self.temp_filt_type_label = QLabel("Filter type")
        self.temp_filter_group.glayout.addWidget(self.temp_filt_type_label, 0, 0, 1, 1)

        self.temp_filter_types = QComboBox()
        self.temp_filter_types.addItems(["Butterworth", "FIR"])
        self.temp_filter_group.glayout.addWidget(self.temp_filter_types, 0, 1, 1, 1)

        self.fps_label = QLabel("Sampling Freq (Hz)")
        self.temp_filter_group.glayout.addWidget(self.fps_label, 0, 2, 1, 1)
        
        self.fps_val = QLineEdit()
        self.fps_val.setText("")
        self.temp_filter_group.glayout.addWidget(self.fps_val, 0, 3, 1, 1)


        self.cutoff_freq_label = QLabel("Cutoff frequency")
        self.temp_filter_group.glayout.addWidget(self.cutoff_freq_label, 1, 0, 1, 1)

        self.butter_cutoff_freq_val = QLineEdit()
        self.butter_cutoff_freq_val.setValidator(QDoubleValidator()) 
        self.butter_cutoff_freq_val.setText("45")
        self.temp_filter_group.glayout.addWidget(self.butter_cutoff_freq_val, 1, 1, 1, 1)
        
        self.filt_order_label = QLabel("Filter order")
        self.temp_filter_group.glayout.addWidget(self.filt_order_label, 1, 2, 1, 1)

        self.butter_order_val = QSpinBox()
        self.butter_order_val.setSingleStep(1)
        self.butter_order_val.setValue(5)
        self.temp_filter_group.glayout.addWidget(self.butter_order_val, 1, 3, 1, 1)

        

        self.apply_temp_filt_btn = QPushButton("apply")
        self.temp_filter_group.glayout.addWidget(self.apply_temp_filt_btn, 2, 0, 1, 4)


        ####### spatial filter subgroup #######
        self.spac_filter_group = VHGroup('Spatial Filters', orientation='G')
        self.filter_group.glayout.addWidget(self.spac_filter_group.gbox, 0, 0)

        
        ######## spatial Filters btns ########
        # self.filters_label = QLabel("Gaussian filter")
        # self.filter_group.glayout.addWidget(self.filters_label, 3, 0, 1, 1)

        self.spatial_filt_type_label = QLabel("Filter type")
        self.spac_filter_group.glayout.addWidget(self.spatial_filt_type_label, 0, 0, 1, 1)
        
        self.spat_filter_types = QComboBox()
        self.spat_filter_types.addItems(["Gaussian", "Box", "Laplace", "Median", "Bilateral"])
        self.spac_filter_group.glayout.addWidget(self.spat_filter_types, 0, 1, 1, 1)

        self.sigma_label = QLabel("Sigma")
        self.spac_filter_group.glayout.addWidget(self.sigma_label, 0, 2, 1, 1)

        self.sigma_filt_spatial_value = QDoubleSpinBox()
        self.sigma_filt_spatial_value.setSingleStep(1)
        self.sigma_filt_spatial_value.setSingleStep(0.1)
        self.sigma_filt_spatial_value.setValue(1)
        self.spac_filter_group.glayout.addWidget(self.sigma_filt_spatial_value, 0, 3, 1, 1)

        self.kernels_label = QLabel("Kernel/window size")
        self.spac_filter_group.glayout.addWidget(self.kernels_label, 1, 2, 1, 1)

        self.filt_kernel_value = QSpinBox()
        self.filt_kernel_value.setSingleStep(1)
        self.filt_kernel_value.setSingleStep(1)
        self.filt_kernel_value.setValue(5)
        self.spac_filter_group.glayout.addWidget(self.filt_kernel_value, 1, 3, 1, 1)

        self.disk_label = QLabel("Sigma 2 (color)")
        self.disk_label.setToolTip(("Use this only in Bilateral filter"))
        self.spac_filter_group.glayout.addWidget(self.disk_label, 1, 0, 1, 1)

        self.sigma_filt_color_value = QDoubleSpinBox()
        self.sigma_filt_color_value.setSingleStep(1)
        self.sigma_filt_color_value.setSingleStep(0.1)
        self.sigma_filt_color_value.setValue(1)
        self.spac_filter_group.glayout.addWidget(self.sigma_filt_color_value, 1, 1, 1, 1)


        self.apply_spat_filt_btn = QPushButton("apply")
        self.apply_spat_filt_btn.setToolTip(("apply selected filter to the image"))
        self.spac_filter_group.glayout.addWidget(self.apply_spat_filt_btn, 2, 0, 1, 4)


        
        ######## Segmentation group ########
        
        self.segmentation_group = VHGroup('Segment shapes', orientation='G')

        self.segmentation_group = VHGroup('Segment Image', orientation='G')
        # self._pre_processing_layout.addWidget(self.filter_group.gbox)

        self._collapse_segmentation_group = QCollapsible('Segmentation', self)
        self._collapse_segmentation_group.addWidget(self.segmentation_group.gbox)
        
        self.auto_segmentation_group = VHGroup('Automatic segmentation', orientation='G')
        self.segmentation_group.glayout.addWidget(self.auto_segmentation_group.gbox, 0, 0, 1, 1)

        self.segmentation_methods_lable = QLabel("Method")
        self.auto_segmentation_group.glayout.addWidget(self.segmentation_methods_lable, 0, 0, 1, 1)

        self.segmentation_methods = QComboBox()
        self.segmentation_methods.addItems(["threshold_triangle", "GHT", "region_base"])
        self.auto_segmentation_group.glayout.addWidget(self.segmentation_methods, 0, 1, 1, 1)


        self.low_threshold_segmment_label = QLabel("low threshold")
        self.auto_segmentation_group.glayout.addWidget(self.low_threshold_segmment_label,  0, 2, 1, 1)
        
        self.low_threshold_segmment_value = QLineEdit()
        self.low_threshold_segmment_value.setValidator(QDoubleValidator()) 
        self.low_threshold_segmment_value.setText("0.05")
        self.auto_segmentation_group.glayout.addWidget(self.low_threshold_segmment_value,  0, 3, 1, 1)

        self.high_threshold_segmment_label = QLabel("high threshold")
        self.auto_segmentation_group.glayout.addWidget(self.high_threshold_segmment_label,  0, 4, 1, 1)
        
        self.high_threshold_segment_value = QLineEdit("High threshold")
        self.high_threshold_segment_value.setValidator(QDoubleValidator()) 
        self.high_threshold_segment_value.setText("0.2")
        self.auto_segmentation_group.glayout.addWidget(self.high_threshold_segment_value,  0, 5, 1, 1)

        self.is_Expand_mask = QCheckBox("Expand")
        self.is_Expand_mask.setChecked(False)
        self.auto_segmentation_group.glayout.addWidget(self.is_Expand_mask,  1, 0, 1, 1)

        self.n_pixels_expand = QComboBox()
        self.n_pixels_expand.addItems([str(i) for i in range(1, 21)])
        self.auto_segmentation_group.glayout.addWidget(self.n_pixels_expand, 1, 1, 1, 1)

        self.return_img_no_backg_btn = QCheckBox("Return image")
        self.return_img_no_backg_btn.setChecked(True)
        self.return_img_no_backg_btn.setToolTip(("Draw current selection as plot profile"))
        # self._plottingWidget_layout.addWidget(self.plot_profile_btn)
        self.auto_segmentation_group.glayout.addWidget(self.return_img_no_backg_btn, 1, 2, 1, 1)

        self.is_inverted_mask = QCheckBox("Inverted mask")
        self.is_Expand_mask.setChecked(False)
        self.auto_segmentation_group.glayout.addWidget(self.is_inverted_mask,  1, 3, 1, 1)

        self.apply_auto_segmentation_btn = QPushButton("segment stack (Auto)")
        self.auto_segmentation_group.glayout.addWidget(self.apply_auto_segmentation_btn, 1, 4, 1, 2)

        self.manual_segmentation_group = VHGroup('Manual segmentation', orientation='G')
        self.segmentation_group.glayout.addWidget(self.manual_segmentation_group.gbox,  1, 0, 1, 1)

        # self.img_list_manual_segment = QComboBox()
        # self.img_list_manual_segment_label = QLabel("Image")
        # self.manual_segmentation_group.glayout.addWidget(self.img_list_manual_segment_label, 0, 0, 1, 1)
        # self.manual_segmentation_group.glayout.addWidget(self.img_list_manual_segment, 0, 1, 1, 1)

        self.mask_list_manual_segment = QComboBox()
        self.mask_list_manual_segment_label = QLabel("Mask")
        self.manual_segmentation_group.glayout.addWidget(self.mask_list_manual_segment_label, 0, 2, 1, 1)
        self.manual_segmentation_group.glayout.addWidget(self.mask_list_manual_segment, 0, 3, 1, 1)

        self.apply_manual_segmentation_btn = QPushButton("segment stack (Manual)")
        self.manual_segmentation_group.glayout.addWidget(self.apply_manual_segmentation_btn, 0, 4, 1, 1)





        ######## Load spool data btns Group ########
        self.load_spool_group = VHGroup('Load Spool data', orientation='G')

        self.dir_btn_label = QLabel("Directory name")
        self.load_spool_group.glayout.addWidget(self.dir_btn_label, 3, 1, 1, 1)

        # self.dir_box_text = FileDropLineEdit()
        self.dir_box_text = QLineEdit()
        self.dir_box_text.installEventFilter(self)
        # self.dir_box_text.setDragEnabled(True)
        self.dir_box_text.setAcceptDrops(True)
        # self.dir_box_text.set
        self.dir_box_text.setPlaceholderText(os.getcwd())
        self.load_spool_group.glayout.addWidget(self.dir_box_text, 3, 2, 1, 1)

        self.load_spool_dir_btn = QPushButton("Load folder")
        self.load_spool_group.glayout.addWidget(self.load_spool_dir_btn, 3, 3, 1, 1)

        self.search_spool_dir_btn = QPushButton("Search Directory")
        self.load_spool_group.glayout.addWidget(self.search_spool_dir_btn, 3, 4, 1, 1)

        # self.fast_loading = QCheckBox("Multithreading")
        # self.load_spool_group.glayout.addWidget(self.fast_loading, 3, 5, 1, 1)


        ######## Segmentation group ########
        # self._pre_processing_layout.addWidget(self.segmentation_group.gbox)

        ######## Segmentation btns ########
        # NOTE: on 26.03.2024 this is zomby code, soon to be removed


        # self.seg_heart_label = QLabel("Segment the heart shape")
        # self.auto_segmentation_group.glayout.addWidget(self.seg_heart_label, 3, 0, 1, 1)
        # self.seg_heart_btn = QPushButton("apply")
        # self.auto_segmentation_group.glayout.addWidget(self.seg_heart_btn, 3, 1, 1, 1)

        # self.sub_bkg_label = QLabel("Subtract Background")
        # self.auto_segmentation_group.glayout.addWidget(self.sub_bkg_label, 4, 0, 1, 1)
        # self.sub_backg_btn = QPushButton("apply")
        # self.auto_segmentation_group.glayout.addWidget(self.sub_backg_btn, 4, 1, 1, 1)

        # self.del_bkg_label = QLabel("Delete Background")
        # self.auto_segmentation_group.glayout.addWidget(self.del_bkg_label, 5, 0, 1, 1)
        # self.rmv_backg_btn = QPushButton("apply")
        # self.auto_segmentation_group.glayout.addWidget(self.rmv_backg_btn, 5, 1, 1, 1)

        # self.pick_frames_btn = QLabel("Pick frames")
        # self.auto_segmentation_group.glayout.addWidget(self.pick_frames_btn, 6, 0, 1, 1)
        # self.pick_frames_btn = QPushButton("apply")
        # self.auto_segmentation_group.glayout.addWidget(self.pick_frames_btn, 6, 1, 1, 1)




         ######## Plotting Group ########

        self.plotting_tabs = QTabWidget()
        self._plotting_profile_tabs_layout = VHGroup('Plot profile', orientation='G')
        # self.plotting_tabs.setLayout(self._plotting_tabs_layout)
        
        # self._plotting_tabs_layout.setAlignment(Qt.AlignTop)
        
        
        

        # self.plot_group = VHGroup('Plot profile', orientation='G')
        # self.main_layout.addWidget(self.plot_grpup.gbox)

        ############################################
        ############ create plot widget ############
        ############################################

        self.main_plot_widget =  BaseNapariMPLWidget(self.viewer) # this is the cleanest widget thatz does not have any callback on napari
        # self.plot_group.glayout.addWidget(self.plot_widget, 0, 1, 1, 1)
        # self.plot_group.glayout.addWidget(self.plot_widget, 0, 1, 1, 1)
        self._plotting_profile_tabs_layout.glayout.addWidget(self.main_plot_widget, 0, 0, 1, 3)


        ######################################################
        ############ create plotting button widget ###########
        ######################################################

        # self.plot_profile_btn = QPushButton("Plot profile")
        self.plot_profile_btn = QCheckBox("Display profile")
        self.plot_profile_btn.setToolTip(("Draw current selection as plot profile"))
        # self._plottingWidget_layout.addWidget(self.plot_profile_btn)
        # self.plot_group.glayout.addWidget(self.plot_profile_btn, 1, 1, 1, 1)
        self._plotting_profile_tabs_layout.glayout.addWidget(self.plot_profile_btn, 1, 0, 1, 1)

        self.x_scale_box_label = QLabel("set x scale")
        self._plotting_profile_tabs_layout.glayout.addWidget(self.x_scale_box_label, 1, 1, 1, 1)
        self.x_scale_box =  QLineEdit()
        self.x_scale_box.setValidator(QDoubleValidator()) 
        self.x_scale_box.setFixedWidth(50)
        self.x_scale_box.setText(f"{1}")
        self._plotting_profile_tabs_layout.glayout.addWidget(self.x_scale_box, 1, 2, 1, 1)
        
        self.clip_trace_btn = QPushButton("Clip Trace")
        self._plotting_profile_tabs_layout.glayout.addWidget(self.clip_trace_btn, 2, 0, 1, 1)
        
        self.is_range_clicked_checkbox = QCheckBox("Show range")
        self._plotting_profile_tabs_layout.glayout.addWidget(self.is_range_clicked_checkbox, 2, 1, 1, 1)

        self.double_slider_clip_trace = QLabeledDoubleRangeSlider(Qt.Orientation.Horizontal)
        self._plotting_profile_tabs_layout.glayout.addWidget(self.double_slider_clip_trace, 2, 2, 1, 1)


        self.plotting_tabs.addTab(self._plotting_profile_tabs_layout.gbox, 'Plot profile')


        self._plotting_hisotgram_tabs_layout = VHGroup('Image histogram', orientation='G')
        self.histogram_plot_widget =  BaseNapariMPLWidget(self.viewer)
        self._plotting_hisotgram_tabs_layout.glayout.addWidget(self.histogram_plot_widget, 0, 0, 1, 7)
        
        self.hist_currf_label = QLabel("current frame")
        self._plotting_hisotgram_tabs_layout.glayout.addWidget(self.hist_currf_label, 1, 0, 1, 1)
        
        self.toggle_hist_data = ToggleButton()
        self._plotting_hisotgram_tabs_layout.glayout.addWidget(self.toggle_hist_data, 1, 1, 1, 1)

        self.hist_currf_label = QLabel("all stack")
        self._plotting_hisotgram_tabs_layout.glayout.addWidget(self.hist_currf_label, 1, 2, 1, 1)

        self.hist_bins_label = QLabel("Bins")
        self._plotting_hisotgram_tabs_layout.glayout.addWidget(self.hist_bins_label, 1, 3, 1, 1)

        self.slider_histogram_bins = QLabeledSlider(Qt.Orientation.Horizontal)
        self.slider_histogram_bins.setRange(5, 256)
        self.slider_histogram_bins.setValue(100)
        self._plotting_hisotgram_tabs_layout.glayout.addWidget(self.slider_histogram_bins, 1, 4, 1, 1)

        self.plot_histogram_btn = QPushButton("Plot Histogram")
        self._plotting_hisotgram_tabs_layout.glayout.addWidget(self.plot_histogram_btn, 1, 5, 1, 1)

        self.clear_histogram_btn = QPushButton("Clear Plot")
        self._plotting_hisotgram_tabs_layout.glayout.addWidget(self.clear_histogram_btn, 1, 6, 1, 1)


        self.plotting_tabs.addTab(self._plotting_hisotgram_tabs_layout.gbox, 'Histogram')


        ######################################################
        ############ create Shape selector widget ############
        ######################################################
        
        self.selector_group = VHGroup('Image/Shapes selector', orientation='G')        
        self.main_layout.addWidget(self.selector_group.gbox)
        
        self.listShapeswidget = QListWidget()
        self.listShapeswidget_label = QLabel("Select *Shape* for plotting profile")
        self.selector_group.glayout.addWidget(self.listShapeswidget_label, 0, 2, 1, 1)
        self.selector_group.glayout.addWidget(self.listShapeswidget, 1, 2, 1, 1)

        ######################################################
        ############ create Image selector widget ############
        ######################################################
        
        self.listImagewidget = QListWidget()
        self.listImagewidget.setSelectionMode(
            QtWidgets.QAbstractItemView.ExtendedSelection
        )

        self.listImagewidget_label = QLabel("Select *Image* for plotting profile")
        self.selector_group.glayout.addWidget(self.listImagewidget_label, 0, 1, 1, 1)
        self.selector_group.glayout.addWidget(self.listImagewidget, 1, 1, 1, 1)

        ########################################################
        # set the layout of the plotting group in the given order
        ########################################################
        self._pre_processing_layout.addWidget(self.load_spool_group.gbox)
        self._pre_processing_layout.addWidget(self.pre_processing_group.gbox)
        self._pre_processing_layout.addWidget(self._collapse_filter_group)
        self._pre_processing_layout.addWidget(self._collapse_segmentation_group)
        # self._pre_processing_layout.addWidget(self.plot_group.gbox)
        self._pre_processing_layout.addWidget(self.plotting_tabs)

        ######## Shapes tab ########
        ############################

        self._layers_processing_layout.setAlignment(Qt.AlignTop)
        
        ######## Rois handeling group ########
        self.copy_rois_group = VHGroup('Copy ROIs from one layer to another', orientation='G')
        self._layers_processing_layout.addWidget(self.copy_rois_group.gbox, 0, 0, 1, 1)
        
        self.ROI_selection_1 = QComboBox()
        self.ROI_1_label = QLabel("From layer")
        self.copy_rois_group.glayout.addWidget(self.ROI_1_label, 1, 0, 1, 1)
        # self.ROI_selection_1.setAccessibleName("From layer")
        # self.ROI_selection_1.addItems(self.get_rois_list())
        self.copy_rois_group.glayout.addWidget(self.ROI_selection_1, 1, 1, 1, 1)
        
        self.ROI_selection_2 = QComboBox()
        self.ROI_2_label = QLabel("To layer")
        self.copy_rois_group.glayout.addWidget(self.ROI_2_label, 2, 0, 1, 1)
        # self.ROI_selection_2.setAccessibleName("To layer")
        # self.ROI_selection_2.addItems(self.get_rois_list())
        self.copy_rois_group.glayout.addWidget(self.ROI_selection_2, 2, 1, 1, 1)

        self.copy_ROIs_btn = QPushButton("Transfer ROIs")
        self.copy_ROIs_btn.setToolTip(("Transfer ROIs from one 'Shape' layer to another 'Shape' layer"))
        self.copy_rois_group.glayout.addWidget(self.copy_ROIs_btn, 3, 0, 1, 2)



        self.crop_from_shape_group = VHGroup('Crop from shape', orientation='G')
        self._layers_processing_layout.addWidget(self.crop_from_shape_group.gbox, 0, 1, 1, 1)
        
        self.shape_crop_label = QLabel("Shape")
        self.crop_from_shape_group.glayout.addWidget(self.shape_crop_label, 1, 0, 1, 1)
        
        self.ROI_selection_crop = QComboBox()
        self.crop_from_shape_group.glayout.addWidget(self.ROI_selection_crop, 1, 1, 1, 1)
        
        self.rotate_l_crop = QCheckBox("Crop + Rotate (L)")
        self.crop_from_shape_group.glayout.addWidget(self.rotate_l_crop, 1, 2, 1, 1)

        self.image_crop_label = QLabel("Image")
        self.crop_from_shape_group.glayout.addWidget(self.image_crop_label, 2, 0, 1, 1)
        
        self.image_selection_crop = QComboBox()
        self.crop_from_shape_group.glayout.addWidget(self.image_selection_crop, 2, 1, 1, 1)
        
        self.rotate_r_crop = QCheckBox("Crop + Rotate (R)")
        self.crop_from_shape_group.glayout.addWidget(self.rotate_r_crop, 2, 2, 1, 1)

        self.crop_from_shape_btn = QPushButton("Crop")
        self.crop_from_shape_group.glayout.addWidget(self.crop_from_shape_btn, 3, 0, 1, 3)



        self.crop_all_views_and_rotate_group = VHGroup('Crop views and rearrange', orientation='G')
        self._layers_processing_layout.addWidget(self.crop_all_views_and_rotate_group.gbox, 1, 0, 1, 2)

        self.pad_h_pixels_label = QLabel("Pad Hor")
        self.crop_all_views_and_rotate_group.glayout.addWidget(self.pad_h_pixels_label, 0, 0, 1, 1)
        # self.c_kernels_label.setToolTip((""))
        self.pad_h_pixels = QLabeledSlider(Qt.Orientation.Horizontal)
        self.pad_h_pixels.setRange(0, 100)
        self.pad_h_pixels.setValue(10)
        self.crop_all_views_and_rotate_group.glayout.addWidget(self.pad_h_pixels, 0, 1, 1, 1)

        self.pad_v_pixels_label = QLabel("Pad Ver")
        self.crop_all_views_and_rotate_group.glayout.addWidget(self.pad_v_pixels_label, 0, 2, 1, 1)
        # self.c_kernels_label.setToolTip((""))
        self.pad_v_pixels = QLabeledSlider(Qt.Orientation.Horizontal)
        self.pad_v_pixels.setRange(0, 100)
        self.pad_v_pixels.setValue(10)
        self.crop_all_views_and_rotate_group.glayout.addWidget(self.pad_v_pixels, 0, 3, 1, 1)

        self.pad_value_label = QLabel("Pad with:")
        self.crop_all_views_and_rotate_group.glayout.addWidget(self.pad_value_label, 1, 0, 1, 1)
        
        self.pad_value = QComboBox()
        self.pad_value.addItems(["background", "0", "NaN"])
        self.crop_all_views_and_rotate_group.glayout.addWidget(self.pad_value, 1, 1, 1, 1)
        
        self.crop_view_orientation_label = QLabel("Orientation:")
        self.crop_all_views_and_rotate_group.glayout.addWidget(self.crop_view_orientation_label, 1, 2, 1, 1)
        
        self.crop_view_orientation = QComboBox()
        self.crop_view_orientation.addItems(["horizontal", "vertical"])
        self.crop_all_views_and_rotate_group.glayout.addWidget(self.crop_view_orientation, 1, 3, 1, 1)

        self.view0_rotate_label = QLabel("View 0")
        self.crop_all_views_and_rotate_group.glayout.addWidget(self.view0_rotate_label, 0, 4, 1, 1)
        
        self.view0_rotate = QComboBox()
        self.view0_rotate.addItems(["R", "L"])
        self.view0_rotate.setCurrentText("L")
        self.crop_all_views_and_rotate_group.glayout.addWidget(self.view0_rotate, 0, 5, 1, 1)
        
        self.view1_rotate_label = QLabel("View 1")
        self.crop_all_views_and_rotate_group.glayout.addWidget(self.view1_rotate_label, 0, 6, 1, 1)
        
        self.view1_rotate = QComboBox()
        self.view1_rotate.addItems(["R", "L"])
        self.view1_rotate.setCurrentText("L")
        self.crop_all_views_and_rotate_group.glayout.addWidget(self.view1_rotate, 0, 7, 1, 1)
        
        self.view2_rotate_label = QLabel("View 2")
        self.crop_all_views_and_rotate_group.glayout.addWidget(self.view2_rotate_label, 1, 4, 1, 1)
        
        self.view2_rotate = QComboBox()
        self.view2_rotate.addItems(["R", "L"])
        self.view2_rotate.setCurrentText("L")
        self.crop_all_views_and_rotate_group.glayout.addWidget(self.view2_rotate, 1, 5, 1, 1)
        
        self.view3_rotate_label = QLabel("View 3")
        self.crop_all_views_and_rotate_group.glayout.addWidget(self.view3_rotate_label, 1, 6, 1, 1)
        
        self.view3_rotate = QComboBox()
        self.view3_rotate.addItems(["R", "L"])
        self.crop_all_views_and_rotate_group.glayout.addWidget(self.view3_rotate, 1, 7, 1, 1)
        self.view_rotates = [self.view0_rotate, 
                            self.view1_rotate, 
                            self.view2_rotate, 
                            self.view3_rotate
                            ]
        
        self.return_bounding_boxes_only_btn = QPushButton("Return only bounding boxes")
        self.crop_all_views_and_rotate_group.glayout.addWidget(self.return_bounding_boxes_only_btn, 3, 2, 1, 2)

        self.crop_all_views_and_rotate_btn = QPushButton("Rearrange views")
        self.crop_all_views_and_rotate_group.glayout.addWidget(self.crop_all_views_and_rotate_btn, 3, 4, 1, 4)

        self.crop_all_views_and_rotate_form_box_btn = QPushButton("Rearrange from boxes")
        self.crop_all_views_and_rotate_group.glayout.addWidget(self.crop_all_views_and_rotate_form_box_btn, 3, 0, 1, 2)

        self.return_mask_form_rearranging = QCheckBox("Return Mask")
        self.crop_all_views_and_rotate_group.glayout.addWidget(self.return_mask_form_rearranging, 2, 6, 1, 2)

        self.join_all_views_and_rotate_group = VHGroup('Join cropped/individual views', orientation='G')
        self._layers_processing_layout.addWidget(self.join_all_views_and_rotate_group.gbox, 2, 0, 1, 2)

        self.join_imgs_selector = QListWidget()
        self.join_imgs_selector.setSelectionMode(
            QtWidgets.QAbstractItemView.ExtendedSelection
        )
        self.join_all_views_and_rotate_group.glayout.addWidget(self.join_imgs_selector, 0, 0, 1, 1)

        self.join_all_views_and_rotate_btn = QPushButton("Join images")
        self.join_all_views_and_rotate_group.glayout.addWidget(self.join_all_views_and_rotate_btn, 1, 0, 1, 1)



        
        ######## Mot-Correction tab ########
        ####################################

        self._motion_correction_layout.setAlignment(Qt.AlignTop)

        ######## Transform group ########
        self.transform_group = VHGroup('Transform Image data', orientation='G') # Zombi code
        # self._motion_correction_layout.addWidget(self.transform_group.gbox) # Zombi code

        ######## Transform btns ########
        # self.inv_img_label = QLabel("Transform data to integer")
        # self.transform_group.glayout.addWidget(self.inv_img_label, 3, 0, 1, 1)
        # self.transform_to_uint16_btn = QPushButton("Apply")
        # self.transform_to_uint16_btn.setToolTip(("Transform numpy array data to type integer np.uint16"))
        # self.transform_group.glayout.addWidget(self.transform_to_uint16_btn, 3, 1, 1, 1)

        ######## Mot-Correction group ########

         ####### NOTE: deprecating this method on 14.08.22024 #######
        # self.mot_correction_group = VHGroup('Apply image registration (motion correction)', orientation='G')
        # self._motion_correction_layout.addWidget(self.mot_correction_group.gbox)


        # self.fottprint_size_label = QLabel("Foot print size")
        # self.fottprint_size_label.setToolTip(("Footprint size for local normalization"))
        # self.mot_correction_group.glayout.addWidget(self.fottprint_size_label, 3, 0, 1, 1)

        # self.use_GPU_label = QLabel("Use GPU")
        # self.mot_correction_group.glayout.addWidget(self.use_GPU_label, 3, 2, 1, 1)
        
        # self.use_GPU = QCheckBox()
        # try:
        #     subprocess.check_output('nvidia-smi')
        #     warn('Nvidia GPU detected!, setting to default GPU use.\nSet GPU use as default')
        #     self.use_GPU.setChecked(True)
        # except Exception: # this command not being found can raise quite a few different errors depending on the configuration
        #     warn('No Nvidia GPU in system!, setting to default CPU use')
        #     self.use_GPU.setChecked(False)
        
        # self.mot_correction_group.glayout.addWidget(self.use_GPU,  3, 3, 1, 1)



        
        # self.footprint_size = QSpinBox()
        # self.footprint_size.setSingleStep(1)
        # self.footprint_size.setValue(10)
        # self.mot_correction_group.glayout.addWidget(self.footprint_size, 3, 1, 1, 1)

        # self.radius_size_label = QLabel("Radius size")
        # self.radius_size_label.setToolTip(("Radius of the window considered around each pixel for image registration"))
        # self.mot_correction_group.glayout.addWidget(self.radius_size_label, 4, 0, 1, 1)
        
        # self.radius_size = QSpinBox()
        # self.radius_size.setSingleStep(1)
        # self.radius_size.setValue(7)
        # self.mot_correction_group.glayout.addWidget(self.radius_size, 4, 1, 1, 1)

        # self.n_warps_label = QLabel("Number of warps")
        # self.mot_correction_group.glayout.addWidget(self.n_warps_label, 5, 0, 1, 1)
        
        # self.n_warps = QSpinBox()
        # self.n_warps.setSingleStep(1)
        # self.n_warps.setValue(8)
        # self.mot_correction_group.glayout.addWidget(self.n_warps, 5, 1, 1, 1)


        # self.apply_mot_correct_btn = QPushButton("apply")
        # self.apply_mot_correct_btn.setToolTip(("apply registration method to correct the image for motion artefacts"))
        # self.mot_correction_group.glayout.addWidget(self.apply_mot_correct_btn, 6, 0, 1, 1)


        
        
        
        
        # add new group for motion compensation using optimap
        self.mot_correction_group_optimap = VHGroup('Apply image registration (optimap)', orientation='G')
        self._motion_correction_layout.addWidget(self.mot_correction_group_optimap.gbox)

        self.c_kernels_label = QLabel("Contrast Kernel")
        # self.c_kernels_label.setToolTip((""))
        self.mot_correction_group_optimap.glayout.addWidget(self.c_kernels_label, 1, 0, 1, 1)
                
        self.c_kernels = QLabeledSlider(Qt.Orientation.Horizontal)
        self.c_kernels.setRange(3, 30)
        self.c_kernels.setValue(5)
        self.mot_correction_group_optimap.glayout.addWidget(self.c_kernels, 1, 1, 1, 1)

        self.pre_smooth_temp_label = QLabel("Pre-smooth Temp")
        # self.c_kernels_label.setToolTip((""))
        self.mot_correction_group_optimap.glayout.addWidget(self.pre_smooth_temp_label, 1, 2, 1, 1)
        
        self.pre_smooth_temp = QLabeledSlider(Qt.Orientation.Horizontal)
        self.pre_smooth_temp.setRange(0, 10)
        self.pre_smooth_temp.setValue(1)
        self.mot_correction_group_optimap.glayout.addWidget(self.pre_smooth_temp, 1, 3, 1, 1)

        self.pre_smooth_spat_label = QLabel("Pre-smooth Spat")
        # self.c_kernels_label.setToolTip((""))
        self.mot_correction_group_optimap.glayout.addWidget(self.pre_smooth_spat_label, 1, 4, 1, 1)

        self.pre_smooth_spat = QLabeledSlider(Qt.Orientation.Horizontal)
        self.pre_smooth_spat.setRange(0, 10)
        self.pre_smooth_spat.setValue(1)
        self.mot_correction_group_optimap.glayout.addWidget(self.pre_smooth_spat, 1, 5, 1, 1)

        # self.ref_frame_label = QLabel("Ref Frame")
        # # self.c_kernels_label.setToolTip((""))
        # self.mot_correction_group_optimap.glayout.addWidget(self.ref_frame_label, 2, 0, 1, 1)
        
        self.ref_frame_label = QLabel("Reference frame")
        self.mot_correction_group_optimap.glayout.addWidget(self.ref_frame_label, 2, 0, 1, 1)
        
        self.ref_frame_val = QLineEdit()
        self.ref_frame_val.setValidator(QIntValidator()) 
        self.ref_frame_val.setText("0")
        self.mot_correction_group_optimap.glayout.addWidget(self.ref_frame_val, 2, 1, 1, 1)

        self.apply_optimap_mot_corr_btn = QPushButton("apply")
        self.mot_correction_group_optimap.glayout.addWidget(self.apply_optimap_mot_corr_btn, 2, 4, 1, 2)


        ######## APD-analysis tab ########
        # ####################################
        self._APD_analysis_layout.setAlignment(Qt.AlignTop)

        ##### APD_plot_group ########
        self.APD_plot_group = VHGroup('APD plot group', orientation='G')
        

        self._APD_plot_widget = BaseNapariMPLWidget(self.viewer) 
        self.APD_plot_group.glayout.addWidget(self._APD_plot_widget, 6, 0, 1, 4)
        
        # self._APD_TSP = NapariMPLWidget(self.viewer)
        # self.APD_plot_group.glayout.addWidget(self._APD_TSP, 3, 0, 1, 8)
        # self.APD_axes = self._APD_TSP.canvas.figure.subplots()

        self.compute_APD_btn = QPushButton("Compute APDs")
        self.compute_APD_btn.setToolTip(("Plot APDs detected in the selected Image in the selector"))
        self.APD_plot_group.glayout.addWidget(self.compute_APD_btn, 3, 0, 1, 1)

        self.clear_plot_APD_btn = QPushButton("Clear traces")
        self.clear_plot_APD_btn.setToolTip(("PLot the current traces displayed in main plotter"))
        self.APD_plot_group.glayout.addWidget(self.clear_plot_APD_btn, 3, 1, 1, 1)

        self.APD_computing_method_label = QLabel("AP detect meth")
        self.APD_computing_method_label.setToolTip(("""        
        Select the method to compute the resting (membrane) to detect the AP. 
         Methods are : 
        - bcl_to_bcl: from BCL (Basal cycle length) to BCL.
        - pre_upstroke_min: minimum value Pre-upstroke, 
        - post_AP_min: minimum value after AP,
        - ave_pre_post_min: average the minimum value before and after stroke.
         """))
        self.APD_plot_group.glayout.addWidget(self.APD_computing_method_label, 3, 2, 1, 1)
        
        self.APD_computing_method = QComboBox()
        self.APD_computing_method.addItems(["bcl_to_bcl", "pre_upstroke_min", "post_AP_min", "ave_pre_post_min"])
        self.APD_plot_group.glayout.addWidget(self.APD_computing_method, 3, 3, 1, 1)
        
        self.slider_APD_detection_threshold = QSlider(Qt.Orientation.Horizontal)
        self.slider_APD_thres_max_range = 10000
        self.slider_APD_detection_threshold.setRange(1, 1000)
        self.slider_APD_detection_threshold.setValue(500)
        self.prominence = self.slider_APD_detection_threshold.value() / (self.slider_APD_thres_max_range)
        self.APD_plot_group.glayout.addWidget(self.slider_APD_detection_threshold, 4, 1, 1, 1)
        
        self.slider_label_current_value = QLabel(f"Sensitivity threshold: {self.slider_APD_detection_threshold.value() / (self.slider_APD_thres_max_range )}")
        self.slider_label_current_value.setToolTip('Change the threshold sensitivity for the APD detection base on peak "prominence"')
        self.APD_plot_group.glayout.addWidget(self.slider_label_current_value, 4, 0, 1, 1)
        
        self.APD_peaks_help_box_label_def_value = 0
        self.APD_peaks_help_box_label = QLabel(f"[AP detected]: {self.APD_peaks_help_box_label_def_value}")
        self.APD_peaks_help_box_label.setToolTip('Display number of peaks detected as you scrol over the "Sensitivity threshold')
        self.APD_plot_group.glayout.addWidget(self.APD_peaks_help_box_label, 5, 0, 1, 4)
        
        self.slider_APD_percentage = QLabeledSlider(Qt.Orientation.Horizontal)
        self.slider_APD_percentage.setRange(10, 100)
        self.slider_APD_percentage.setValue(75)
        self.slider_APD_percentage.setSingleStep(5)
        self.APD_plot_group.glayout.addWidget(self.slider_APD_percentage, 4, 3, 1, 1)
        
        self.slider_APD_perc_label = QLabel(f"APD %")
        self.slider_APD_perc_label.setToolTip('Change the APD at the given percentage')
        self.APD_plot_group.glayout.addWidget(self.slider_APD_perc_label, 4, 2, 1, 1)
        values = []
        self.AP_df_default_val = pd.DataFrame({"image_name": values,
                                               "ROI_id" : values, 
                                               "APD_perc" : values,
                                               "APD_perc" : values, 
                                               "APD" : values, 
                                               "AcTime_dVdtmax": values, 
                                               "BasCycLength_bcl": values})

        
        model = PandasModel(self.AP_df_default_val)
        self.APD_propert_table = QTableView()
        self.APD_propert_table.setModel(model)      

        self.APD_propert_table.horizontalHeader().setStretchLastSection(True)
        self.APD_propert_table.setAlternatingRowColors(False)
        self.APD_propert_table.setSelectionBehavior(QTableView.SelectRows)
        self.APD_plot_group.glayout.addWidget(self.APD_propert_table, 7, 0, 1, 4)


         ##### APD export results ########
        self.APD_export_group = VHGroup('Export results', orientation='G')


        

        

        self.APD_rslt_dir_btn_label = QLabel("Save to Directory")
        self.APD_rslt_dir_btn_label.setToolTip("Drag and drop folders here to change the current directory to save your APD results")
        self.APD_export_group.glayout.addWidget(self.APD_rslt_dir_btn_label, 1, 0, 1, 1)

        self.APD_rslts_dir_box_text = QLineEdit()
        self.APD_rslts_dir_box_text.installEventFilter(self)
        self.APD_rslts_dir_box_text.setAcceptDrops(True)
        self.APD_rslts_dir_box_text.setPlaceholderText(os.getcwd())
        self.APD_rslts_dir_box_text.setToolTip(("Drag and drop or copy/paste a directory path to export your results"))
        self.APD_export_group.glayout.addWidget(self.APD_rslts_dir_box_text, 1, 1, 1, 1)

        self.search_dir_APD_rslts_btn = QPushButton("change directory")
        self.search_dir_APD_rslts_btn.setToolTip(("Navigate to change the current directory to save your APD results"))
        self.APD_export_group.glayout.addWidget(self.search_dir_APD_rslts_btn, 1, 2, 1, 2)        
        
        self.label_rstl_name = QLabel("Rename results")
        self.label_rstl_name.setToolTip(("Set the name for the resulting table"))
        self.APD_export_group.glayout.addWidget(self.label_rstl_name, 2, 0,  1, 1)
        
        self.table_rstl_name = QLineEdit()
        self.table_rstl_name.setPlaceholderText("APD_results")
        self.APD_export_group.glayout.addWidget(self.table_rstl_name, 2, 1, 1, 1)

        self.APD_rslts_export_file_format_label = QLabel("Change format")
        self.APD_export_group.glayout.addWidget(self.APD_rslts_export_file_format_label, 2, 2, 1, 1)
        
        self.APD_rslts_export_file_format = QComboBox()
        self.APD_rslts_export_file_format.addItems([".csv", ".xlsx"])
        self.APD_export_group.glayout.addWidget(self.APD_rslts_export_file_format, 2, 3, 1, 1)

        self.copy_APD_rslts_btn = QPushButton("Copy to clipboard")
        self.copy_APD_rslts_btn.setToolTip(("Copy to clipboard the current APD results."))
        self.APD_export_group.glayout.addWidget(self.copy_APD_rslts_btn, 4, 0, 1, 2)

        self.save_APD_rslts_btn = QPushButton("Export table")
        self.save_APD_rslts_btn.setToolTip(("Export current APD results to a directory in .csv format."))
        self.APD_export_group.glayout.addWidget(self.save_APD_rslts_btn, 4, 2, 1, 2)


        ######## Mapping tab ########
        # ####################################
        self.mapping_tabs = QTabWidget()
        self._mapping_processing_layout.setAlignment(Qt.AlignTop)

        ##### APD_plot_group ########
        self.average_trace_group = VHGroup('Average individual AP traces', orientation='G')
        self._mapping_processing_layout.addWidget(self.average_trace_group.gbox)

        self.preview_AP_splitted_btn = QPushButton("Preview traces")
        self.preview_AP_splitted_btn.setToolTip(("Preview individual overlaper AP detected from the current trace"))
        self.average_trace_group.glayout.addWidget(self.preview_AP_splitted_btn, 1, 0, 1, 1)

        self.create_average_AP_btn = QPushButton("Average traces")
        self.create_average_AP_btn.setToolTip(("Create a single AP by averaging the from the individula APs displayed"))
        self.average_trace_group.glayout.addWidget(self.create_average_AP_btn, 1, 1, 1, 1)

        self.clear_AP_splitted_btn = QPushButton("Clear Plot")
        self.clear_AP_splitted_btn.setToolTip(("Clear the current trace"))
        self.average_trace_group.glayout.addWidget(self.clear_AP_splitted_btn, 1, 2, 1, 1)

        self.create_AP_gradient_btn = QPushButton("Check Activation times")
        self.average_trace_group.glayout.addWidget(self.create_AP_gradient_btn, 1, 3, 1, 2)

        self.slider_label_current_value_2 = QLabel(self.slider_label_current_value.text())
        self.slider_label_current_value_2.setToolTip('Change the threshold sensitivity for the APD detection base on peak "prominence"')
        self.average_trace_group.glayout.addWidget(self.slider_label_current_value_2, 2, 0, 1, 1)
        
        self.slider_APD_detection_threshold_2 = QSlider(Qt.Orientation.Horizontal)
        self.slider_APD_thres_max_range = 10000
        self.slider_APD_detection_threshold_2.setRange(1, 1000)
        self.slider_APD_detection_threshold_2.setValue(500)
        self.average_trace_group.glayout.addWidget(self.slider_APD_detection_threshold_2, 2, 1, 1, 1)

        self.APD_peaks_help_box_label_2 = QLabel(f"[AP detected]: {self.APD_peaks_help_box_label_def_value}")
        self.APD_peaks_help_box_label_2.setToolTip('Display number of peaks detected as you scrol over the "Sensitivity threshold')
        self.average_trace_group.glayout.addWidget(self.APD_peaks_help_box_label_2, 2, 2, 1, 1)

        self.remove_mean_label = QLabel("Display mean")
        self.average_trace_group.glayout.addWidget(self.remove_mean_label, 2, 3, 1, 1)

        self.remove_mean_check = QCheckBox()
        self.remove_mean_check.setChecked(False)
        self.average_trace_group.glayout.addWidget(self.remove_mean_check, 2, 4, 1, 1)

        self.slider_N_APs_label = QLabel("Slide to select your current AP")
        self.average_trace_group.glayout.addWidget(self.slider_N_APs_label, 3, 0, 1, 1)

        self.slider_N_APs = QLabeledSlider(Qt.Orientation.Horizontal)
        self.slider_N_APs.setValue(0)
        self.average_trace_group.glayout.addWidget(self.slider_N_APs, 3, 1, 1, 1)

        self.shift_AP_label = QLabel("Shift selected AP")
        self.average_trace_group.glayout.addWidget(self.shift_AP_label, 3, 2, 1, 1)

        self.mv_left_AP_btn = QToolButton()
        self.shif_trace = False
        self.mv_left_AP_btn.setArrowType(QtCore.Qt.LeftArrow) # type: ignore
        self.average_trace_group.glayout.addWidget(self.mv_left_AP_btn, 3, 3, 1, 1)
        
        self.mv_righ_AP_btn = QToolButton()
        self.mv_righ_AP_btn.setArrowType(QtCore.Qt.RightArrow)
        self.average_trace_group.glayout.addWidget(self.mv_righ_AP_btn, 3, 4, 1, 1)

        self.average_AP_plot_widget =  BaseNapariMPLWidget(self.viewer) # this is the cleanest widget thatz does not have any callback on napari
        self.average_trace_group.glayout.addWidget(self.average_AP_plot_widget, 4, 0, 1, 5)
        
        self.activation_map_label = QLabel("Activation Maps")
        self.average_trace_group.glayout.addWidget(self.activation_map_label, 5, 0, 1, 1)
        
        # self.toggle_map_type = Toggle()
        self.toggle_map_type = ToggleButton()
        self.average_trace_group.glayout.addWidget(self.toggle_map_type, 5, 1, 1, 1)

        self.APD_map_label = QLabel("APD Maps")
        self.average_trace_group.glayout.addWidget(self.APD_map_label, 5, 2, 1, 1)
        
        self.make_interpolation_label = QLabel("Interpolate Maps")
        self.average_trace_group.glayout.addWidget(self.make_interpolation_label, 5, 3, 1, 1)

        self.make_interpolation_check = QCheckBox()
        self.make_interpolation_check.setChecked(False)
        self.average_trace_group.glayout.addWidget(self.make_interpolation_check, 5, 4, 1, 1)
        
        self.activation_map_percentage_label = QLabel("APD Map percentage")
        self.average_trace_group.glayout.addWidget(self.activation_map_percentage_label, 6, 0, 1, 1)

        self.slider_APD_map_percentage = QLabeledSlider(Qt.Orientation.Horizontal)
        self.slider_APD_map_percentage.setRange(10, 100)
        self.slider_APD_map_percentage.setValue(75)
        self.slider_APD_map_percentage.setSingleStep(5)
        self.average_trace_group.glayout.addWidget(self.slider_APD_map_percentage, 6, 1, 1, 1)
        
        self.make_maps_btn = QPushButton("Create Maps")
        self.average_trace_group.glayout.addWidget(self.make_maps_btn, 6, 2, 1, 3)

        self.average_roi_on_map_btn = QPushButton("Get current ROI mean")
        self.average_trace_group.glayout.addWidget(self.average_roi_on_map_btn, 7, 0, 1, 1)
        
        self.average_roi_value_container = QLineEdit()
        self.average_roi_value_container.setPlaceholderText("select a ROI and click the 'Get current ROI mean' button.")
        self.average_trace_group.glayout.addWidget(self.average_roi_value_container, 7, 1, 1, 1)

        self.plot_APD_boundaries_btn = QPushButton("display boundaries")
        self.average_trace_group.glayout.addWidget(self.plot_APD_boundaries_btn, 7, 2, 1, 3)



        ##### Postprocessing Map group ########
        self.postprocessing_group = VHGroup('Postprocessing Maps', orientation='G')
        # self._mapping_processing_layout.addWidget(self.postprocessing_group.gbox)

        self.maps_plot_widget =  BaseNapariMPLWidget(self.viewer) # this is the cleanest widget thatz does not have any callback on napari
        self.postprocessing_group.glayout.addWidget(self.maps_plot_widget, 0, 0, 1, 5)

        self.maps_selector_label =  QLabel("Image selector:")
        self.maps_selector_label.setToolTip("Image Selector for diplaying and post-processing maps.")
        self.postprocessing_group.glayout.addWidget(self.maps_selector_label, 1, 0, 1, 1)

        # self.map_imgs_selector = MultiComboBox()
        self.map_imgs_selector = QListWidget()
        # allow for multiple selection
        self.map_imgs_selector.setSelectionMode(
            QtWidgets.QAbstractItemView.ExtendedSelection
        )
        # self.map_imgs_selector.addItems(["Option 1", "Option 2", "Option 3", "Option 4"])
        self.postprocessing_group.glayout.addWidget(self.map_imgs_selector, 1, 1, 1, 4)

        self.map_lower_clip_limit_label = QLabel("Set Lower limit")
        self.postprocessing_group.glayout.addWidget(self.map_lower_clip_limit_label, 2, 0, 1, 1)
        self.map_lower_clip_limit =  QLineEdit()
        self.map_lower_clip_limit.setValidator(QDoubleValidator()) 
        self.map_lower_clip_limit.setFixedWidth(50)
        self.map_lower_clip_limit.setText(f"{0}")
        self.postprocessing_group.glayout.addWidget(self.map_lower_clip_limit, 2, 1, 1, 1)

        self.plot_curr_map_btn = QPushButton("Plot Maps")
        self.postprocessing_group.glayout.addWidget(self.plot_curr_map_btn, 2, 4, 1, 1)

        self.colormap_n_levels_label = QLabel("No Levels (colormap)")
        self.postprocessing_group.glayout.addWidget(self.colormap_n_levels_label, 2, 2, 1, 1)
        
        self.colormap_n_levels = QSpinBox()
        self.colormap_n_levels.setSingleStep(1)
        self.colormap_n_levels.setValue(10)
        self.postprocessing_group.glayout.addWidget(self.colormap_n_levels, 2, 3, 1, 1)
        
        self.map_upper_clip_limit_label = QLabel("Set Upper limit")
        self.postprocessing_group.glayout.addWidget(self.map_upper_clip_limit_label, 3, 0, 1, 1)
        self.map_upper_clip_limit =  QLineEdit()
        self.map_upper_clip_limit.setValidator(QDoubleValidator()) 
        self.map_upper_clip_limit.setFixedWidth(50)
        self.map_upper_clip_limit.setText(f"{200}")
        self.postprocessing_group.glayout.addWidget(self.map_upper_clip_limit, 3, 1, 1, 1)



        self.apply_cip_limits_map_label = QLabel("Apply limits")
        self.postprocessing_group.glayout.addWidget(self.apply_cip_limits_map_label, 3, 2, 1, 1)
        
        self.apply_cip_limits_map = QCheckBox()
        self.apply_cip_limits_map.setChecked(False)
        self.postprocessing_group.glayout.addWidget(self.apply_cip_limits_map, 3, 3, 1, 1)

        self.clear_curr_map_btn = QPushButton("Clear Maps")
        self.postprocessing_group.glayout.addWidget(self.clear_curr_map_btn, 3, 4, 1, 1)
        
        self.erode_siluete_label = QLabel("Reduce Map Edge (px)")
        self.postprocessing_group.glayout.addWidget(self.erode_siluete_label, 4, 0, 1, 1)

        self.n_pixels_erode = QSpinBox()
        self.n_pixels_erode.setSingleStep(1)
        self.n_pixels_erode.setValue(1)
        self.postprocessing_group.glayout.addWidget(self.n_pixels_erode, 4, 1, 1, 1)

        self.preview_postProcessingMAP_btn = QPushButton("Preview")
        self.postprocessing_group.glayout.addWidget(self.preview_postProcessingMAP_btn, 4, 2, 1, 1)


        # Adding mapping subtabs
        self.mapping_tabs.addTab(self.average_trace_group.gbox, 'Pre-processing Maps')
        self.mapping_tabs.addTab(self.postprocessing_group.gbox, 'Post-processing Maps')
        self._mapping_processing_layout.addWidget(self.mapping_tabs)





        ######## Settings tab ########
        ####################################

        ######## Macro record group ########
        self._settings_layout.setAlignment(Qt.AlignTop)
        self.processing_steps_group = VHGroup('Tracking analyis steps', orientation='G')

        self.record_script_label = QLabel("Your current actions")
        self.record_script_label.setToolTip('Display bellow the recorded set of actions of your processing pipeline.')
        self.processing_steps_group.glayout.addWidget(self.record_script_label, 1, 0, 1, 4)
       
        self.processing_steps_tree = QTreeWidget()
        self.processing_steps_tree.setColumnCount(2)
        self.processing_steps_tree.setHeaderLabels(["Step ID", "Operation"])
        # self.processing_steps_tree.setStyleSheet("border: 1px solid black;") 
        # self.processing_steps_tree.setPlaceholderText("###### Start doing operations to populate your macro ######")
        self.processing_steps_group.glayout.addWidget(self.processing_steps_tree, 2, 0, 1, 4)

        # self.activate_macro_label = QLabel("Enable/disable Macro recording")
        # self.activate_macro_label.setToolTip('Set on if you want to keep track of the script for reproducibility or further reuse in batch processing')
        # self.processing_steps_group.glayout.addWidget(self.activate_macro_label, 3, 0, 1, 1)
        
        # self.record_macro_check = QCheckBox()
        # self.record_macro_check.setChecked(True) 
        # self.processing_steps_group.glayout.addWidget(self.record_macro_check,  3, 1, 1, 1)

        # self.clear_last_step_macro_btn = QPushButton("Delete last step")
        # self.processing_steps_group.glayout.addWidget(self.clear_last_step_macro_btn,  3, 2, 1, 1)
        
        # self.clear_macro_btn = QPushButton("Clear Macro")
        # self.processing_steps_group.glayout.addWidget(self.clear_macro_btn,  3, 3, 1, 1)       

        self.record_operations_label = QLabel("Record operations")
        self.record_operations_label.setToolTip('Set on if you want to keep track of the processing steps and operations to be recorded and added to the meatadata.')
        self.processing_steps_group.glayout.addWidget(self.record_operations_label, 3, 0, 1, 1)
        
        self.record_metadata_check = QCheckBox()
        self.record_metadata_check.setChecked(True) 
        self.processing_steps_group.glayout.addWidget(self.record_metadata_check,  3, 1, 1, 1)

        
        self._APD_analysis_layout.addWidget(self.APD_plot_group.gbox)
        self._APD_analysis_layout.addWidget(self.APD_export_group.gbox)
        

        # sub_backg_btn = QPushButton("Subtract Background")
        # rmv_backg_btn = QPushButton("Delete Background")
        # pick_frames_btn = QPushButton("Pick frames")

        ##### instanciate buttons #####
        
        # segmentation
        

        # self.layout().addWidget(seg_heart_btn)
        # self.layout().addWidget(sub_backg_btn)
        # self.layout().addWidget(rmv_backg_btn)
        # self.layout().addWidget(pick_frames_btn)


        ############################################
        ##### Metadata display  widget #############
        ############################################

        self.metadata_display_group = VHGroup('Image metadata', orientation='G')
        self.metadata_tree = QTreeWidget()
        # self.metadata_tree.setGeometry(30, 30, 300, 100)
        self.metadata_tree.setColumnCount(2)
        self.metadata_tree.setHeaderLabels(["Parameter", "Value"])
        self.metadata_display_group.glayout.addWidget(self.metadata_tree, 0, 0, 1, 4)

        self.export_data_metadata_group = VHGroup('Export Data / Processing steps', orientation='G')

        self.name_image_to_export_label =  QLabel("Save Image as:")
        self.export_data_metadata_group.glayout.addWidget(self.name_image_to_export_label,  0, 0, 1, 1)

        self.name_image_to_export =  QLineEdit()
        self.name_image_to_export.setToolTip('Define name to save current selected image + metadata in .tiff format.')
        self.name_image_to_export.setPlaceholderText("my_image")
        self.export_data_metadata_group.glayout.addWidget(self.name_image_to_export,  0, 1, 1, 1)

        self.name_procsteps_to_export_label =  QLabel("Save Proc-steps as:")
        self.export_data_metadata_group.glayout.addWidget(self.name_procsteps_to_export_label,  0, 2, 1, 1)

        self.procsteps_file_name =  QLineEdit()
        self.procsteps_file_name.setToolTip('Define name to save processing steps of curremnt image in .yml format.')
        self.procsteps_file_name.setPlaceholderText("ProcessingSteps")
        self.export_data_metadata_group.glayout.addWidget(self.procsteps_file_name,  0, 3, 1, 1)

        self._save_img_dir_box_text_label = QLabel("To Directory:")
        self._save_img_dir_box_text_label.setToolTip("Type the directory path or drag and drop folders here to change the current directory.")
        self.export_data_metadata_group.glayout.addWidget(self._save_img_dir_box_text_label, 1, 0, 1, 1)

        self.save_img_dir_box_text = QLineEdit()
        self.save_img_dir_box_text.installEventFilter(self)
        self.save_img_dir_box_text.setAcceptDrops(True)
        self.save_img_dir_box_text.setDragEnabled(True)
        self.save_img_dir_box_text.setPlaceholderText(os.getcwd())
        self.export_data_metadata_group.glayout.addWidget(self.save_img_dir_box_text, 1, 1, 1, 2)

        self.change_dir_to_save_img_btn = QPushButton("Change Directory")
        self.export_data_metadata_group.glayout.addWidget(self.change_dir_to_save_img_btn,  1, 3, 1, 1)
        
        self.export_image_btn = QPushButton("Export Image + metadata")
        self.export_data_metadata_group.glayout.addWidget(self.export_image_btn,  0, 4, 1, 1)

        self.export_processing_steps_btn = QPushButton("Export Proc-steps")
        self.export_data_metadata_group.glayout.addWidget(self.export_processing_steps_btn, 1, 4, 1, 1)
        # self.layout().addWidget(self.metadata_display_group.gbox) # temporary silence hide the metadatda

        # self._settings_layout.setAlignment(Qt.AlignTop)
        # self.macro_group = VHGroup('Record the scrips for analyis', orientation='G')

        



        self._settings_layout.addWidget(self.metadata_display_group.gbox)
        self._settings_layout.addWidget(self.processing_steps_group.gbox)
        self._settings_layout.addWidget(self.export_data_metadata_group.gbox)


        ######################
        ##### Plotters ######
        ######################

        ##### using pyqtgraph ######
        # self.plotting_group = VHGroup('Plot profile', orientation='G')
        # self.layout().addWidget(self.plotting_group.gbox)

        ######## pre-processing btns ########
        # self.inv_img_label = QLabel("Invert image")
        # self.pre_processing_group.glayout.addWidget(self.inv_img_label, 3, 0, 1, 1)
        # self.inv_data_btn = QPushButton("Apply")
        # self.pre_processing_group.glayout.addWidget(self.inv_data_btn, 3, 1, 1, 1)

        # self.plotting_group = VHGroup('Pre-porcessing', orientation='G')
        # self._pre_processing_layout.addWidget(self.plotting_group.gbox)







        # graph_container = QWidget()

        # # histogram view
        # self._graphics_widget = pg.GraphicsLayoutWidget()
        # self._graphics_widget.setBackground("w")


        # self.plotting_group.glayout.addWidget(self._graphics_widget, 3, 0, 1, 1)

        # hour = [1,2,3,4,5,6,7,8,9,10]
        # temperature = [30,32,34,32,33,31,29,32,35,45]

        # self.p2 = self._graphics_widget.addPlot()
        # axis = self.p2.getAxis('bottom')
        # axis.setLabel("Distance")
        # axis = self.p2.getAxis('left')
        # axis.setLabel("Intensity")

        # self.p2.plot(hour, temperature, pen="red", name="test")

        # individual layers: legend
        # self._labels = QWidget()
        # self._labels.setLayout(QVBoxLayout())
        # # self._labels.layout().setSpacing(0)

        # # setup layout
        # self.setLayout(QVBoxLayout())

        # self.layout().addWidget(graph_container)
        # self.layout().addWidget(self._labels)


        # ##### using TSPExplorer ######

        # self._graphics_widget_TSP = TSPExplorer(self.viewer)
        # self.layout().addWidget(self._graphics_widget_TSP, 1)

        
        

       




        ##################################################################
        ############################ callbacks ###########################
        ##################################################################
        
        self.inv_data_btn.clicked.connect(self._on_click_inv_data_btn)
        self.apply_normalization_btn.clicked.connect(self._on_click_norm_data_btn)
        self.inv_and_norm_data_btn.clicked.connect(self._on_click_inv_and_norm_data_btn)
        self.split_chann_btn.clicked.connect(self._on_click_splt_chann)
        # self.glob_norm_data_btn.clicked.connect(self._on_click_glob_norm_data_btn)
        # self.rmv_backg_btn.clicked.connect(self._on_click_seg_heart_btn)

        self.apply_spat_filt_btn.clicked.connect(self._on_click_apply_spat_filt_btn)
        # self.filter_types.activated.connect(self._filter_type_change)
        # rmv_backg_btn.clicked.connect(self._on_click_rmv_backg_btn)
        # sub_backg_btn.clicked.connect(self._on_click_sub_backg_btn)
        # self.pick_frames_btn.clicked.connect(self._on_click_pick_frames_btn)
        # inv_and_norm_btn.clicked.connect(self._on_click_inv_and_norm_btn)
        # inv_and_norm_btn.clicked.connect(self._on_click_inv_data_btn, self._on_click_norm_data_btn)
        # load_ROIs_btn.clicked.connect(self._on_click_load_ROIs_btn)
        # save_ROIs_btn.clicked.connect(self._on_click_save_ROIs_btn)
        # self.ROI_selection.currentIndexChanged.connect(self.???)
        self.ROI_selection_1.activated.connect(self._get_ROI_selection_1_current_text)
        self.ROI_selection_2.activated.connect(self._get_ROI_selection_2_current_text)
        self.copy_ROIs_btn.clicked.connect(self._on_click_copy_ROIS)
        # self.apply_mot_correct_btn.clicked.connect(self._on_click_apply_mot_correct_btn)
        # self.transform_to_uint16_btn.clicked.connect(self._on_click_transform_to_uint16_btn)
        self.apply_temp_filt_btn.clicked.connect(self._on_click_apply_temp_filt_btn)
        self.compute_APD_btn.clicked.connect(self._get_APD_call_back)
        self.clear_plot_APD_btn.clicked.connect(self._clear_APD_plot)
        self.slider_APD_detection_threshold.valueChanged.connect(self._get_APD_thre_slider_vlaue_func)
        self.slider_APD_detection_threshold_2.valueChanged.connect(self._get_APD_thre_slider_vlaue_func)
        self.load_spool_dir_btn.clicked.connect(self._load_current_spool_dir_func)
        self.search_spool_dir_btn.clicked.connect(self._search_and_load_spool_dir_func)
        self.copy_APD_rslts_btn.clicked.connect(self._on_click_copy_APD_rslts_btn_func)
        self.search_dir_APD_rslts_btn.clicked.connect(self._on_click_search_new_dir_APD_rslts_btn_func)
        self.save_APD_rslts_btn.clicked.connect(self._on_click_save_APD_rslts_btn_func)
        self.APD_computing_method.activated.connect(self._get_APD_call_back)
        # self.get_AP_btn.clicked.connect(self.show_pop_window_ave_trace)
        self.preview_AP_splitted_btn.clicked.connect(self._preview_multiples_traces_func)
        self.remove_mean_check.stateChanged.connect(self._remove_mean_check_func)
        self.slider_N_APs.valueChanged.connect(self._slider_N_APs_changed_func)
        self.mv_left_AP_btn.clicked.connect(self._on_click_mv_left_AP_btn_func)
        self.mv_righ_AP_btn.clicked.connect(self._on_click_mv_right_AP_btn_func)
        self.clear_AP_splitted_btn.clicked.connect(self._on_click_clear_AP_splitted_btn_fun )
        self.create_average_AP_btn.clicked.connect(self._on_click_create_average_AP_btn_func )
        self.make_maps_btn.clicked.connect(self._on_click_make_maps_btn_func)
        self.create_AP_gradient_btn.clicked.connect(self._on_click_create_AP_gradient_bt_func)
        self.apply_auto_segmentation_btn.clicked.connect(lambda: self._on_click_apply_segmentation_btn_fun(return_result_as_layer=True, return_mask=False)) # This trick allow to pass just the param as expected
        self.average_roi_on_map_btn.clicked.connect(self._on_click_average_roi_on_map_btn_fun)
        self.plot_histogram_btn.clicked.connect(self._on_click_plot_histogram_btn_func)
        self.clear_histogram_btn.clicked.connect(self._on_click_clear_histogram_btn_func)
        self.clip_trace_btn.clicked.connect(self._on_click_clip_trace_btn_func)
        self.is_range_clicked_checkbox.stateChanged.connect(self._dsiplay_range_func)
        self.double_slider_clip_trace.valueChanged.connect(self._double_slider_clip_trace_func)
        self.export_processing_steps_btn.clicked.connect(self._export_processing_steps_btn_func)
        self.export_image_btn.clicked.connect(self._export_image_btn_func)
        self.change_dir_to_save_img_btn.clicked.connect(self.change_dir_to_save_img_btn_func)
        self.apply_optimap_mot_corr_btn.clicked.connect(self._apply_optimap_mot_corr_btn_func)
        self.crop_from_shape_btn.clicked.connect(self._on_click_crop_from_shape_btn_func)
        self.apply_manual_segmentation_btn.clicked.connect(self._on_click_segment_manual_btn_func)
        self.plot_APD_boundaries_btn.clicked.connect(self._plot_APD_boundaries_btn_func)
        self.compute_ratio_btn.clicked.connect(self._compute_ratio_btn_func)
        self.slider_APD_percentage.valueChanged.connect(self._update_APD_value_for_MAP_func)
        self.slider_APD_map_percentage.valueChanged.connect(self._update_APD_value_for_APD_func)
        self.x_scale_box.textChanged.connect(self._update_x_scale_box_func)
        self.plot_curr_map_btn.clicked.connect(self._plot_curr_map_btn_fun)
        self.clear_curr_map_btn.clicked.connect(self._clear_curr_map_btn_func)
        self.preview_postProcessingMAP_btn.clicked.connect(self._preview_postProcessingMAP_btn_func)
        self.crop_all_views_and_rotate_btn.clicked.connect(self._crop_all_views_and_rotate_btn_func)
        self.return_bounding_boxes_only_btn.clicked.connect(lambda: self._crop_all_views_and_rotate_btn_func(return_only_bounding_box=True))
        self.join_all_views_and_rotate_btn.clicked.connect(self._join_all_views_and_rotate_btn_func)
        
        
        
        ##### handle events #####
        # self.viewer.layers.events.inserted.connect(self._shapes_layer_list_changed_callback)
        # self.viewer.layers.events.removed.connect(self._shapes_layer_list_changed_callback)
        # self.viewer.layers.events.reordered.connect(self._shapes_layer_list_changed_callback)
        self.viewer.layers.selection.events.active.connect(self._retrieve_metadata_call_back)
        # self.plot_widget.plotter.selector.model().itemChanged.connect(self._get_current_selected_TSP_layer_callback)
        # callback for insert /remove / reordered layers
        self.viewer.layers.events.inserted.connect(self._layer_list_changed_callback)
        self.viewer.layers.events.removed.connect(self._layer_list_changed_callback)
        self.viewer.layers.events.reordered.connect(self._layer_list_changed_callback)
        # callback for selection of layers in the selectors
        self.listShapeswidget.itemClicked.connect(self._data_changed_callback)
        self.listImagewidget.itemClicked.connect(self._data_changed_callback)
        # updtae FPS label
        self.viewer.window._qt_viewer.canvas._scene_canvas.measure_fps(callback = self.update_fps)
        # callback for trace plotting
        # self.plot_profile_btn.clicked.connect(self._on_click_plot_profile_btn_func)
        self.plot_profile_btn.stateChanged.connect(self._on_click_plot_profile_btn_func)
        # self.selection_layer.events.data.connect()
        

    def _on_click_inv_data_btn(self):
        current_selection = self.viewer.layers.selection.active

        try:
            if isinstance(current_selection, Image):
                results =invert_signal(current_selection.data)

                self.add_result_img(
                    result_img=results,
                    operation_name="invert_signal",
                    method_name= "invert_signal",
                    sufix="Inv", 
                    parameters=None, 
                    )
                print(f"{'*'*5} Applying '{invert_signal.__name__}' to image: '{current_selection}' {'*'*5} ")
                
                
            else:
                return warn(f"Select an Image layer to apply this function. \nThe selected layer: '{current_selection}' is of type: '{current_selection._type_string}'")
        except Exception as e:
            raise CustomException(e, sys)
            # print (CustomException(e, sys))
            


    def _on_click_norm_data_btn(self):
        current_selection = self.viewer.layers.selection.active

        type_of_normalization = self.data_normalization_options.currentText()
        normalization_methods = [self.data_normalization_options.itemText(i) for i in range(self.data_normalization_options.count())]

        if isinstance(current_selection, Image):
            try:
                
                add_metadata = self.record_metadata_check.isChecked()

                if type_of_normalization == normalization_methods[0]:                    
                    suffix = "Loc"
                    method_name = "local_normal_fun"
                    results = local_normal_fun(current_selection.data)

                elif type_of_normalization == normalization_methods[1]:
                    wind_size = self.slide_wind_n.value()
                    suffix = f"SliWind{wind_size}"
                    method_name = "slide_window_normalization_func"
                    results = slide_window_normalization_func(current_selection.data, slide_window=wind_size)

                elif type_of_normalization == normalization_methods[2]:
                    suffix = "Glob"
                    method_name = "global_normal_fun"
                    results = global_normal_fun(current_selection.data)
                else:
                    warn(f"Normalization method '{type_of_normalization}' no found.")

                parameters= {"Normalization_method": type_of_normalization}
                parameters = {"Normalization_method": type_of_normalization, "options": {"slide_window" : wind_size}} if type_of_normalization == normalization_methods[1] else parameters
                
                self.add_result_img(
                    result_img=results,
                    operation_name="Normalization",
                    method_name= method_name,
                    sufix=f"Norm{suffix}", 
                    parameters=parameters, 
                    track_metadata=add_metadata,
                    )
                
                print(f"{'*'*5} Applying normalization:'{type_of_normalization}' to image: '{current_selection}' {'*'*5} ")
            except Exception as e:
                raise CustomException(e, sys)
        else:
           return  warn(f"Select an Image layer to apply this function. \nThe selected layer: '{current_selection}' is of type: '{current_selection._type_string}'")


    def _on_click_inv_and_norm_data_btn(self):
        self._on_click_inv_data_btn()
        self._on_click_norm_data_btn()


    def _on_click_splt_chann(self):
        current_selection = self.viewer.layers.selection.active

        if isinstance(current_selection, Image):
            print(f'applying "split_channels" to image {current_selection}')
            my_splitted_images = split_channels_fun(current_selection.data)
            curr_img_name = current_selection.name
            metadata = current_selection.metadata
            # overwrite the new cycle time
            if not "CycleTime" in metadata:
                warn('not "CycleTime" found in metadata. Setted to 1')
                metadata["CycleTime"] = 1
            if "CycleTime" in metadata:
                half_cycle_time = metadata["CycleTime"] * 2
            
            new_metadata = metadata.copy()
            new_metadata["CycleTime"] = half_cycle_time


            for channel in range(len(my_splitted_images)):
                
                params = {"Channel" : {"current_channel":channel},
                          "cycle_time_changed": {"original_cycle_time": round(metadata["CycleTime"], 3), 
                                                 "new_cycle_time": round(half_cycle_time, 3)}
                                                 }
                self.add_result_img(result_img=my_splitted_images[channel], 
                                    operation_name="Split_Channels", 
                                    method_name="split_channels_fun", 
                                    sufix=f"Ch{channel}", 
                                    custom_metadata=new_metadata, 
                                    custom_img_name= curr_img_name,
                                    custom_outputs=[curr_img_name + "_Ch0", curr_img_name + "_Ch1"],
                                    parameters=params)

                
        else:
            warn(f"Select an Image layer to apply this function. \nThe selected layer: '{current_selection}' is of type: '{current_selection._type_string}'")

    
    # def _on_click_glob_norm_data_btn(self):
    # NOTE: this fucntion can retire soon. updated on new qcombobox method [self.data_normalization_options] on date 11.04.224
    #     current_selection = self.viewer.layers.selection.active
        
    #     if isinstance(current_selection, Image):
    #         results = global_normal_fun(current_selection.data)
    #         self.add_result_img(result_img=results, single_label_sufix="GloNor", add_to_metadata = "Global_norm_signal")

    #     else:
    #         return warn(f"Select an Image layer to apply this function. \nThe selected layer: '{current_selection}' is of type: '{current_selection._type_string}'")


    #  NOTE: deprecated on 07.02.2023 you can kill this function
    
    # def get_rois_list(self):

    #     shape_layer_list = [layer.name for layer in self.viewer.layers if layer._type_string == 'shapes']
        
    #     return shape_layer_list

    # def update_roi_list(self):

    #     self.clear()
    #     self.addItems(self.get_rois_list())
        

    
        
    # def _shapes_layer_list_changed_callback(self, event):
    #      if event.type in ['inserted', 'removed']:
    #         value = event.value
    #         etype = event.type
    #         if value._type_string == 'shapes' :
    #             if value:
    #                 if etype == 'inserted':  # add layer to model
    #                     # print("you  enter the event loop")
    #                     self.ROI_selection_1.clear()
    #                     self.ROI_selection_1.addItems(self.get_rois_list()) 
    #                     self.ROI_selection_2.clear()
    #                     self.ROI_selection_2.addItems(self.get_rois_list())
                        
    #                 elif etype == 'removed':  # remove layer from model
    #                     self.ROI_selection_1.clear()
    #                     self.ROI_selection_1.addItems(self.get_rois_list())
    #                     self.ROI_selection_2.clear()
    #                     self.ROI_selection_2.addItems(self.get_rois_list())
                        

    #                 elif etype == 'reordered':  # remove layer from model
    #                     self.ROI_selection_1.clear()
    #                     self.ROI_selection_1.addItems(self.get_rois_list())
    #                     self.ROI_selection_2.clear()
    #                     self.ROI_selection_2.addItems(self.get_rois_list())
    
    def _on_click_copy_ROIS(self):
        
        shape1_name = self.ROI_selection_1.currentText()
        shape2_name = self.ROI_selection_2.currentText()

        # shapes_from = self.viewer.layers[shape1_name]
        # shapes_to = self.viewer.layers[shape2_name]
        
        self.viewer.layers[shape2_name].data = self.viewer.layers[shape1_name].data

        # no sure why the append method is not working
        # for shapes in self.viewer.layers[shape1_name].data:
        #     self.viewer.layers[shape2_name].data.append(shapes)
        
        # shapes_to.selected_data = set()
        

        
        # print(f"copy file shapes from layer _>{shape1_name} to layer ->{shape2_name}")
        # print(f" number of shapes from: {len(shapes_from)} and shapes to: {len(shapes_to)}/nshpes frm--->{shapes_from} /nshapes to, {shapes_to}" )
    # def _filter_type_change(self, _):
    #    ctext = self.filter_types.currentText()
    #    print(f"Current layer 1 is {ctext}")
    
    def _compute_ratio_btn_func(self):
        
        
        # if [img0.data.shape] != [img1.data.shape]:
        #     return warn(f"The shape of your images does not seems to be the same. Please check the images. dim of '{img0_name}' = {img0.data.shape} and dim of '{img1_name}' = {img1.data.shape}")
        # else :
        img0_name = self.Ch0_ratio.currentText()
        img0 = self.viewer.layers[img0_name].data
        img1_name = self.Ch1_ratio.currentText()
        img1 = self.viewer.layers[img1_name].data

        # metadata = img0.metadata
        params = {"is_ratio_inverted": self.is_ratio_inverted.isChecked()}

        try:

            # check that the dimensions are compatible for broadcasting division
            # Get the shapes of the arrays
            shape1 = img0.shape
            shape2 = img1.shape

            # Determine the minimum shape along each dimension
            min_shape = tuple(min(s1, s2) for s1, s2 in zip(shape1, shape2))

            # Slice the larger array to match the smaller one
            img0 = img0[:min_shape[0], :min_shape[1], :min_shape[2]]
            img1 = img1[:min_shape[0], :min_shape[1], :min_shape[2]]


            if self.is_ratio_inverted.isChecked():
                results = img1/img0
                
                self.add_result_img(result_img=results, operation_name= "Compute_Ratio", method_name="/", sufix=f"RatCh1Ch0", custom_inputs=[img1_name, img0_name], )                                    
                
                print(f"Computing ratio of '{img1_name[:20]}...{img1_name[-5:]}' / '{img0_name[:20]}...{img0_name[-5:]}'")

            else:
                results = img0/img1
                
                self.add_result_img(result_img=results, operation_name= "Compute_Ratio", method_name="/", sufix=f"RatCh0Ch1", custom_inputs=[img0_name, img1_name], parameters=params)                                    

                print(f"Computing ratio of '{img0_name[:20]}...{img0_name[-5:]}' / '{img1_name[:20]}...{img1_name[-5:]}'")

        except Exception as e:
            print(CustomException(e, sys))

    def _on_click_apply_spat_filt_btn(self):
        current_selection = self.viewer.layers.selection.active
        if isinstance(current_selection, Image):
        
            filter_type = self.spat_filter_types.currentText()
            all_my_filters = [self.spat_filter_types.itemText(i) for i in range(self.spat_filter_types.count())]
            sigma = self.sigma_filt_spatial_value.value()
            kernel_size = self.filt_kernel_value.value()
            sigma_col = self.sigma_filt_color_value.value()
            metadata = current_selection.metadata

            try:
                            
                if filter_type == all_my_filters[0]:
                    results = apply_gaussian_func(current_selection.data, 
                                                sigma= sigma, 
                                                kernel_size=kernel_size)
                    
                    met_name = "apply_gaussian_func"
                    params = {
                        "filter_type":filter_type,
                        "sigma": sigma,
                        "kernel_size": kernel_size
                        }
                
                elif filter_type == all_my_filters[3]:
                    results = apply_median_filt_func(current_selection.data, kernel_size)
                    met_name = "apply_median_filt_func"
                    params = {
                        "filter_type":filter_type,
                        "kernel_size": kernel_size
                        }

                elif filter_type == all_my_filters[1]:
                    results = apply_box_filter(current_selection.data, kernel_size)
                    met_name = "apply_box_filter"
                    params = {
                        "filter_type":filter_type,
                        "kernel_size": kernel_size
                        }
                
                elif filter_type == all_my_filters[2]:
                    results = apply_laplace_filter(current_selection.data, kernel_size=kernel_size, sigma=sigma)
                    met_name = "apply_laplace_filter"
                    params = {
                        "filter_type":filter_type,
                         "sigma": sigma,
                        "kernel_size": kernel_size
                        }
                                    
                elif filter_type == all_my_filters[4]:
                    results = apply_bilateral_filter(current_selection.data, sigma_spa=sigma, sigma_col = sigma_col, wind_size = kernel_size)
                    met_name = "apply_bilateral_filter"
                    params = {
                        "filter_type":filter_type,
                         "sigma_spa": sigma,
                         "sigma_col": sigma_col,
                        "wind_size": kernel_size
                        }
                
                
                self.add_result_img(result_img=results, operation_name="Saptial_filter", method_name=met_name, sufix= f"SpatFilt{filter_type[:4]}", parameters=params)
                print(f"{'*'*5} Applying '{filter_type}' filter to image: '{current_selection}' {'*'*5} ")
                
            except Exception as e:
                raise CustomException(e, sys)
                # print( CustomException(e, sys))
                
                

        else:
            warn(f"Select an Image layer to apply this function. \nThe selected layer: '{current_selection}' is of type: '{current_selection._type_string}'")
                
    
    
    def add_result_img(self, 
                        result_img, 
                        operation_name, 
                        method_name,
                        custom_metadata = None,
                        custom_img_name = None,
                        custom_inputs = None,
                        custom_outputs = None,
                        sufix = None, 
                        parameters = None,
                        track_metadata = None, 
                        colormap="turbo"):
        """
        add_result_img: Add new image and handle metadata.

        This function create and handle metadata for the 
        different processing steps or operations in images.

        Parameters
        ----------
        result_img : 'napari.types.ImageData'
            Image resulting from an operation.

        operation_name : srt, optional
            Name of the current processing step or operation, by default None

        method_name : srt, optional
            Name of the current method or function used, by default None

        sufix : str, optional
            Add sufix to image, by default None

        parameters : dict, optional
            Set of parameters currently being used, by default None

        track_metadata : bool, optional
            Set True if you wish to keep track of operations changes and be added to metadata, by default True

        colormap : str, optional
            Pseudo-color definition for resulting image, by default "turbo"

        """

        if custom_metadata is None:
            img_metadata = copy.deepcopy(self.viewer.layers.selection.active.metadata)
        else:
            img_metadata = copy.deepcopy(custom_metadata)

        if custom_img_name is None:
            img_name = self.viewer.layers.selection.active.name
        else:
            img_name = custom_img_name
        new_img_name = img_name

        if sufix is not None:
            new_img_name += f"_{sufix}"

        if custom_inputs is None:
            custom_inputs = [img_name]
        else:
            custom_inputs
        
        if custom_outputs is None:
            custom_outputs = [new_img_name]
        else:
            custom_outputs


        track_metadata = self.record_metadata_check.isChecked() if track_metadata is None else track_metadata
        
        if track_metadata:

            # create "ProcessingSteps" key if does not exist
            key_name = "ProcessingSteps"

            # if key_name in img_metadata:
            self.metadata_recording_steps.steps = img_metadata[key_name] if key_name in img_metadata else []
                # img_metadata[key_name] = []
            # else:
            #     self.metadata_recording_steps.steps = img_metadata[key_name] if len(img_metadata[key_name]) > 1, else: 
            
            self.metadata_recording_steps.add_step(
                    operation=operation_name,
                    method_name=method_name,
                    inputs=custom_inputs,
                    outputs=custom_outputs,
                    parameters=parameters
                    )
            img_metadata[key_name] = self.metadata_recording_steps.steps
            
            
            return self.viewer.add_image(result_img,
                        colormap = colormap,
                        name = new_img_name,
                        metadata = img_metadata
                        )
        else:
            return self.viewer.add_image(result_img,
                        colormap = colormap,
                        name = new_img_name,
                        metadata = img_metadata
                        )


    
    # NOTE: refactor this function 21.082024
    # def add_result_img(self, 
    #                    result_img, 
    #                    single_label_sufix = None, 
    #                    auto_metadata = True, 
    #                    operation_name = None, 
    #                    parameters = None,
    #                    custom_metadata = None,
    #                    track_metadata = True, 
    #                    colormap="turbo", 
    #                    img_custom_name = None, 
    #                    **label_and_value_sufix):
    #     """
    #     add_result_img: Add new image and handle metadata.

    #     This function create and handle metadata for the 
    #     different processing steps or operations in images.

    #     Parameters
    #     ----------
    #     result_img : 'napari.types.ImageData'
    #         Image resulting from an operation.

    #     single_label_sufix : str, optional
    #         Add sufix to image, by default None

    #     auto_metadata : bool, optional
    #         If True, tkes curent image metadata and update it, otherwise a new metadata template is required, by default True

    #     add_to_metadata : str, optional
    #         Parameter to add as metadata, typically the name of the operation, by default None

    #     custom_metadata : dict, optional
    #         When auto_metadata = False, metadata dict to use as template, by default None

    #     track_metadata : bool, optional
    #         Set True if you wish to keep track of operations changes and be added to metadata, by default True

    #     colormap : str, optional
    #         Pseudo-color definition for resulting image, by default "turbo"
            
    #     img_custom_name : str, optional
    #         When you decide to change or modify the current image name, by default None
        
    #     Returns
    #     -------
    #     result_img_and_metadata : 'napari.types.ImageData'
    #         The image with metadata updated.
    #     """
        

    #     if auto_metadata:
    #         img_metadata = copy.deepcopy(self.viewer.layers.selection.active.metadata)
    #     else: 
    #         img_metadata = copy.deepcopy(custom_metadata)


    #     if img_custom_name is not None:
    #         img_name = img_custom_name
    #     else:
    #         img_name = self.viewer.layers.selection.active.name
        
    #     new_img_name = img_name
            
    #     if track_metadata:

    #         # create "ProcessingSteps" key if does not exist
    #         key_name = "ProcessingSteps"

    #         if key_name not in img_metadata:
    #             img_metadata[key_name] = []

    #     if single_label_sufix is not None:
    #         # for value in single_label_sufix:
    #         new_img_name += f"_{single_label_sufix}"

    #     if label_and_value_sufix is not None:
    #         for key, value in label_and_value_sufix.items():
    #             new_img_name += f"_{key}{value}"
                

    #         # append the given processing step(s) to the key
    #         if operation_name is not None:            

    #             self.metadata_recording_steps.add_step(
    #                 operation=operation_name,
    #                 inputs=[img_name],
    #                 outputs=[new_img_name],
    #                 parameters=[None]
    #                 )
    #             img_metadata[key_name].append(self.metadata_recording_steps.steps)



            
    #         return self.viewer.add_image(result_img,
    #                     colormap = colormap,
    #                     name = new_img_name,
    #                     metadata = img_metadata
    #                     )
    #     else:
            
    #         return self.viewer.add_image(result_img,
    #                     colormap = colormap,
    #                     name = new_img_name,
    #                     metadata = img_metadata
    #                     )

        
        


    def add_result_label(self, result_labl, 
                         single_label_sufix = None, 
                         metadata = True, 
                         add_to_metadata = None, 
                         colormap="turbo", 
                         img_custom_name = None, 
                         **label_and_value_sufix):
        
        if img_custom_name is not None:
            img_name = img_custom_name
        else:
            img_name = self.viewer.layers.selection.active.name

        self.curr_img_metadata = copy.deepcopy(self.viewer.layers.selection.active.metadata)

        key_name = "ProcessingSteps"
        if key_name not in self.curr_img_metadata:
            self.curr_img_metadata[key_name] = []

        if add_to_metadata is not None:            
            self.curr_img_metadata[key_name].append(add_to_metadata)


        if single_label_sufix is not None:
            # for value in single_label_sufix:
            img_name += f"_{single_label_sufix}"

        if label_and_value_sufix is not None:
            for key, value in label_and_value_sufix.items():
                img_name += f"_{key}{value}"
        
        
        if metadata:
            self.viewer.add_labels(result_labl, 
                        metadata = self.curr_img_metadata,
                        name = img_name)

        else: 
            self.viewer.add_labels(result_labl,
                        name = img_name)
    





        # print(f"its responding {str(self.filt_param.value())}")

        # results = apply_gaussian_func(self.viewer.layers.selection, sigma)

        # self.viewer.add_image(results, 
        # colormap = "turbo",
        # # colormap= "twilight_shifted", 
        # name= f"{self.viewer.layers.selection.active}_Gaus_{str(sigma)}")

    
    def _get_ROI_selection_1_current_text(self, _): # We receive the index, but don't use it.
        ctext = self.ROI_selection_1.currentText()
        print(f"Current layer 1 is '{ctext}'")

    def _get_ROI_selection_2_current_text(self, _): # We receive the index, but don't use it.
        ctext = self.ROI_selection_2.currentText()
        print(f"Current layer 2 is '{ctext}'")

    
    ####### NOTE: deprecating this method on 14.08.22024 #######
    # def _on_click_apply_mot_correct_btn(self):
    #     foot_print = self.footprint_size.value()
    #     radius_size = self.radius_size.value()
    #     n_warps = self.n_warps.value()
    #     try:
    #         subprocess.check_output('nvidia-smi')
    #         print('Nvidia GPU detected!')
    #     except Exception: # this command not being found can raise quite a few different errors depending on the configuration
    #         warn('No Nvidia GPU in system!, setting to default CPU use')
    #         self.use_GPU.setChecked(False)
        
    #     gpu_use = self.use_GPU.isChecked() # put this in the GUI         
    #     ref_frame_indx = int(self.ref_frame_val.text()) # put this in the GUI
    #     current_selection = self.viewer.layers.selection.active
    #     raw_data = current_selection.data
    #     if gpu_use == True:
    #         raw_data = cp.asarray(raw_data)

    #     if current_selection._type_string == "image":
                
    #         scaled_img = scaled_img_func(raw_data, 
    #                                     foot_print_size=foot_print)
                
    #         results = register_img_func(scaled_img, orig_data= raw_data, radius_size=radius_size, num_warp=n_warps, ref_frame=ref_frame_indx)
            
    #         if not isinstance(results, numpy_ndarray):
    #             results =  results.get()    

    #         self.add_result_img(results, MotCorr_fp = foot_print, rs = radius_size, nw=n_warps)
        
    #         

            
    #     else:
    #         warn(f"Select an Image layer to apply this function. \nThe selected layer: '{current_selection}' is of type: '{current_selection._type_string}'")

        
        # NOTE: DEPRECATE this function 21.08.2024
    # def _on_click_transform_to_uint16_btn(self):
        
    #     results = transform_to_unit16_func(self.viewer.layers.selection)
    #     # print( "is doing something")

    #     self.viewer.add_image(results, 
    #         colormap = "turbo",
    #      # colormap= "twilight_shifted", 
    #         name= f"{self.viewer.layers.selection.active}_uint16")

    def _on_click_apply_temp_filt_btn(self):
        current_selection = self.viewer.layers.selection.active
        
        if isinstance(current_selection, Image):
            filter_type = self.temp_filter_types.currentText()
            all_my_filters = [self.temp_filter_types.itemText(i) for i in range(self.temp_filter_types.count())]
            cutoff_freq_value = self.butter_cutoff_freq_val.text()
            order_value = self.butter_order_val.value()
            fps_val = float(self.fps_val.text())
            cutoff_freq_value = int(cutoff_freq_value)
            metadata = current_selection.metadata

            try:

                if filter_type == all_my_filters[0]:
                    
                    results = apply_butterworth_filt_func(current_selection.data, 
                                                        ac_freq=fps_val, 
                                                        cf_freq= cutoff_freq_value, 
                                                        fil_ord=order_value)
                    
                    met_name = "apply_butterworth_filt_func"
                    params = {
                    "filter_type":filter_type,
                    "acquisition_freq": fps_val,
                    "cutoff_freq": cutoff_freq_value,
                    "order_size": order_value
                    }
            
                elif filter_type == all_my_filters[1]:                   

                    n_taps = 21 #NOTE: this is hard coded, need to test it if make an impact
                    
                    results = apply_FIR_filt_func(current_selection.data, n_taps=n_taps, cf_freq=cutoff_freq_value, acquisition_freq = fps_val)

                    met_name = "apply_FIR_filt_func"
                    params = {
                    "acquisition_freq": fps_val,
                    "n_taps":n_taps,
                    "cutoff_freq": cutoff_freq_value,
                    }
                
                
                print(f"{'*'*5} Applying '{filter_type}' filter to image: '{current_selection}' {'*'*5} ")
                self.add_result_img(result_img=results, operation_name="Temporal_filter", method_name=met_name, sufix= f"TempFilt{filter_type[:4]}", parameters=params)
                
            
            except Exception as e:
                # raise CustomException(e, sys)
                print(CustomException(e, sys))
        else:
            warn(f"Select an Image layer to apply this function. \nThe selected layer: '{current_selection}' is of type: '{current_selection._type_string}'")
    
                

        # print (f"it's responding with freq: {freq_value},  order_val {order_value} and fps = {fps_val}")

    
    ####### helper functions #######
    # def add_result_img_helper_func(self, results, sufix_name = None, color_map = "turbo"):
        
    #     self.viewer.add_image(
    #         results,
    #         name = f"{self.viewer.layers.selection.active}_{str(sufix_name)}"
    #         colormap = color_map

    #     )

        

    
    # def _on_click_sub_backg_btn(self):


    
    # def _on_click_rmv_backg_btn(self):
    #     results =segment_heart_func(self.viewer.layers.selection)
    #     self.viewer.add_labels(results, 
    #     # colormap= "turbo", 
    #     name= f"{self.viewer.layers.selection.active}_Bck")

    def _on_click_seg_heart_btn(self):
        results =segment_heart_func(self.viewer.layers.selection)
        self.viewer.add_labels(results, 
        # colormap= "turbo", 
        name= f"{self.viewer.layers.selection.active}_Bck")

    def _on_click_pick_frames_btn(self):
        results =pick_frames_fun(self.viewer.layers.selection)
        self.viewer.add_image(results, 
        colormap= "twilight_shifted", 
        name= f"{self.viewer.layers.selection.active}_sliced")    
    # def _on_click_inv_and_norm_btn(self):
        # self._on_click_inv_data_btn(self)
        # self._on_click_norm_data_btn(self)




    # def _on_click_load_ROIs_btn(self, event=None, filename=None):
    #     if filename is None: filename, _ = QFileDialog.getOpenFileName(self, "Load ROIs", ".", "ImageJ ROIS(*.roi *.zip)")
    #     self.viewer.open(filename, plugin='napari_jroireader')
        
    
    # def _on_click_save_ROIs_btn(self, event=None, filename=None):
    #     if filename is None: filename, _ = QFileDialog.getSaveFileName(self, "Save as .csv", ".", "*.csv")
    #     # self.viewer.layers.save(filename, plugin='napari_jroiwriter')
    #     self.viewer.layers.save(filename, plugin='napari')

    # NOTE: deprecating this method on 05.09.2024
    # def _get_current_selected_TSP_layer_callback(self, event):
    #     # this object is a list of image(s) selected from the Time_series_plotter pluggin layer selector
    #             try:
    #                 self.current_seleceted_layer_from_TSP = self.main_plot_widget.plotter.selector.model().get_checked()[0].name
    #             except:
    #                 self.current_seleceted_layer_from_TSP = "ImageID"
                
    #             self.table_rstl_name.setPlaceholderText(f"{self.current_seleceted_layer_from_TSP}_APD_rslts")
    
    def _retrieve_metadata_call_back(self, event):

        try:
            layer = event.value
            etype = event.type
            # handle name change by bypasing the event to the _layer_list_changed_callback
            if layer is not None and not isinstance(layer, list):
                @layer.events.name.connect
                def _on_rename(name_event):
                    # print(f'Layer {id(layer)} changed name to {layer.name}')
                    self._layer_list_changed_callback(event)

            if etype in ['active']:
                if isinstance(layer, Image):
                    self.name_image_to_export.setPlaceholderText(self.viewer.layers.selection.active.name)

                    # handle metadata in images saved with tifffile
                    self.img_metadata_dict = self.viewer.layers.selection.active.metadata
                    self.img_metadata_dict = self.img_metadata_dict["shaped_metadata"][0] if "shaped_metadata" in self.img_metadata_dict else self.img_metadata_dict

                    # self.viewer.layers.selection.active.metadata = self.img_metadata_dict
                    # self.viewer.layers.selection.active.metadata = self.viewer.layers.selection.active.metadata["shaped_metadata"][0] if "shaped_metadata" in self.viewer.layers.selection.active.metadata else self.img_metadata_dict
                    # self.viewer.layers.selection.active.metadata = self.img_metadata_dict["shaped_metadata"][0] if "shaped_metadata" in self.img_metadata_dict else self.img_metadata_dict
                    if "ProcessingSteps" in self.img_metadata_dict:
                        self.processing_steps_tree.clear()
                        # items = []
                        # for step in range(len(self.img_metadata_dict['ProcessingSteps'])):
                        #     item = QTreeWidgetItem([str(step + 1), self.img_metadata_dict['ProcessingSteps'][step]['operation']])
                        #     items.append(item)
                            # for key, values in self.img_metadata_dict['ProcessingSteps'][0].items():
                            #     item = QTreeWidgetItem([key, str(values)])
                            #     items.append(item)
                        # self.processing_steps_tree.insertTopLevelItems(0, items)

                        for item in self.img_metadata_dict['ProcessingSteps']:
                            parent_item = QTreeWidgetItem(self.processing_steps_tree, [str(item.get('id', 'No ID')), item.get('operation', 'No Operation')])
                            self.processing_steps_tree.addTopLevelItem(parent_item)
                            self.add_children_tree_widget(parent_item, item)

                    else:
                        self.processing_steps_tree.clear()

                    if "CycleTime" in self.img_metadata_dict:
                        # print(f"getting image: '{self.viewer.layers.selection.active.name}'")
                        self.metadata_tree.clear()
                        # metadata = self.img_metadata_dict
                        items = []
                        for key, values in self.img_metadata_dict.items():
                            item = QTreeWidgetItem([key, str(values)])
                            items.append(item)
                    
                        self.metadata_tree.insertTopLevelItems(0, items)  
                        # Set the scale base on the metadata 
                        # if metadata["CycleTime"]:
                        # self.xscale = self.img_metadata_dict["CycleTime"]
                            # self.plot_widget.axes.set_xlabel("Time (ms)")
                        cycl_time = self.img_metadata_dict["CycleTime"]
                        self.fps_val.setText(f"{round(1/cycl_time, 2)}")

                        # set the current x scale
                        self.x_scale_box.clear()
                        self.x_scale_box.setText(f"{round(cycl_time * 1000, 2)}")
                        self.xscale = cycl_time * 1000
                        # self.main_plot_widget.axes.set_xlabel("Time (ms)")
                    else:
                        self.x_scale_box.clear()
                        self.x_scale_box.setText(f"{1}")
                        self.xscale = 1
                        # self.main_plot_widget.axes.set_xlabel("Frames")
                        self.fps_val.setText("Unknown sampling frequency (fps)")
                    
                elif isinstance(layer, Shapes):
                    self.table_rstl_name.setPlaceholderText(f"{layer.name}")                                   
                    
                else:
                    # Update name of current image name to export
                    self.table_rstl_name.setPlaceholderText("APD_results")
                    self.name_image_to_export.setPlaceholderText("my_image")
                    self.name_image_to_export.setText(None)
                    self.fps_val.setText("")
                    self.metadata_tree.clear()
                    self.processing_steps_tree.clear()
                    # self.x_scale_box.clear()
                    # self.x_scale_box.setText(f"{1}")
                    # self.xscale = 1
        except Exception as e:
            print(CustomException(e, sys))

    def _update_x_scale_box_func(self, event):
        new_x_scale = self.x_scale_box.text()
        if len(new_x_scale) > 0:
            self.xscale = np.float16(new_x_scale)
            self._on_click_plot_profile_btn_func()

    def _get_APD_call_back(self, event):

        # assert that there is a trace in the main plotting canvas
        if len(self.main_plot_widget.figure.axes) > 0 :

            self._APD_plot_widget.figure.clear()
            self._APD_plot_widget.add_single_axes()
            traces = self.data_main_canvas["y"]
            time = self.data_main_canvas["x"]
            rmp_method = self.APD_computing_method.currentText()
            is_interpolated = self.make_interpolation_check.isChecked()
            apd_percentage = self.slider_APD_percentage.value()
            # self.prominence = self.slider_APD_detection_threshold.value() / (self.slider_APD_thres_max_range)
            
            APD_props = []
            # get selection of images iand shape from the selector
            selected_img_list, selected_shps_list = self._get_imgs_and_shapes_items_from_selector(return_layer=True)

            try:
                
                for img_indx, img in enumerate(selected_img_list):

                # for shape_indx, shape in enumerate(selected_shps_list[0].data):
                    for shape_indx in range(len(selected_shps_list[0].data)):
                        if 'ID' in self.shape_layer.features.iloc[shape_indx]:
                            roi_id = self.shape_layer.features.iloc[shape_indx]['ID']
                            if 'position' in self.shape_layer.features.iloc[shape_indx]:
                                roi_id = f"{self.shape_layer.features.iloc[shape_indx]['ID'].split('_')[0]}_{self.shape_layer.features.iloc[shape_indx]['position']}"

                        else:
                            roi_id = f"{shape_indx}"

                        img_label = f"{img.name}_{selected_shps_list[0]}_ROI:{roi_id}"
                        
                        if len(img_label) > 40:
                            img_label = img_label[4:][:15] + "..." + img_label[-9:]
                            # warn("Label name too long to accomodate aesthetics. Truncated to 40 characters")


                        # update detected APs labels
                        # n_peaks = return_peaks_found_fun(promi=self.prominence, np_1Darray=traces[img_indx + shape_indx])
                        # peaks_indx_props = [split_peaks_1d_traces(my_1d_array=traces[img_indx + shape_indx], 
                        #                                           cycle_length_ms = self.xscale,
                        #                                           promi=self.prominence)]
                        n_peaks = return_peaks_found_fun(promi=self.prominence, np_1Darray = traces[img_indx + shape_indx])
                        self.APD_peaks_help_box_label.setText(f'[AP detected]: {n_peaks}')
                        self.APD_peaks_help_box_label_2.setText(f'[AP detected]: {n_peaks}')

                        # self.APD_axes.plot(time, traces[img_indx + shpae_indx], label=f'{lname}_ROI-{shpae_indx}', alpha=0.5)
                        # self._APD_plot_widget.axes.plot(time[img_indx + shape_indx], traces[img_indx + shape_indx], label=f'{img.name}_ROI-{shape_indx}', alpha=0.8)
                        self._APD_plot_widget.axes.plot(time[img_indx + shape_indx], traces[img_indx + shape_indx], label=img_label, alpha=0.8)

                        ##### catch error here and exit nicely for the user with a warning or so #####
                        # try:

                        self.APs_props = compute_APD_props_func(traces[img_indx + shape_indx],
                                                        curr_img_name = img.name, 
                                                        # cycle_length_ms= self.curr_img_metadata["CycleTime"],
                                                        cycle_length_ms= self.xscale,
                                                        rmp_method = rmp_method, 
                                                        apd_perc = apd_percentage, 
                                                        promi=self.prominence, 
                                                        roi_indx=shape_indx, 
                                                        roi_id = roi_id,
                                                        interpolate= is_interpolated,
                                                        curr_file_id = img.metadata["CurrentFileSource"])
                        # collect indexes of AP for plotting AP boudaries: ini, end, baseline
                        ini_indx = self.APs_props["indx_at_AP_resting"]
                        upstroke_indx = self.APs_props["indx_at_AP_upstroke"]
                        peak_indx = self.APs_props["indx_at_AP_peak"]
                        end_indx = self.APs_props["indx_at_AP_end"]

                        y_min = self.APs_props["resting_V"]
                        y_max = traces[img_indx + shape_indx][peak_indx]
                        # plot vline of AP start
                        self._APD_plot_widget.axes.vlines(time[img_indx + shape_indx][ini_indx], 
                                            ymin= y_min,
                                            ymax= y_max,
                                            linestyles='dashed', color = "green", 
                                            # label=f'AP_ini',
                                            lw = 0.5, alpha = 0.8)
                        # plot point at upstroke
                        self._APD_plot_widget.axes.scatter(
                                            time[img_indx + shape_indx][upstroke_indx], 
                                            traces[img_indx + shape_indx][upstroke_indx],
                                            marker="_",
                                            color = "yellow", 
                                            # label=f'AP_upstroke',
                                            lw = 0.5, alpha = 0.5)
                        # plot vline of AP end
                        self._APD_plot_widget.axes.vlines(time[img_indx + shape_indx][end_indx], 
                                            ymin= y_min,
                                            ymax= y_max,
                                            linestyles='dashed', color = "red", 
                                            # label=f'AP_end',
                                            lw = 0.5, alpha = 0.8)
                        # plot hline of AP baseline
                        self._APD_plot_widget.axes.hlines(y_min,
                                            xmin = time[img_indx + shape_indx][ini_indx],
                                            xmax = time[img_indx + shape_indx][end_indx],
                                            linestyles='dashed', color = "grey", 
                                            # label=f'AP_base',
                                            lw = 0.5, alpha = 0.8)

                        # APD_props[f"ImgIndx{img_indx}_ROIIndx{shape_indx}"] = self.APs_props
                        APD_props.append(self.APs_props)
                        
                        print(f"APD computed on image '{img.name}' with roi: {shape_indx}")

                        # except Exception as e:
                        #     # warn(f"ERROR: Computing APD parameters fails witht error: {repr(e)}.")
                        #     raise CustomException(e, sys)

                    self._APD_plot_widget.axes.legend(fontsize="8")
                    self._APD_plot_widget.canvas.draw()

            # try:

                self.APD_props_df = pd.DataFrame( [pro for pro in APD_props]).explode(column = list(APD_props[0].keys())).reset_index(drop=True)
                # convert back to the correct type the numeric columns
                cols_to_keep = ["image_name", "ROI_id", "AP_id" ]
                cols_to_numeric = self.APD_props_df.columns.difference(cols_to_keep)

                self.APD_props_df[cols_to_numeric] = self.APD_props_df[cols_to_numeric].apply(pd.to_numeric, errors = "coerce")
                # convert numeric values to ms and round then
                self.APD_props_df = self.APD_props_df.apply(lambda x: np.round(x, 2) if x.dtypes == "float64" else x ) 

                
                model = PandasModel(self.APD_props_df[["image_name",
                                                "ROI_id", 
                                                "AP_id" ,
                                                "APD_perc" ,
                                                "APD",
                                                "AcTime_dVdtmax",
                                                "BasCycLength_bcl"]])
                    # self.APD_propert_table = QTableView()
                self.APD_propert_table.setModel(model)

                
            except Exception as e:
                # warn(f"ERROR: Computing APD parameters fails witht error: {repr(e)}.")
                raise CustomException(e, sys)
        else:
            return warn("Create a trace first by clicking on 'Display Profile'") 

                

    
    def _clear_APD_plot(self, event):
        """
        Clear the canvas.
        """
        try:            
            self._APD_plot_widget.figure.clear()
            self._APD_plot_widget.canvas.draw()
        except Exception as e:
            print(f">>>>> this is your error: {e}")


        model = PandasModel(self.AP_df_default_val)
        self.APD_propert_table.setModel(model)


    def _get_APD_thre_slider_vlaue_func(self, value):
        value = int(value)
        self.prominence = value / (self.slider_APD_thres_max_range)
        self.slider_APD_detection_threshold.setValue(value)
        self.slider_APD_detection_threshold_2.setValue(value)

        self.slider_label_current_value.setText(f'Sensitivity threshold: {self.prominence}')
        self.slider_label_current_value_2.setText(self.slider_label_current_value.text())
        
        # check that you have content in the graphics panel
        try:
            if len(self.main_plot_widget.figure.axes) > 0 :
                traces = self.data_main_canvas["y"]
                selected_img_list, shapes = self._get_imgs_and_shapes_items_from_selector(return_layer=True)
                for img_indx, img_name in enumerate(selected_img_list):
                    for shpae_indx, shape in enumerate(shapes[0].data):


                            traces[img_indx + shpae_indx]
                            n_peaks = return_peaks_found_fun(promi=self.prominence, np_1Darray=traces[img_indx + shpae_indx])
                            self.APD_peaks_help_box_label.setText(f'[AP detected]: {n_peaks}')
                            self.APD_peaks_help_box_label_2.setText(f'[AP detected]: {n_peaks}')

                    break
        except Exception as e:
            # raise CustomException(e, sys)
            print(CustomException(e, sys))
            # print(f">>>>> this is a known error when computing peaks found while creating shapes interactively: '{e}'")

    
    def _search_and_load_spool_dir_func(self, event=None):
        self.spool_dir = QFileDialog.getExistingDirectory(self, "Select Spool Directory", self.dir_box_text.text())
        self.dir_box_text.setText(self.spool_dir)
        self.load_current_spool_dir()
    
    def _load_current_spool_dir_func(self):
        spool_dir_name =self.dir_box_text.text()
        if os.path.isdir(spool_dir_name):
            self.load_current_spool_dir()
        else:
            print("the selected entry does not seem to be a valid directory")
    
    def _get_imgs_and_shapes_items_from_selector(self, return_layer = False):
        """
        Helper function that return the names of imags and shapes picked in the selector
        """
        if not return_layer:

            img_items = [item.text() for item in self.listImagewidget.selectedItems()]
            shapes_items = [item.text() for item in self.listShapeswidget.selectedItems()]
        elif return_layer:
            img_items = [self.viewer.layers[item.text()] for item in self.listImagewidget.selectedItems()]
            shapes_items = [self.viewer.layers[item.text()] for item in self.listShapeswidget.selectedItems()]

        return img_items, shapes_items
    
    def _get_imgs_and_shapes_items_from_main_layer_list(self, return_layer = False):
        """
        Helper function that return the names of imags and shapes from the main layer list
        """
        if not return_layer:
            img_items = [item.name for item in self.viewer.layers if isinstance(item, Image) ]
            shapes_items = [item.name for item in self.viewer.layers if isinstance(item, Shapes)]
        elif return_layer:
            img_items = [item.name for item in self.viewer.layers if isinstance(item, Image)]
            shapes_items = [item.name for item in self.viewer.layers if isinstance(item, Shapes)]

        return img_items, shapes_items
    
    def _get_imgs2d_from_map_selector(self, return_img = False):
        """
        Helper function that retunr the names of imags and shapes picked in the selector
        """
        if not return_img:
            img_items = [item.text() for item in self.map_imgs_selector.selectedItems()]
            
        elif return_img:
            img_items = [self.viewer.layers[item.text()] for item in self.map_imgs_selector.selectedItems()]
            

        return img_items
    


    
    def load_current_spool_dir(self):
        self.spool_dir =self.dir_box_text.text()
        if os.path.isdir(self.spool_dir):
            data, info = return_spool_img_fun(self.spool_dir)
            self.viewer.add_image(data,
                        colormap = "turbo",
                        name = os.path.basename(self.spool_dir),
                         metadata = info)
        else:
            warn(f"the selected item {self.spool_dir} does not seem to be a valid directory")



    def eventFilter(self, source, event):
        """
        #### NOTE: in order to allow drop events, you must allow to drag!!
        found this solution here: https://stackoverflow.com/questions/25505922/dragdrop-from-qlistwidget-to-qplaintextedit 
        and here:https://www.programcreek.com/python/?CodeExample=drop+event
        """
        if (event.type() == QtCore.QEvent.DragEnter): # and source is self.textedit):
            event.accept()
            # print ('DragEnter')
            return True
        elif (event.type() == QtCore.QEvent.Drop): # and source is self.textedit):
            dir_name = event.mimeData().text().replace("file://", "")
            
            # handel windows path
            if os.name == "nt":
                # print(dir_name)
                # print(Path(dir_name))
                dir_name = Path(dir_name)
                last_part = dir_name.parts[0]
                # this load files hosted locally
                if last_part.startswith("\\"):
                    # print("ozozozozozoz")
                    dir_name = str(dir_name)[1:]
                else:
                    # this load files hosted in servers
                    # print("lllalalalalalalala")
                    dir_name = "//" + str(dir_name)
                    
            # handel Unix path
            elif os.name == "posix":
                dir_name = dir_name[:-1]
            
            else:
                warn(f"your os with value os.name ='{os.name}' has not be normalized for directory paths yet. Please reach out with the package manteiner to discuss this feature.")
            
            dir_name = os.path.normpath(dir_name)  # find a way here to normalize path
            self.dir_box_text.setText(dir_name)
            self.APD_rslts_dir_box_text.setText(dir_name)
            self.save_img_dir_box_text.setText(dir_name)
            # print ('Drop')
            return True
        else:
            return super(OMAAS, self).eventFilter(source, event)
    


    def _on_click_copy_APD_rslts_btn_func(self, event):
        try:
            if hasattr(self, "APD_props_df"):

                if isinstance(self.APD_props_df, pd.DataFrame) and len(self.APD_props_df) > 0:

                    self.APD_props_df.to_clipboard(index=False) 
                    print(">>>>> data copied to clipboard <<<<<<")
                    warn("APD Table copied to clipboard")               

                else:
                    return warn("No data was copied! Make sure you have a APD reulst table and has len > 0")
            else:
                return warn("No data was copied! Make sure you have a APD reulst table.")

                    
        except Exception as e:
            print(f">>>>> this is your error: {e}")


    def _on_click_search_new_dir_APD_rslts_btn_func(self, event):

        self.APD_output_dir = str(QFileDialog.getExistingDirectory(self, "Select Directory", self.APD_rslts_dir_box_text.placeholderText()))
        self.APD_rslts_dir_box_text.setText(self.APD_output_dir)


    def _on_click_save_APD_rslts_btn_func(self, event):
        try:
            if hasattr(self, "APD_props_df"):
                if isinstance(self.APD_props_df, pd.DataFrame) and len(self.APD_props_df) > 0:
                    if not len(self.table_rstl_name.text()) > 0:
                        filename = self.table_rstl_name.placeholderText()
                    else:
                        filename =  self.table_rstl_name.text()
                    if not len(self.APD_rslts_dir_box_text.text()) > 0:
                        output_dir = self.APD_rslts_dir_box_text.placeholderText()
                    else:
                        output_dir = self.APD_rslts_dir_box_text.text()

                    file_format = self.APD_rslts_export_file_format.currentText()

                    if file_format == ".csv":
                        file_path = os.path.join(output_dir, f"{filename}{file_format}")
                        self.APD_props_df.to_csv(file_path, index=False)
                        warn(f"File '{filename}{file_format}' exported to: {file_path}")

                    elif file_format == ".xlsx":
                        file_path = os.path.join(output_dir, f"{filename}{file_format}")
                        self.APD_props_df.to_excel(file_path, index=False)
                        warn(f"File '{filename}{file_format}' exported to: {file_path}")
                else:
                    return warn("No APD results table found or len of the table is < 0.")
            else:
                    return warn("No APD results table found.")

        except Exception as e:
            return print(f">>>>> this is your error: {e}")
    
    def draw(self)-> None:

        self.main_plot_widget.canvas.draw() # you must add this in order tp display the plot

        
    def _layer_list_changed_callback(self, event):
        """Callback function for layer list changes.
        Update the selector model on each layer list change to insert or remove items accordingly,
        while preserving the current selection.
        """
        
        value = event.value
        etype = event.type
        try:

            # handle name change by bypasing the event to the _layer_list_changed_callback
            if etype in ['inserted', 'removed', 'reordered', 'active']:

                image_layers, shape_layers = self._populate_main_ImgShap_selector()

                # Capture the current selected items
                # curr_img_items, curr_shapes_items = self._get_imgs_and_shapes_items_from_selector(return_layer=False)
                curr_img_items_2d = self._get_imgs2d_from_map_selector(return_img=False)
                if isinstance(value, Shapes) or isinstance(value, LayerList):

                    # Update other selectors
                    self.ROI_selection_1.clear()
                    self.ROI_selection_1.addItems(shape_layers)

                    self.ROI_selection_2.clear()
                    self.ROI_selection_2.addItems(shape_layers)

                    self.ROI_selection_crop.clear()
                    self.ROI_selection_crop.addItems(shape_layers)
                    self.ROI_selection_crop.setCurrentIndex(0)

                if isinstance(value, Image) or isinstance(value, LayerList):
                    
                    # Update image selector for cropping
                    self.image_selection_crop.clear()
                    self.image_selection_crop.addItems(image_layers)
                    self.image_selection_crop.setCurrentIndex(0)
                    
                    # Update image selector for maps
                    self.map_imgs_selector.clear()
                    all_images_2d = [layer.name for layer in self.viewer.layers if isinstance(layer, Image) and layer.ndim == 2]
                    for image in all_images_2d:
                        item = QtWidgets.QListWidgetItem(image)
                        self.map_imgs_selector.addItem(item)
                        # Restore the selection if the item was selected before
                        if item.text() in curr_img_items_2d:
                            item.setSelected(True)
                    
                    # Update image selector for Ratio
                    self.Ch0_ratio.clear()
                    self.Ch0_ratio.addItems(image_layers)
                    self.Ch1_ratio.clear()
                    self.Ch1_ratio.addItems(image_layers)

                    # Update image selector for cropping/joining views
                    self.join_imgs_selector.clear()
                    cropped_imgs_list = [imag_name for imag_name in image_layers if "Crop" in imag_name]
                    if len(cropped_imgs_list) > 0:
                        self.join_imgs_selector.addItems(cropped_imgs_list)

                    if len(image_layers) >= 3:
                        n_imgs = len(image_layers)
                        self.Ch0_ratio.setCurrentIndex(n_imgs - 2)
                        self.Ch1_ratio.setCurrentIndex(n_imgs - 1)
                    
                    # Update name of current image name to export
                    # trick for case when is a removing image event
                    if etype == 'removed':
                        self.name_image_to_export.setPlaceholderText("my_image")
                    else:
                        if not isinstance(value, LayerList):
                            self.name_image_to_export.setPlaceholderText(value.name)
                        else:
                            self.name_image_to_export.setPlaceholderText(value[0].name) 
                

                if isinstance(value, Labels) or isinstance(value, LayerList):
                    all_labels = [layer.name for layer in self.viewer.layers if isinstance(layer, Labels)]
                    # Update mask selector for manual segmentation
                    self.mask_list_manual_segment.clear()
                    self.mask_list_manual_segment.addItems(all_labels)

        except Exception as e:
            raise CustomException(e, sys)
    
    def _populate_main_ImgShap_selector(self)-> dict[list[str], list[str]]:
        """
        Populate the main Image and Shapes selector

        Helper function to update and populate the current images and shapes to the main selector.

        Returns
        -------
        dict[list[str], list[str]]
            Retunrs two list containing the names of image layers and shapes layers respectively.
        """

        curr_img_items, curr_shapes_items = self._get_imgs_and_shapes_items_from_selector(return_layer=False)
        self.listImagewidget.clear()
        
        # Clear and update the list
        image_layers = [layer.name for layer in self.viewer.layers if isinstance(layer, Image) and layer.ndim > 2]
        for image in image_layers:
            item = QtWidgets.QListWidgetItem(image)
            self.listImagewidget.addItem(item)
            # Restore the selection if the item was selected before
            if item.text() in curr_img_items:
                item.setSelected(True)
        
        # Clear and update the list
        self.listShapeswidget.clear()
        shape_layers = [layer.name for layer in self.viewer.layers if isinstance(layer, Shapes)]
        
        items = [QtWidgets.QListWidgetItem(shape) for shape in shape_layers]
        for item in items:
            self.listShapeswidget.addItem(item)
            # Restore the selection if the item was selected before
            if item.text() in curr_shapes_items:
                item.setSelected(True)
        
        return (image_layers, shape_layers)


    

    def update_fps(self, fps):
        """Update fps."""
        self.viewer.text_overlay.text = f"Currently rendering at: {fps:1.1f} FPS"
    
    def _on_click_plot_profile_btn_func(self):
        state = self.plot_profile_btn.isChecked()
        
        if state == True:
            # print('Checked')
            img_items, shapes_items = self._get_imgs_and_shapes_items_from_selector(return_layer=False)
            
            if not shapes_items and img_items:
                return warn("Please create and Select a SHAPE from the Shape selector to plot profile")
            if not img_items and shapes_items:
                return warn("Please open and Select an IMAGE from the Image selector to plot profile")
            if not img_items and not shapes_items:
                return warn("Please select a SHAPE & IMAGE from the Shape and Image selectors")
            
            try:
                # img_layer = self.viewer.layers[img_items[0]]
                img_layer = [self.viewer.layers[layer] for layer in img_items]
                self.shape_layer = self.viewer.layers[shapes_items[0]]
                n_shapes = len(self.shape_layer.data)
                if n_shapes == 0:
                    return warn("Draw a new square shape to plot profile in the current selected shape")
                else:
                    self.main_plot_widget.figure.clear()
                    self.main_plot_widget.add_single_axes()
                    # define container for data
                    self.data_main_canvas = {"x": [], "y": []}
                    # take a list of the images that contain "CycleTime" metadata
                    if len(img_layer) > 1:

                        fps_metadata = [image.metadata["CycleTime"] for image in img_layer if "CycleTime" in image.metadata ]
                        imgs_metadata_names = [image.name for image in img_layer if "CycleTime" in image.metadata ]
                        
                        # check that all images contain contain compatible "CycleTime" metadataotherwise trow error
                        if fps_metadata and not (len(img_layer) == len(fps_metadata)):

                            return warn(f"Imcompatible metedata for plotting. Not all images seem to have the same fps metadata as 'CycleTime'. Check that the images have same 'CycleTime'. Current 'CycleTime' values are: {fps_metadata} for images : {imgs_metadata_names}")
                            
                        elif not all(fps == fps_metadata[0] for fps in fps_metadata):

                            return warn(f"Not all images seem to have the same 'CycleTime'. Check that the images have same 'CycleTime'. Current 'CycleTime' values are: {fps_metadata}")
                        else:
                            self.img_metadata_dict = img_layer[0].metadata                        
                    

                    if "CycleTime" in img_layer[0].metadata:
                        if np.allclose(round(img_layer[0].metadata["CycleTime"] * 1000, 2), round(self.xscale, 2), 2):
                            self.main_plot_widget.axes.set_xlabel("Time (ms)")
                        else:
                            self.main_plot_widget.axes.set_xlabel("Frames/custom")
                            
                    else:
                        self.main_plot_widget.axes.set_xlabel("Frames/custom")

                    # loop over images
                    for img in img_layer:
                        # loop over shapes
                        for roi in range(n_shapes):
                            if 'ID' in self.shape_layer.features.iloc[roi]:
                                roi_id = self.shape_layer.features.iloc[roi]['ID']
                            else:
                                roi_id = f"{roi}"

                            img_label = f"{img.name}_{shapes_items[0]}_ROI:{roi_id}"
                            x, y = extract_ROI_time_series(img_layer = img, shape_layer = self.shape_layer, idx_shape = roi, roi_mode="Mean", xscale = self.xscale)
                            if len(img_label) > 40:
                                img_label = img_label[4:][:15] + "..." + img_label[-9:]
                                self.main_plot_widget.axes.plot(x, y, label= img_label)
                                # warn("Label name too long to accomodate aesthetics. Truncated to 40 characters")
                            else:
                                self.main_plot_widget.axes.plot(x, y, label= img_label)

                            self.main_plot_widget.axes.legend()                                
                            self.draw()

                            self.data_main_canvas["x"].append(x)
                            self.data_main_canvas["y"].append(y)
                    # update range for cliping trace
                    max_range = x.size * self.xscale
                    self.double_slider_clip_trace.setRange(0, max_range  )
                    self.double_slider_clip_trace.setValue((max_range * 0.2, max_range * 0.8))

                    self.shape_layer.events.data.connect(self._data_changed_callback)
            except Exception as e:
                print(f"You have the following error @ function _on_click_plot_profile_btn_func: --->> '{e}' <----")
                raise CustomException(e, sys)
        else:
            # print('Unchecked')
            self.main_plot_widget.figure.clear()
            self.draw()
            # reset some variables
            if hasattr(self, "data_main_canvas"):
                del self.data_main_canvas

        
    
    def _data_changed_callback(self, event):

        try:
            # self.prominence = self.slider_APD_detection_threshold.value() / (self.slider_APD_thres_max_range)
            self._get_APD_thre_slider_vlaue_func(value=self.prominence * self.slider_APD_thres_max_range)
            # self._retrieve_metadata_call_back(event)
            state = self.plot_profile_btn.isChecked()
            if state:
                self._on_click_plot_profile_btn_func()
                self.main_plot_widget.canvas.draw()
            else:
                # warn("Please Check on 'Plot profile' to creaate the plot")
                return
        except Exception as e:
            # raise CustomException(e, sys)
            print(CustomException(e, sys))
        
    def _preview_multiples_traces_func(self):

        # assert that there is a trace in the main plotting canvas
        if len(self.main_plot_widget.figure.axes) > 0 :

            # self._data_changed_callback(event)
            self.shape_layer.events.data.connect(self._data_changed_callback)
            # prominence = self.slider_label_current_value / (self.slider_APD_thres_max_range)
            
            traces = self.main_plot_widget.axes.lines[0].get_ydata()
            time = self.main_plot_widget.axes.lines[0].get_xdata()
            label = self.main_plot_widget.figure.axes[0].lines[0].get_label()
            rmp_method = self.APD_computing_method.currentText()
            # img = self.viewer.layers.selection.active
            img_layers, _ = self._get_imgs_and_shapes_items_from_selector(return_layer=True)
            img = img_layers[0]
            is_interpolated = self.make_interpolation_check.isChecked()
            

            try:
            #     self.ini_i_spl_traces, _, self.end_i_spl_traces = return_AP_ini_end_indx_func(my_1d_array = traces, 
            #                                                                         #    cycle_length_ms = self.xscale, 
            #                                                                         promi= self.prominence)
                self.APs_props = compute_APD_props_func(traces,
                                                        curr_img_name = img.name, 
                                                        # cycle_length_ms= self.curr_img_metadata["CycleTime"],
                                                        cycle_length_ms= self.xscale,
                                                        rmp_method = rmp_method, 
                                                        apd_perc = 100, 
                                                        promi=self.prominence, 
                                                        roi_indx=0, 
                                                        # roi_id = roi_id,
                                                        interpolate= is_interpolated,
                                                        curr_file_id = img.metadata["CurrentFileSource"])
                self.ini_i_spl_traces, self.end_i_spl_traces = self.APs_props['indx_at_AP_upstroke'] - min(self.APs_props['indx_at_AP_upstroke']), self.APs_props['indx_at_AP_end'] + min(self.APs_props['indx_at_AP_upstroke'])*2//3
                # self.ini_i_spl_traces, self.end_i_spl_traces = upstroke_indx, end_indx
                
            except Exception as e:
                print(CustomException(e, sys))
                # print(f"You have the following error @ method '_preview_multiples_traces_func' with function: 'return_AP_ini_end_indx_func' : --->> {e} <----")
                return

            self.slider_N_APs.setRange(0, len(self.ini_i_spl_traces) - 1)
            
            # re-create canvas
            self.average_AP_plot_widget.figure.clear()
            self.average_AP_plot_widget.add_single_axes()
            
            if len(self.ini_i_spl_traces) == 1:
                self.average_AP_plot_widget.axes.plot(time, traces, "--", label = f"AP [{0}]_{label}", alpha = 0.8)
                # remove splitted_stack value if exists

                try:
                    if hasattr(self, "splitted_stack"):
                        # del self.splitted_stack
                        self.splitted_stack = traces
                    # else:
                #         raise AttributeError
                except Exception as e:
                    print(CustomException(e, sys))
                

                warn(f"Only one AP detected")
                print(f"{'*'*5} Preview from image: '{self.viewer.layers.selection.active.name}' created {'*'*5}")

            elif len(self.ini_i_spl_traces) > 1:

                # NOTE: need to fix this function
                self.splitted_stack = split_AP_traces_and_ave_func(traces, self.ini_i_spl_traces, self.end_i_spl_traces, type = "1d", return_mean=False)
                new_time_len = self.splitted_stack.shape[-1]
                time = time[:new_time_len]            

                for indx, array in progress(enumerate(self.splitted_stack)):
                    # handle higlighting of selected AP
                    if indx == self.slider_N_APs.value():
                        # if self.shif_trace:

                        #     # NOTE!!!: eveytime the plotting is call it recalculate the peak index, etc and therefore no further shift happen if called multiples time.
                        #     # need to find a way to store the data/canvas and thereafter manipulate/update the figure with the new data
                        #     # functions affected by this behaviour are: _slider_N_APs_changed_func, _remove_mean_check_func, _on_click_mv_left_AP_btn_func and _on_click_mv_right_AP_btn_func
                            
                        #     # duplicate the last value and pad the tail with that
                        #     if self.shift_to_left:
                        #         y = array[-1]
                        #         array = np.concatenate([array[1:], [y]])

                        #         self.average_AP_plot_widget.axes.plot(time[:new_time_len], array, "--", label = f"AP [{indx}]", alpha = 0.8)
                                
                        #         self.splitted_stack[indx] = array
                        #         self.shif_trace = False
                        #         self.shift_to_left = False
                            
                        #     elif  self.shift_to_right:
                        #         y = array[0]
                        #         array = np.concatenate([[y], array[:-1]])

                        #         self.average_AP_plot_widget.axes.plot(time[:new_time_len], array, "--", label = f"AP [{indx}]", alpha = 0.8)
                                
                        #         self.splitted_stack[indx] = array
                        #         self.shif_trace = False
                        #         self.shift_to_right = False
                            
                        # else:
                        self.average_AP_plot_widget.axes.plot(time, array, "--", label = f"AP [{indx}]", alpha = 0.8)
                    else:
                        self.average_AP_plot_widget.axes.plot(time, array, "--", label = f"AP [{indx}]", alpha = 0.2)
                
                # plot the average
                if self.remove_mean_check.isChecked():

                    self.average_AP_plot_widget.axes.plot(time, np.mean(self.splitted_stack, axis = 0), label = f"Mean", c = "b")
             
               
                print(f"{'*'*5} Preview from image: '{self.viewer.layers.selection.active.name}' created {'*'*5}")

            else:
                self._on_click_clear_AP_splitted_btn_fun()
                return warn("No AP detected")
        
            self.average_AP_plot_widget.axes.legend()
            self.average_AP_plot_widget.canvas.draw()             
            print("done")

        else:
            return warn("Create a trace first by clicking on 'Plot Profile'") 


    def _remove_mean_check_func(self):
        # print("lalala")
        self._slider_N_APs_changed_func()
        # if not self.remove_mean_check.isChecked():
        #     self._preview_multiples_traces_func()
        # else:
        #     self._preview_multiples_traces_func()
        # self._preview_multiples_traces_func()


    
    def _on_click_clear_AP_splitted_btn_fun(self):
        # self._remove_attribute_widget()
        self.average_AP_plot_widget.figure.clear()
        self.average_AP_plot_widget.canvas.draw()
    
    # this method can be reomoved
    # def _remove_attribute_widget(self):
    #     my_attr_list = ["slider_N_APs", "slider_N_APs_label"]
    #     for attr in my_attr_list:
    #         if hasattr(self, attr):
    #             for widget in list_of_widgets:
    #                 widget.destroy() 

        
        
        # [self.attr = None for attr in my_attr_list if hasattr(self, attr)]
        # if hasattr(self, "slider_N_APs"):
        #         self.slider_N_APs = None
        #         self.slider_N_APs_label = None

    
    def _on_click_create_average_AP_btn_func(self):
        # traces = self.data_main_canvas["y"][0]
        # time = self.data_main_canvas["x"][0]
        # NOTE: make new logic: fistr check that tupdated data is collected here: self.splitted_stack
        # and then use this info for averaging teh full image stack
        # current_selection = self.viewer.layers.selection.active

        # if isinstance(current_selection, Image):
        #     print(f'computing "local_normal_fun" to image {current_selection}')
        #     results = local_normal_fun(current_selection.data)
        #     self.add_result_img(result_img=results, single_label_sufix="LocNor", add_to_metadata = "Local_norm_signal")
        #     

        
        # assert that you have content in the canvas
        if len(self.average_AP_plot_widget.figure.axes) != 0 and hasattr(self, "data_main_canvas"):

            

            # ini_i, _, end_i = return_AP_ini_end_indx_func(my_1d_array = self.data_main_canvas["y"][0], promi= self.prominence)
            # ini_i, end_i = self.ini_i_spl_traces.tolist(), self.end_i_spl_traces.tolist()
            end_i = self.end_i_spl_traces.tolist() if not isinstance( self.end_i_spl_traces, list ) else self.end_i_spl_traces
            ini_i = self.ini_i_spl_traces.tolist() if not isinstance( self.ini_i_spl_traces, list ) else self.ini_i_spl_traces


            if len(ini_i) > 1:

                img_items, _ = self._get_imgs_and_shapes_items_from_selector(return_layer=True)
                if len(img_items) > 1:
                    return warn("Please select only one image in the image selector")
                current_img_selected = img_items[0]
                params={"prestep": {"method_name": "return_AP_ini_end_indx_func",
                                      "parameters": {"promi": self.prominence}},
                        "ini_index": ini_i,
                        "end_index": end_i}
                
                results= split_AP_traces_and_ave_func(current_img_selected.data, ini_i, end_i, type = "3d", return_mean=True)

                self.add_result_img(result_img=results, 
                                    custom_metadata=current_img_selected.metadata, 
                                    custom_img_name=current_img_selected.name,
                                    operation_name="Average_from_mutiples_APs", 
                                    method_name=split_AP_traces_and_ave_func.__name__,
                                    sufix="AveAP", parameters=params)
                print(f"{'*'*5} Average from image: '{current_img_selected.name,}' created {'*'*5}")
                

            elif len(ini_i) == 1:
                return warn(f"Only {len(ini_i)} AP detected. No average computed.")
            elif len(ini_i) < 1:
                self._on_click_clear_AP_splitted_btn_fun()
                return warn("No AP detected")

        else:
            return warn("Make first a Preview of the APs detected using the 'Preview traces' button.") 


    def _on_click_mv_left_AP_btn_func(self):
        
        if hasattr(self, "data_main_canvas"):
             
            time = self.data_main_canvas["x"][0]
            new_time_len = self.splitted_stack.shape[-1]
            time = time[:new_time_len]
            # self._preview_multiples_traces_func()
            self.average_AP_plot_widget.figure.clear()
            self.average_AP_plot_widget.add_single_axes()

            selected_AP = self.slider_N_APs.value()
            # for the selected AP shift to the left 1 position and append the last value (to match size)
            self.splitted_stack[selected_AP] = np.concatenate([ self.splitted_stack[selected_AP][1:], [self.splitted_stack[selected_AP][-1]] ])
            
            for indx, array in progress(enumerate(self.splitted_stack)):
                # handle higlighting of selected AP
                if indx == self.slider_N_APs.value():
                    self.average_AP_plot_widget.axes.plot(time, array, "--", label = f"AP [{indx}]", alpha = 0.8)
                else:
                    self.average_AP_plot_widget.axes.plot(time, array, "--", label = f"AP [{indx}]", alpha = 0.2)
                
            if self.remove_mean_check.isChecked():
                self.average_AP_plot_widget.axes.plot(time, np.mean(self.splitted_stack, axis = 0), label = f"Mean", c = "b")

            self.average_AP_plot_widget.axes.legend()
            self.average_AP_plot_widget.canvas.draw()

            print("move to left")
        
        else:
            return warn("Make first a Preview of the APs detected using the 'Preview traces' button.") 


    def _on_click_mv_right_AP_btn_func(self):

        if hasattr(self, "data_main_canvas"):

            time = self.data_main_canvas["x"][0]
            new_time_len = self.splitted_stack.shape[-1]
            time = time[:new_time_len]
            # self._preview_multiples_traces_func()
            self.average_AP_plot_widget.figure.clear()
            self.average_AP_plot_widget.add_single_axes()

            selected_AP = self.slider_N_APs.value()
            # for the selected AP shift to the left 1 position and append the last value (to match size)
            self.splitted_stack[selected_AP] = np.concatenate([ [self.splitted_stack[selected_AP][0]], self.splitted_stack[selected_AP][:-1]  ])
            
            for indx, array in progress(enumerate(self.splitted_stack)):
                # handle higlighting of selected AP
                if indx == self.slider_N_APs.value():
                    self.average_AP_plot_widget.axes.plot(time, array, "--", label = f"AP [{indx}]", alpha = 0.8)
                else:
                    self.average_AP_plot_widget.axes.plot(time, array, "--", label = f"AP [{indx}]", alpha = 0.2)
                
            if self.remove_mean_check.isChecked():
                self.average_AP_plot_widget.axes.plot(time, np.mean(self.splitted_stack, axis = 0), label = f"Mean", c = "b")

            self.average_AP_plot_widget.axes.legend()
            self.average_AP_plot_widget.canvas.draw()
            
            print("move to right")
        
        else:
            return warn("Make first a Preview of the APs detected using the 'Preview traces' button.") 
        




    # def show_pop_window_ave_trace(self):
    #     print ("Opening a new popup window...")
    #     self.average_tracce_pop_pup_window = MyPopup(self)
    #     self.average_tracce_pop_pup_window.setGeometry(QRect(100, 100, 400, 200))
    #     self.average_tracce_pop_pup_window.show()
    def _slider_N_APs_changed_func(self):
        if hasattr(self, "data_main_canvas"):

            time = self.data_main_canvas["x"][0]
            new_time_len = self.splitted_stack.shape[-1]
            time = time[:new_time_len]
            # self._preview_multiples_traces_func()
            self.average_AP_plot_widget.figure.clear()
            self.average_AP_plot_widget.add_single_axes()
            
            if self.splitted_stack.ndim > 1:

                for indx, array in progress(enumerate(self.splitted_stack)):
                    # handle higlighting of selected AP
                    if indx == self.slider_N_APs.value():
                        self.average_AP_plot_widget.axes.plot(time, array, "--", label = f"AP [{indx}]", alpha = 0.8)
                    else:
                        self.average_AP_plot_widget.axes.plot(time, array, "--", label = f"AP [{indx}]", alpha = 0.2)
                    
                if self.remove_mean_check.isChecked():
                    self.average_AP_plot_widget.axes.plot(time, np.mean(self.splitted_stack, axis = 0), label = f"Mean", c = "b")
            else:
                array = self.splitted_stack
                self.average_AP_plot_widget.axes.plot(time, array, "--", label = f"AP [{0}]", alpha = 0.8)
                warn("Cannot average from a single AP")

            self.average_AP_plot_widget.axes.legend()
            self.average_AP_plot_widget.canvas.draw()
            print("update plot")

        else:

            return warn("Make first a Preview of the APs detected using the 'Preview traces' button.") 
        # self._preview_multiples_traces_func()


    
    def _on_click_make_maps_btn_func(self):
        # NOTE: you need to decide if you use image form the selector o from the 
        # the napary layer list!!! and assert accordingly the image properties

        try:
            # # assert that a profile was created
            # if hasattr(self, "data_main_canvas"):

            #     time = self.data_main_canvas["x"][0]
            #     _, AP_peaks_indx, _ = return_AP_ini_end_indx_func(self.data_main_canvas["y"][0], promi=self.prominence)
            #     # assert that you have a single AP detected
            #     if len(AP_peaks_indx) == 1:

                    #########################
                    #  start computing maps #
                    #########################

            percentage = self.slider_APD_map_percentage.value()
            # current_img_selection_name = self.listImagewidget.selectedItems()[0].text()
            current_img_selection_name = self.viewer.layers.selection.active.name
            current_img_selection = self.viewer.layers[current_img_selection_name]

            if not isinstance(current_img_selection, Image)  or  current_img_selection.ndim !=3 :
                return warn(f"Select an Image layer with ndim = 3 to apply this function. \nThe selected layer: '{current_img_selection_name}' is of type: '{type(current_img_selection)}' and has ndim = '{current_img_selection.ndim}'")

            # NOTE: 2 states for map type: 0 for Act maps and 2 for APD maps
            map_type = self.toggle_map_type.checkState()
            
            # check for "CycleTime" in metadtata
            if "CycleTime" in self.img_metadata_dict:
                cycl_t = self.img_metadata_dict["CycleTime"]
            else:
                cycl_t = 1
            
            is_interpolated = self.make_interpolation_check.isChecked()

            if map_type == 0:
            
                results = return_maps(current_img_selection.data, 
                                    cycle_time=cycl_t,
                                    map_type = map_type,
                                    percentage = percentage)
                params = {"Activation_map":{"cycle_time": cycl_t,
                        "map_type": map_type,
                        "percentage": percentage}}
                meth_name = return_maps.__name__
                sufix = "ActMap"
                
                # self.add_result_img(result_img=results, 
                #                 img_custom_name=current_img_selection.name, 
                #                 single_label_sufix=f"ActMap_Interp{str(is_interpolated)[0]}", 
                #                 operation_name = f"Activattion Map cycle_time={round(cycl_t, 4)}, interpolate={self.make_interpolation_check.isChecked()}")
                
                
            
            elif map_type == 2:
                image = current_img_selection.data.copy()
                n_frames, y_size, x_size = image.shape
                rmp_method = self.APD_computing_method.currentText()
                apd_percentage = self.slider_APD_percentage.value()

                results = np.zeros((y_size, x_size))
                mask = np.isnan(image[0, ...])
                results[mask] = np.nan

                for y_px  in progress(range(y_size)):
                    for x_px in progress(range(x_size)):
                        if not np.isnan(results[y_px, x_px]).any():
                            try:
                                APs_props = compute_APD_props_func(image[:, y_px, x_px],
                                                                curr_img_name = current_img_selection_name, 
                                                                # cycle_length_ms= self.curr_img_metadata["CycleTime"],
                                                                cycle_length_ms= self.xscale,
                                                                rmp_method = rmp_method, 
                                                                apd_perc = apd_percentage, 
                                                                promi=self.prominence, 
                                                                interpolate = is_interpolated)
                                if not APs_props["APD"]:
                                    print(f"Could not detect APD at pixel coordinate: [{y_px}, {x_px}].")
                                    results[y_px, x_px] = np.nan
                                else:
                                    apd_value = APs_props["APD"]
                                    results[y_px, x_px] = apd_value
                            
                            except Exception as e:
                                results[y_px, x_px] = np.nan
                                # print(CustomException(e, sys))    
                                print(CustomException(e, sys, additional_info=f"error @ pixel [{y_px}, {x_px}]"))
                        # else:
                        #     APD[:, y_px, x_px] = 0
                
                # self.average_AP_plot_widget.axes.plot(time, image[:, y_px, x_px], "-", label = "test", alpha = 0.8)
                # self.average_AP_plot_widget.axes.legend()
                # self.average_AP_plot_widget.canvas.draw()
                np.clip(results, a_min=0, a_max=None, out=results)

                params = {"APD_maps": {"curr_img_name": current_img_selection_name,
                        "cycle_length_ms": self.xscale,
                        "rmp_method" : rmp_method, 
                        "apd_perc" : apd_percentage,
                        "promi":self.prominence,
                        "interpolate" : is_interpolated}}
                meth_name = compute_APD_props_func.__name__
                sufix = f"APDMap{percentage}"
                
                # self.add_result_img(result_img=APD, 
                #                     auto_metadata=False, 
                #                     custom_metadata=current_img_selection.metadata,
                #                     img_custom_name=current_img_selection.name, 
                #                     single_label_sufix=f"APDMap{percentage}_Interp{str(is_interpolated)[0]}", 
                #                     operation_name = f"APD{percentage} Map cycle_time_ms={round(cycl_t, 4)}, promi={self.prominence}, interpolate={self.make_interpolation_check.isChecked()}")

                print("finished")

                # results,  mask_repol_indx_out, t_index_out,  resting_V = return_maps(current_img_selection.data, 
                #                                                                     cycle_time=cycl_t,
                #                                                                     map_type = map_type,
                #                                                                     percentage = percentage)

                # self._preview_multiples_traces_func()
                
                # _, shapes_items = self._get_imgs_and_shapes_items_from_selector(return_img=True)
                # if isinstance(current_img_selection, Image) and len(shapes_items) > 0:
                #     ndim = current_img_selection.ndim
                #     dshape = current_img_selection.data.shape
                #     _, y_px, x_px = np.nonzero(shapes_items[0].to_masks(dshape[-2:]))

                #     if len(y_px) == 1 and len(x_px) == 1:
                #         self.average_AP_plot_widget.axes.axvline(x = t_index_out[y_px, x_px ] * cycl_t, #* 1000, 
                #                                                  linestyle='dashed', 
                #                                                  color = "green", 
                #                                                  label=f'AP_ini',
                #                                                  lw = 0.5, 
                #                                                  alpha = 0.8)
                        
                #         self.average_AP_plot_widget.axes.axvline(x = mask_repol_indx_out[y_px, x_px ] * cycl_t,# * 1000, 
                #                                                  linestyle='dashed', 
                #                                                  color = "red", 
                #                                                  label=f'AP_end',
                #                                                  lw = 0.5, 
                #                                                  alpha = 0.8)
                        
                #         self.average_AP_plot_widget.axes.axhline(y = resting_V,# * 1000, 
                #                                                  linestyle='dashed', 
                #                                                  color = "grey", 
                #                                                  label=f'AP_resting_V',
                #                                                  lw = 0.5, 
                #                                                  alpha = 0.8)
                        
                #         self.average_AP_plot_widget.axes.legend()
                #         self.average_AP_plot_widget.canvas.draw()
                    # else:
                    #     warn(" Not ROI larger than a single pixel. Please reduce the size to plot it")


                
                # self.add_result_img(result_img=APD, 
                #                     auto_metadata=False, 
                #                     custom_metadata=current_img_selection.metadata,
                #                     img_custom_name=current_img_selection.name, 
                #                     single_label_sufix=f"APDMap{percentage}_Interp{str(is_interpolated)[0]}", 
                #                     add_to_metadata = f"APD{percentage} Map cycle_time_ms={round(cycl_t, 4)}, interpolate={self.make_interpolation_check.isChecked()}")
            self.add_result_img(result_img=results, 
                                operation_name="Generate_maps", 
                                method_name=meth_name, 
                                custom_img_name=current_img_selection.name, 
                                custom_metadata=current_img_selection.metadata, 
                                sufix=sufix, parameters=params)


            
            print("Map generated")
            #     else:
            #         return warn("Either non or more than 1 AP detected. Please average your traces, clip 1 AP or make sure you have at least one AP detected by changing the 'Sensitivity threshold'.") 
        
            # else:
            #     return warn("Make first a Preview of the APs detected using the 'Preview traces' button.") 
        except Exception as e:
            raise CustomException(e, sys)



    def _plot_APD_boundaries_btn_func(self):

        img_items, _ = self._get_imgs_and_shapes_items_from_selector(return_layer=True)
        if len(img_items)  < 1 :
            return warn("Select an Image layer from the selector to apply this function.")

        current_img_selection = img_items[0]
        current_img_selection_name = current_img_selection.name
        # current_img_selection = self.viewer.layers[current_img_selection_name]
        
        # if not isinstance(current_img_selection, Image)  or  current_img_selection.ndim !=3 :
        #             return warn(f"Select an Image layer with ndim = 3 to apply this function. \nThe selected layer: '{current_img_selection_name}' is of type: '{type(current_img_selection)}' and has ndim = '{current_img_selection.ndim}'")
        
        self._preview_multiples_traces_func()
        
        # check that you have data in the canvas
        if len(self.average_AP_plot_widget.figure.axes) == 1:
            percentage = self.slider_APD_map_percentage.value()
            # check for "CycleTime" in metadtata
            if "CycleTime" in current_img_selection.metadata:
                cycl_t = current_img_selection.metadata["CycleTime"]
            else:
                cycl_t = 1

            

            map_type = self.toggle_map_type.checkState()

            if map_type == 0:                   
                
                return warn("you re plotting the activation map which is not yet implemented.")
                # results = return_index_for_map(current_img_selection.data, 
                #                       cycle_time=cycl_t,
                #                       map_type = map_type,
                #                       percentage = percentage)
            elif map_type == 2:                               
                
                _, shapes_items = self._get_imgs_and_shapes_items_from_selector(return_layer=True)
                # cropped_img, _, _  = crop_from_shape(shapes_items[0], current_img_selection)

                # results,  mask_repol_indx_out, t_index_out = return_index_for_map(cropped_img, 
                #                                                         cycle_time=cycl_t,
                #                                                         map_type = map_type,
                # APD, mask_repol_indx_out, t_index_out, resting_V = return_maps(cropped_img, 
                #                                                         cycle_time=cycl_t,
                #                                                         map_type = map_type,
                #                                                         percentage = percentage)
                # if cropped_img.squeeze().ndim > 1:
                #     cropped_img = np.mean(cropped_img, axis = (1, 2))
                # else:
                #     cropped_img = cropped_img.squeeze()

                for shape_indx, _ in enumerate(shapes_items[0].data):
                    try:
                        rmp_method = self.APD_computing_method.currentText()
                        apd_percentage = self.slider_APD_percentage.value()
                        traces = self.average_AP_plot_widget.axes.lines[shape_indx].get_ydata()
                        time = self.average_AP_plot_widget.axes.lines[shape_indx].get_xdata()

                        APs_props = compute_APD_props_func(traces,
                                                            curr_img_name = current_img_selection_name, 
                                                            # cycle_length_ms= self.curr_img_metadata["CycleTime"],
                                                            cycle_length_ms= self.xscale,
                                                            rmp_method = rmp_method, 
                                                            apd_perc = apd_percentage, 
                                                            promi=self.prominence, 
                                                            roi_indx=shape_indx)
                        # collect indexes of AP for plotting AP boudaries: ini, end, baseline
                        ini_indx = APs_props[-3]
                        peak_indx = APs_props[-2]
                        end_indx = APs_props[-1]
                        dVdtmax = APs_props[5]
                        resting_V = APs_props[8]
                        y_min = resting_V

                        y_max = traces[peak_indx]
                        # plot vline of AP start
                        self.average_AP_plot_widget.axes.vlines(time[ini_indx], 
                                            ymin= y_min,
                                            ymax= y_max,
                                            linestyles='dashed', color = "green", 
                                            label=f'AP_ini',
                                            lw = 0.5, alpha = 0.8)
                        # plot vline of AP end
                        self.average_AP_plot_widget.axes.vlines(time[end_indx], 
                                            ymin= y_min,
                                            ymax= y_max,
                                            linestyles='dashed', color = "red", 
                                            label=f'AP_end',
                                            lw = 0.5, alpha = 0.8)
                        # plot hline of AP baseline
                        self.average_AP_plot_widget.axes.hlines(resting_V,
                                            xmin = time[ini_indx],
                                            xmax = time[end_indx],
                                            linestyles='dashed', color = "grey", 
                                            label=f'AP_resting',
                                            lw = 0.5, alpha = 0.8)

                        # APD_props.append(self.APs_props)
                        self.average_AP_plot_widget.axes.legend(fontsize="8")
                        self.average_AP_plot_widget.canvas.draw()
                        
                        print(f"APD computed on image '{current_img_selection_name}' with roi: {shape_indx}")

                    except Exception as e:
                        print(f">>>>> this is your error @ method: '_plot_APD_boundaries_btn_func': {e}")
                    # # collect indexes of AP for plotting AP boudaries: ini, end, baseline
                    # ini_indx = self.APs_props[-3]
                    # peak_indx = self.APs_props[-2]
                    # end_indx = self.APs_props[-1]
                    # dVdtmax = self.APs_props[5]
                    # resting_V = self.APs_props[8]
                    # y_min = resting_V

                    # y_max = traces[img_indx + shape_indx][peak_indx]
                
                

                # plot boundaries in indexes found
                # dshape = cropped_img.shape
                # _, y_px, x_px = np.nonzero(shapes_items[0].to_masks(dshape[-2:]))
                # if len(y_px) == 0:
                #     y_px = 0
                # if len(x_px) == 0:
                #     x_px = 0
                # self.average_AP_plot_widget.axes.axvline(x = t_index_out[y_px, x_px ] * cycl_t, #* 1000, 
                #                                                      linestyle='dashed', 
                #                                                      color = "green", 
                #                                                      label=f'AP_ini',
                #                                                      lw = 0.5, 
                #                                                      alpha = 0.8)
                            
                # self.average_AP_plot_widget.axes.axvline(x = mask_repol_indx_out[y_px, x_px ] * cycl_t,# * 1000, 
                #                                             linestyle='dashed', 
                #                                             color = "red", 
                #                                             label=f'AP_end',
                #                                             lw = 0.5, 
                #                                             alpha = 0.8)
                
                # self.average_AP_plot_widget.axes.axhline(y = resting_V,# * 1000, 
                #                                             linestyle='dashed', 
                #                                             color = "grey", 
                #                                             label=f'AP_resting_V',
                #                                             lw = 0.5, 
                #                                             alpha = 0.8)
                
                
            
            
            # traces = self.average_AP_plot_widget.axes.lines[0].get_ydata()
            # time = self.average_AP_plot_widget.axes.lines[0].get_xdata()

        print("plotting AP boundaries")



    def _on_click_average_roi_on_map_btn_fun(self):
        
        _, shapes_items = self._get_imgs_and_shapes_items_from_selector(return_layer=True)
        current_img_selected = self.viewer.layers.selection.active
        
        try:
            
            if isinstance(current_img_selected, Image):
                if current_img_selected.ndim == 2:
                    if len(shapes_items) > 0:
                        ndim = current_img_selected.ndim
                        dshape = current_img_selected.data.shape
                        masks = shapes_items[0].to_masks(dshape)
                        
                        if masks.ndim == 3:
                            for indx_roi, roi in enumerate(masks):
                                
                                results = np.nanmean(current_img_selected.data[roi])
                                # self.average_roi_value_container.
                                print(f"mean of roi: '{indx_roi}' in '{shapes_items[0].name}' for '{current_img_selected.name}' is: \n{round(results, 2)}")
                                self.average_roi_value_container.setText(f"mean of roi: '{indx_roi}' in '{shapes_items[0].name}' for '{current_img_selected.name}' is: \n{round(results, 2)}")
                            return            
                        
                        elif masks.ndim == 2:
                            
                            results = np.nanmean(current_img_selected.data[masks])
                            self.average_roi_value_container.setText(f"mean of roi: '{indx_roi}' in '{shapes_items[0].name}' for '{current_img_selected.name}' is: \n{round(results, 2)}")
                            return print(f"mean of roi in shape '{shapes_items[0].name}' for image '{current_img_selected.name}' is: \n{round(results, 2)}")
                        else:
                            self.average_roi_value_container.setText("")
                            # self.average_roi_value_container.setPlaceholderText("select a ROI and click the 'Get current ROI mean' button.")
                            warn(f"You seem to have an issue with your shapes")
                    else:
                        return warn(f"Select a shape from the shape selector")
                else:
                    return warn(f"Select an 2d image from a map. Current image: '{current_img_selected.name}' has ndim = {ndim}.")
                
            else:
                self.average_roi_value_container.setText("")
                # self.average_roi_value_container.setPlaceholderText("select a ROI and click the 'Get current ROI mean' button.")
                return warn("No Image layer selected. Select an 2d image retunred from a map.")
        except Exception as e:
            raise CustomException(e, sys)



# def extract_ROI_time_series(img_layer, shape_layer, idx_shape, roi_mode, xscale = 1):
#     """Extract the array element values inside a ROI along the first axis of a napari viewer layer.

#     :param current_step: napari viewer current step
#     :param layer: a napari image layer
#     :param labels: 2D label array derived from a shapes layer (Shapes.to_labels())
#     :param idx_shape: the index value for a given shape
#     :param roi_mode: defines how to handle the values inside of the ROI -> calc mean (default), median, sum or std
#     :return: shape index, ROI mean time series
#     :rtype: np.ndarray
#     """

#     ndim = img_layer.ndim
#     dshape = img_layer.data.shape

#     mode_dict = dict(
#         Min=np.min,
#         Max=np.max,
#         Mean=np.mean,
#         Median=np.median,
#         Sum=np.sum,
#         Std=np.std,
#     )
#     # convert ROI label to mask
#     if ndim == 3:    
#         label = shape_layer.to_labels(dshape[-2:])
#         mask = np.tile(label == (idx_shape + 1), (dshape[0], 1, 1))

#     if mask.any():
#         return add_index_dim(mode_dict[roi_mode](img_layer.data[mask].reshape(dshape[0], -1), axis=1), xscale)
    





    
    def _on_click_create_AP_gradient_bt_func(self):
        current_img_selected = self.viewer.layers.selection.active
        
        if isinstance(current_img_selected, Image):

            if current_img_selected.ndim == 3:
                results = np.gradient(current_img_selected.data, axis=0)

                print(f"Computing Gradient on image: {current_img_selected.name}")
                self.add_result_img(result_img=results, 
                                    custom_img_name=current_img_selected.name,
                                    custom_metadata=current_img_selected.metadata,
                                    operation_name="AP Gradient", 
                                    method_name=np.gradient.__name__, 
                                    sufix="ActTime", 
                                    parameters={"axis": 0})
            else: 
                return warn(f"The image selected: '{current_img_selected.name}'  has ndim = {current_img_selected.ndim }. \nPlease select an image with ndim = 3 to compute Gradient.")
        
        return warn(f"No an image selected or image. Please select an image layer.")
    


        

    def _on_click_apply_segmentation_btn_fun(self, return_result_as_layer = True, return_mask = False):
        current_selection = self.viewer.layers.selection.active
        if isinstance(current_selection, Image):

            try:
        
                segmentation_method_selected = self.segmentation_methods.currentText()
                segmentation_methods = [self.segmentation_methods.itemText(i) for i in range(self.segmentation_methods.count())]
                is_mask_inverted = self.is_inverted_mask.isChecked()
                is_return_image = self.return_img_no_backg_btn.isChecked()
                
                # Handeling size ndim of data (2d or 3d allow only)
                if current_selection.ndim == 3:
                    # data = current_selection.data.mean(axis = 0)
                    array2d_for_mask = current_selection.data.max(axis = 0)
                elif current_selection.ndim == 2:
                    array2d_for_mask = current_selection.data
                else : 
                    raise ValueError(f"Not implemented segemntation for Image with dimensions = {current_selection.ndim}.")
                
                if segmentation_method_selected == segmentation_methods[0]:
                    try:
                        mask = segment_image_triangle(array2d_for_mask)
                        mask = polish_mask(mask)
                        meth_name = segment_image_triangle.__name__ 
                        params = {"Segmentation_mode": "Auto",
                                  "Segmentation_method": f"{segmentation_method_selected}",
                                  "postprocessing_step":{"method":{polish_mask.__name__:
                                                              {"parameters":"default"}}}}
                            # {"postprocessing_step":{"method":{polish_mask.__name__:
                            #                                   {"parameters":"default"}}}}}
                        print(f"{'*'*5} Aplying 'Auto' segmentation with method: '{segmentation_method_selected}' to image: '{current_selection}' {'*'*5}")

                    except Exception as e:
                        raise CustomException(e, sys)

                    
                elif segmentation_method_selected == segmentation_methods[1]:
                    try:
                        mask, threshold = segment_image_GHT(array2d_for_mask, return_threshold=True)
                        mask = polish_mask(mask)
                        meth_name = segment_image_GHT.__name__ 
                        params = {"Segmentation_mode": "Auto",
                                  "Segmentation_method": f"{segmentation_method_selected}",
                                  "postprocessing_step":{"method":{polish_mask.__name__:
                                                              {"parameters":"default"}}}}

                        print(f"{'*'*5} Aplying 'Auto' segmentation with method: '{segmentation_method_selected}' to image: '{current_selection}' {'*'*5}")

                    except Exception as e:
                        raise CustomException(e, sys)
                
                
                elif segmentation_method_selected == segmentation_methods[2]:
                    # take fisrt frame and use it for segementation
                    try:   

                        lo_t = float(self.low_threshold_segmment_value.text())
                        hi_t = float(self.high_threshold_segment_value.text())
                        params = {"Segmentation_mode": "Auto",
                                    "Segmentation_method": f"{segmentation_method_selected}",
                                    "parameters":{"lo_t":lo_t, 
                                                "hi_t": hi_t}}
                        meth_name = segement_region_based_func.__name__ 

                        if self.is_Expand_mask.isChecked():
                            expand = int(self.n_pixels_expand.currentText())
                            # using maximum pixels intetnsity as reference                            
                            mask = segement_region_based_func(array2d_for_mask, lo_t = lo_t, hi_t = hi_t, expand = expand)
                            params["parameters"]["expand"] = expand                            
                            
                        else:
                            # using maximum pixels intetnsity as reference
                            mask = segement_region_based_func(array2d_for_mask, lo_t = lo_t, hi_t = hi_t, expand = None)                            
                            params["parameters"]["expand"] = None
                            
                        print(f"{'*'*5} Aplying 'Auto' segmentation with method: '{segmentation_method_selected}' to image: '{current_selection}' {'*'*5}")
                    except Exception as e:
                        raise CustomException(e, sys)

                    
                else:
                    return warn( f"selected filter '{segmentation_method_selected}' no known.")
                
                
                if is_mask_inverted:
                    mask = np.invert(mask.astype(bool))
                params["inverted_mask"]= is_mask_inverted

                if return_mask:
                    return mask
                
                if return_result_as_layer:
                    self.add_result_label(mask, 
                                            img_custom_name="Heart_labels", 
                                            single_label_sufix = f"NullBckgrnd", 
                                            add_to_metadata = f"Background image masked")
                
                if is_return_image:
                    params["return_image"] = is_return_image
                    # 8. remove background using mask
                    n_frames =current_selection.data.shape[0]
                    masked_image = current_selection.data.copy()

                    if masked_image.ndim == 3:
                        
                        try:

                            if np.issubdtype(masked_image.dtype, np.integer):
                                masked_image[~np.tile(mask.astype(bool), (n_frames, 1, 1))] = 0

                            else:
                                masked_image[~np.tile(mask.astype(bool), (n_frames, 1, 1))] = None
                        except Exception as e:
                                raise CustomException(e, sys)
                    else:

                        try:
                            if np.issubdtype(masked_image.dtype, np.integer):
                                masked_image[~mask.astype(bool)] = 0
                            else:
                                masked_image[~mask.astype(bool)] = None

                        except Exception as e:
                                raise CustomException(e, sys)




                    # 9. subtract bacground from original image 
                    # background = np.nanmean(masked_image)                    
                    # masked_image = masked_image - background

                    self.add_result_img(result_img=masked_image, operation_name="Image_segmentation", 
                                        sufix=f"{params['Segmentation_mode'][:3]}Segm{segmentation_method_selected[:3].capitalize()}", 
                                        custom_outputs=[current_selection.name + f"_Segm{segmentation_method_selected[:3].capitalize()}", "Heart_labels"],
                                        method_name=meth_name, 
                                        custom_img_name=current_selection.name, parameters=params)
                    
                        
                
                
            
            except Exception as e:
                raise CustomException(e, sys)
                # print(CustomException(e, sys))

        else:
            warn(f"Select an Image layer to apply this function.")
    


    def _on_click_segment_manual_btn_func(self):

        current_selection = self.viewer.layers.selection.active
        if isinstance(current_selection, Image):

        
            # current_selection = self.viewer.layers[self.img_list_manual_segment.currentText()]
            mask_layer = self.viewer.layers[self.mask_list_manual_segment.currentText()]
            current_timpe_point = self.viewer.dims.current_step[0]
            n_frames = current_selection.data.shape[0]
            is_mask_inverted = self.is_inverted_mask.isChecked()

            if mask_layer.data.ndim == 3:
                mask = mask_layer.data[current_timpe_point, ...] > 0
            elif mask_layer.data.ndim == 2:
                mask = mask_layer.data > 0
            else:
                raise ValueError(" Not implemented yet how to handle mask of ndim = {mask_layer.data.ndim}. Please report this or file an issue via github")

            masked_image = current_selection.data.copy()

            if is_mask_inverted:
                mask = np.invert(mask.astype(bool))
            
            params = {"Segmentation_mode": "Manual"}
            params["inverted_mask"]= is_mask_inverted

            try:

                if np.issubdtype(masked_image.dtype, np.integer):
                    masked_image[~np.tile(mask.astype(bool), (n_frames, 1, 1))] = 0
                elif np.issubdtype(masked_image.dtype, np.inexact):
                    masked_image[~np.tile(mask.astype(bool), (n_frames, 1, 1))] = None
                else:
                        masked_image[~np.tile(mask.astype(bool), (n_frames, 1, 1))] = None
            except Exception as e:
                raise CustomException(e, sys)

            self.add_result_img(result_img=masked_image, operation_name="Image_segmentation", 
                                sufix=f"{params['Segmentation_mode'][:3]}Segm", 
                                custom_inputs=[current_selection.name , mask_layer.name],
                                method_name=None, 
                                custom_img_name=current_selection.name, parameters=params)

            print(f"{'*'*5} Aplying 'Manual' segmentation to image: '{current_selection}' {'*'*5}")

        else:
            warn(f"Select an Image layer to apply this function. \nThe selected layer: '{current_selection}' is of type: '{current_selection._type_string}'")
    
    
    
    def _on_click_plot_histogram_btn_func(self):

        layer = self.viewer.layers.selection.active
        # NOTE: assert here that layer is an image and you have layers
        if isinstance(layer, Image):

            self.histogram_plot_widget.figure.clear()
            self.histogram_plot_widget.add_single_axes()
            n_bins = self.slider_histogram_bins.value()
            _COLORS = {"r": "tab:red", "g": "tab:green", "b": "tab:blue"}
            
            if not self.toggle_hist_data.isChecked():
                time_point = self.viewer.dims.current_step[0]
                # layer, _ = self._get_imgs_and_shpes_items(return_img=True)
                # layer = layer[0]

                if layer.data.ndim - layer.rgb == 3:
                    # 3D data, can be single channel or RGB
                    data = layer.data[time_point]
                    self.histogram_plot_widget.axes.set_title(f"z={time_point}")
                else:
                    data = layer.data
                # Read data into memory if it's a dask array
                data = np.asarray(data)

                # Important to calculate bins after slicing 3D data, to avoid reading
                # whole cube into memory.
                # bins = np.linspace(np.min(data), np.max(data), n_bins)

                if layer.rgb:
                    # Histogram RGB channels independently
                    for i, c in enumerate("rgb"):
                        self.histogram_plot_widget.axes.hist(
                            data[..., i].ravel(),
                            bins=n_bins,
                            label=c,
                            # histtype="step",
                            edgecolor='white',
                            # linewidth=1.2,
                            color=_COLORS[c],
                        )
                else:
                    self.histogram_plot_widget.axes.hist(data.ravel(), 
                                                        bins=n_bins, 
                                                        #  histtype="step",
                                                        edgecolor='white',
                                                        #  linewidth=1.2,
                                                        label=layer.name)

                self.histogram_plot_widget.axes.legend()
                print(f"{'*'*5} Histogram of image: '{layer.name}' at frame: '{time_point}'.")

            else:
                data = layer.data
                # bins = np.linspace(np.min(data), np.max(data), n_bins)
                self.histogram_plot_widget.axes.hist(data.ravel(), 
                                                        bins=n_bins, 
                                                        #  histtype="step",
                                                        edgecolor='white',
                                                        #  linewidth=1.2,
                                                        label=layer.name)
                
                print(f"{'*'*5} Histogram of image: '{layer.name}'. Full stack ({data.shape[0]} frames) created {'*'*5} ")
        else:
            return warn(f"Select an Image layer to display histogrma. \nThe selected layer: '{layer}' is of type: '{layer._type_string}'")



        
        self.histogram_plot_widget.canvas.draw()
    
    def _on_click_clear_histogram_btn_func(self):
        self.histogram_plot_widget.figure.clear()
        self.histogram_plot_widget.canvas.draw()
        print("Clearing Histogram plot")
    

    def _on_click_clip_trace_btn_func(self):

        # self.main_plot_widget.figure.clear()
        # self.main_plot_widget.add_single_axes()

        start_indx, end_indx = self.double_slider_clip_trace.value()
        start_indx = int(start_indx / self.xscale)
        end_indx = int(end_indx / self.xscale)
        params = {"start_indx": start_indx,
                  "end_indx": end_indx}
        # assert that there is a trace in the main plotting canvas
        if len(self.main_plot_widget.figure.axes) > 0 :
            
            if self.is_range_clicked_checkbox.isChecked():
                
                time = self.data_main_canvas["x"]
                selected_img_list, _ = self._get_imgs_and_shapes_items_from_selector(return_layer=True)
                for image in selected_img_list:
                    results = image.data[start_indx:end_indx]
                    self.add_result_img(result_img=results, 
                                        custom_metadata=image.metadata,
                                        custom_img_name=image.name,
                                        operation_name= "clip_image",
                                        method_name="indexing", 
                                        sufix="Clip", parameters=params)
                    # 
                    self.is_range_clicked_checkbox.setChecked(False)
                    self.plot_profile_btn.setChecked(False)
                    self.listImagewidget.clearSelection()
                    new_img_indx = len([self.listImagewidget.item(n).text() for n in range(self.listImagewidget.count())]) -1
                    self.listImagewidget.item(new_img_indx).setSelected(True)
                    self.plot_profile_btn.setChecked(True)
                    # self.plot_last_generated_img()
                    
                    print(f"{'*'*5} Clipping from time index: [{start_indx}:{end_indx}] to image: '{image.name}'. {'*'*5}")
            else:
                return warn("Preview the clipping range firts by ticking the 'Show range'.")
        else:
            return warn("Create a trace first by clicking on 'Display Profile'") 
    
    def _dsiplay_range_func(self):
        state = self.is_range_clicked_checkbox.isChecked()
        
        if state == True and self.plot_profile_btn.isChecked() :
            start_indx, end_indx = self.double_slider_clip_trace.value()
            # assert that there is a trace in the main plotting canvas
            if len(self.main_plot_widget.figure.axes) > 0 :
                time = self.data_main_canvas["x"]
                selected_img_list, _ = self._get_imgs_and_shapes_items_from_selector(return_layer=True)
                self.main_plot_widget.axes.axvline(start_indx, c = "silver", linestyle = 'dashed', linewidth = 1)
                self.main_plot_widget.axes.axvline(end_indx, c = "silver", linestyle = 'dashed', linewidth = 1)
                self.main_plot_widget.canvas.draw()
            else:
                return warn("Create a trace first by clicking on 'Display Profile'") 
        else:
            if len(self.main_plot_widget.figure.axes) > 0 and len(self.main_plot_widget.figure.axes[0].lines) > 1:
                for i in range(2):
                    self.main_plot_widget.figure.axes[0].lines[-1].remove()
            self.main_plot_widget.canvas.draw()

            return
    
    def _double_slider_clip_trace_func(self):
        state = self.is_range_clicked_checkbox.isChecked()

        try:
        
            if state == True and self.plot_profile_btn.isChecked():
                # self._dsiplay_range_func()
                n_lines = len(self.main_plot_widget.figure.axes[0].lines)
                if n_lines == 3:
                    for i in range(2):
                        self.main_plot_widget.figure.axes[0].lines[-1].remove()
                else:
                    self.main_plot_widget.figure.axes[0].lines[-1].remove()
                
                self._dsiplay_range_func()


            else:
                return
        except Exception as e:
            raise CustomException(e, sys)
    
    def _export_processing_steps_btn_func(self):
        
        current_selection = self.viewer.layers.selection.active
        
        if isinstance(current_selection, Image):

            try:

                metadata = current_selection.metadata
                key = "ProcessingSteps"

                if key in metadata.keys():

                    dirname = self.save_img_dir_box_text.text()
                    dirname = dirname if len(dirname) != 0 else self.save_img_dir_box_text.placeholderText()

                    fileName = self.procsteps_file_name.text()
                    fileName = fileName if len(fileName) != 0 else self.procsteps_file_name.placeholderText()

                    fileName = os.path.join(dirname, fileName + ".yml") 

                    self.metadata_recording_steps.steps = metadata[key] if key in metadata else []
                    self.metadata_recording_steps.save_to_yaml(fileName)
                            
                    print(f"{'*'*5} Exporting processing steps from image: '{current_selection.name}' as '{os.path.basename(fileName)}' to folder '{dirname}' {'*'*5}")
                else:
                    return warn("No 'Preprocessing' steps detected.")
            except Exception as e:
                raise CustomException(e, sys)
        else:
            return warn("Please select an image leyer.")
    
    def change_dir_to_save_img_btn_func(self):

        current_selection = self.viewer.layers.selection.active
        
        if isinstance(current_selection, Image):
            try:
                options = QFileDialog.Options()
                # options |= QFileDialog.DontUseNativeDialog                        
                dir_name = QFileDialog.getExistingDirectory(self, "Select Directory")
                dir_name = os.path.normpath(dir_name)
                self.save_img_dir_box_text.setText(dir_name)
                print(dir_name)
            except Exception as e:
                raise CustomException(e, sys)


    

    def _export_image_btn_func(self):

        current_selection = self.viewer.layers.selection.active
        
        if isinstance(current_selection, Image):
            
            try:
                dirname = self.save_img_dir_box_text.text()
                dirname = dirname if len(dirname) != 0 else self.save_img_dir_box_text.placeholderText()
                
                fileName = self.name_image_to_export.text()
                fileName = fileName if len(fileName) != 0 else self.name_image_to_export.placeholderText()
                
                fileName = os.path.join(dirname, fileName + ".tif") # here you can eventually to change 

                metadata = convert_to_json_serializable(current_selection.metadata)
                self.metadata_recording_steps.save_to_tiff(
                    current_selection.data, 
                    metadata, 
                    fileName
                    )
                
                print(f"{'*'*5} Exporting image: '{os.path.basename(fileName)}' to folder '{dirname}' {'*'*5}")
            except Exception as e:
                raise CustomException(e, sys)
        else:
            return warn("Please select an image leyer.")
    

    def _apply_optimap_mot_corr_btn_func(self):
        
        current_selection = self.viewer.layers.selection.active
        
        if isinstance(current_selection, Image) and current_selection.ndim == 3:
            c_k = self.c_kernels.value()
            pre_smooth_t=self.pre_smooth_temp.value()
            pre_smooth_s=self.pre_smooth_spat.value()
            ref_frame_indx = int(self.ref_frame_val.text()) # put this in the GUI

            params = {"Contrast_Kernel": c_k,
                      "pre_smooth_temporal" : pre_smooth_t,
                      "pre_smooth_sapatial" : pre_smooth_s,
                      "reference_frame" : ref_frame_indx}

            print("running motion stabilization")
            results = optimap_mot_correction(current_selection.data, 
                                             c_k = c_k,
                                             pre_smooth_t= pre_smooth_t,
                                             proe_smooth_s= pre_smooth_s, 
                                             ref_fr=ref_frame_indx)
            
            self.add_result_img(result_img=results, 
                                operation_name="Motion_correction", 
                                method_name="optimap_mot_correction", 
                                sufix="MotStab", 
                                parameters=params)
            
            

        else:
            
            return warn(f"No an image selected or image: '{current_selection.name}' has ndim = {current_selection.ndim }. Select an temporal 3d image stack")
        
    
    def _on_click_crop_from_shape_btn_func(self):

        try:
            shape_name = self.ROI_selection_crop.currentText()
            images_layers, shapes_layers = self._get_imgs_and_shapes_items_from_main_layer_list(return_layer=False)
            if shape_name not in shapes_layers:
                self.rotate_l_crop.setChecked(False)
                self.rotate_r_crop.setChecked(False)
                return warn("No source Shape layer found for cropping operation")
            shape_layer = self.viewer.layers[shape_name]

            img_name = self.image_selection_crop.currentText()
            if img_name not in images_layers:
                self.rotate_l_crop.setChecked(False)
                self.rotate_r_crop.setChecked(False)
                return warn("No source Image layer found for cropping operation")
            img_layer = self.viewer.layers[img_name]
            metadata = img_layer.metadata

            if len(shape_layer.data) == 0:
                return warn("Selected shape for cropping is empty. Please draw a square ROI to use for cropping operation.")

            for shape in shape_layer.data:
                cropped_img, ini_index, end_index = crop_from_shape(shape, img_layer)

                a, b, c, d = shape
                param = {
                    "from_shape": {"name": shape_name,
                                "data": {"t_right" : a.tolist(),
                                            "t_left" : b.tolist(),
                                            "b_left" : c.tolist(),
                                            "b_right" : d.tolist()}
                                            },
                    "crop_indexes": {"y": {"ini_index":int(ini_index[0]),
                                        "end_index": int(end_index[0])},
                                    "x": {"ini_index":int(ini_index[1]),
                                                "end_index": int(end_index[1])}}
                                            
                    }

                if self.rotate_l_crop.isChecked():
                    cropped_img = np.rot90(cropped_img, axes=(1, 2))
                    print(f"result image rotate 90° to the left")
                    param["rotate_image"] = {"method_name" : "np.rot90", "axes": [1, 2]}
                elif self.rotate_r_crop.isChecked():
                    cropped_img = np.rot90(cropped_img, axes=(2, 1))
                    param["rotate_image"] = {"method_name" : "np.rot90", "axes": [2, 1]}
                    print(f"result image rotate 90° to the right")

                self.add_result_img(result_img=cropped_img, 
                                    operation_name="Crop_image", 
                                    custom_img_name=img_name, 
                                    method_name="crop_from_shape", 
                                    custom_metadata= metadata, 
                                    sufix="Crop", parameters=param)
            
            self.rotate_l_crop.setChecked(False)
            self.rotate_r_crop.setChecked(False)
            
            print(f"image '{img_name}' cropped")
            # return


        except Exception as e:
            # raise CustomException(e, sys)
            self.rotate_l_crop.setChecked(False)
            self.rotate_r_crop.setChecked(False)
            print(CustomException(e, sys))


    def _update_APD_value_for_MAP_func(self):
        new_value = self.slider_APD_percentage.value()
        self.slider_APD_map_percentage.setValue(new_value)

    def _update_APD_value_for_APD_func(self):
        new_value = self.slider_APD_map_percentage.value()
        self.slider_APD_percentage.setValue(new_value)
    
    def plot_last_generated_img(self, shape_indx = 0):
        """
        easy helper to change selections and plot last image generated
        """
        self.listShapeswidget.clearSelection()
        self.listShapeswidget.item(shape_indx).setSelected(True)
        
        self.listImagewidget.clearSelection()
        last_img_indx = self.listImagewidget.count() -1 if self.listImagewidget.count() > 0 else None 
        self.listImagewidget.item(last_img_indx).setSelected(True)
        if self.plot_profile_btn.isChecked():
            self.plot_profile_btn.click()
            self.plot_profile_btn.click()
        else:
            self.plot_profile_btn.click()
    
    def _plot_curr_map_btn_fun(self):

        # selectedItems = self.map_imgs_selector.lineEdit().text().split(", ")
        try:
            # selectedItems = [x.strip() for x in self.map_imgs_selector.lineEdit().text().split(',')]
            selectedItems = self._get_imgs2d_from_map_selector(return_img=False)
            # selectedItems = [selectedItems] if isinstance(selectedItems, str) else selectedItems
            # current_selection = self.viewer.layers.selection.active
            self.maps_plot_widget.figure.clear()
            self.maps_plot_widget.add_single_axes()

            if len(selectedItems) == 1 and len(selectedItems[0]) > 0:
                current_selection = self.viewer.layers[selectedItems[0]]
                self.map_data = current_selection.data

            elif len(selectedItems) > 1:
                current_selection = [self.viewer.layers[item].data for item in selectedItems]
                self.map_data = concatenate_and_padd_with_nan_2d_arrays(current_selection)
            
            else:
                return warn(f"No image selected. Please select an Image from the selector")
                    
                # print("Selected items:", selectedItems)

            if self.apply_cip_limits_map.isChecked():
                lower_limit = float(self.map_lower_clip_limit.text())
                upper_limit = float(self.map_upper_clip_limit.text())
                self.map_data = np.clip(self.map_data, lower_limit, upper_limit)

            else:
                lower_limit = None
                upper_limit = None
                self.map_data = self.map_data
                
            
            self.maps_plot_widget.axes.contour(self.map_data, 
                                            levels= self.colormap_n_levels.value(), 
                                            colors='k', origin="image", linewidths=1)
            CSF = self.maps_plot_widget.axes.contourf(self.map_data, 
                                                    levels= self.colormap_n_levels.value(), 
                                                    cmap = "turbo", origin="image", 
                                                    #   vmin = lower_limit, 
                                                    #   vmax = upper_limit, 
                                                    extend = "neither")

            self.maps_plot_widget.figure.colorbar(CSF)
            self.maps_plot_widget.axes.axis('off')

            # def extract_and_combine(s):

                # prefix = s[:19]  # Extract the first 19 characters
                # # Find the substring that contains "Map" surrounded by underscores
                # map_part = [part for part in s.split('_') if 'Map' in part][0]
                # return f"{prefix}_{map_part}"
            # img_title = [extract_and_combine(item) for item in selectedItems]
            
            pattern = re.compile(r'APDMap\d{2}')
            self.img_title = [pattern.search(s).group() for s in selectedItems if pattern.search(s)]
            
            self.maps_plot_widget.axes.set_title(f"{'    '.join(str(i) for i in self.img_title)}", color = "k")

            print(f"{'*'*5} plotting maps succesfully {'*'*5}")
            # self.maps_plot_widget.axes.set_facecolor("white")
            self.maps_plot_widget.figure.set_facecolor("white")
            self.maps_plot_widget.canvas.draw()
        
        except Exception as e:
            print(CustomException(e, sys))

            
    
    def _clear_curr_map_btn_func(self):
        self.maps_plot_widget.figure.clear()
        print("clearing maps")
        self.maps_plot_widget.canvas.draw()
    
    
    
    def _preview_postProcessingMAP_btn_func(self):
        # self.w = AnotherWindow()
        # self.w.show()

        self.InterctiveWindod_edit_map = InterctiveWindowMapErode(self.viewer, self)
        self.InterctiveWindod_edit_map.show()

    
    def _crop_all_views_and_rotate_btn_func(self, return_only_bounding_box = False):
        try:
        # 1. get mask from current Image using auto segemntation

            # pad_value = 0 if self.pad_value.currentText() == "0" else np.nan 
            current_selection = self.viewer.layers.selection.active

            if isinstance(current_selection, Image):

                h_padding = self.pad_h_pixels.value()
                v_padding = self.pad_v_pixels.value()
                orientation = self.crop_view_orientation.currentText()
                list_of_rotate_directions = [combo.currentText() for combo in self.view_rotates]
                
                    
                mask = self._on_click_apply_segmentation_btn_fun(return_result_as_layer=False, 
                                                                 return_mask=True)
        
                # 2. from mask create bounding box
                # boxes = bounding_box_vertices(my_labels_data=mask, 
                #                               area_threshold=1000, 
                #                                 vertical_padding=0, 
                #                                 horizontal_padding=0)
                

                # 3.create and crop boxes from labels
                cropped_images, cropped_labels, bounding_boxes = crop_from_bounding_boxes(img_layer=current_selection,
                                                                                          rotate_directions=list_of_rotate_directions,
                                                                                            my_labels_data=mask,
                                                                                            area_threshold=1000,
                                                                                            vertical_padding=v_padding,
                                                                                            horizontal_padding=h_padding)
                if return_only_bounding_box:
                    return self.viewer.add_shapes(bounding_boxes)

                # 3. arrange and combine boxes
                if self.pad_value.currentText() == "0":
                    pad_value = 0
                elif self.pad_value.currentText() == "NaN":
                    pad_value = np.nan 
                elif self.pad_value.currentText() == "background":
                    # takes the mean of the backgorund
                    pad_value = np.mean(current_selection.data[0][~mask.astype(bool)])

                results = arrange_cropped_images(cropped_images=cropped_images, 
                                                arrangement=orientation, 
                                                padding_value=pad_value)
                
                cropped_labels_3d = [label[np.newaxis, :, :] for label in cropped_labels]
                arranged_labels = arrange_cropped_images([(label, None, None) for label in cropped_labels_3d], 
                                                         arrangement=orientation, 
                                                         padding_value=0)
                if self.return_mask_form_rearranging.isChecked():
                    self.add_result_label(arranged_labels[0], 
                                                img_custom_name="Heart_labels", 
                                                single_label_sufix = f"NullBckgrnd", 
                                                add_to_metadata = f"Background image masked")
                
                self.add_result_img(result_img=results, 
                                operation_name="crop_and_rearrange_views", 
                                custom_img_name=current_selection.name, 
                                method_name="crop_from_shape", 
                                custom_metadata= current_selection.metadata, 
                                sufix="Crop", parameters=None)
            
            else:
                return warn("Please select an image leyer.")
                
        # arrange_cropped_images
            print("cropping")
        except Exception as e:
            print(CustomException(e, sys))
    
    def _join_all_views_and_rotate_btn_func(self):
        current_selection = self.viewer.layers.selection.active
        mask = self._on_click_apply_segmentation_btn_fun(return_result_as_layer=False, 
                                                                 return_mask=True)
        orientation = self.crop_view_orientation.currentText()
        # 3. arrange and combine boxes
        if self.pad_value.currentText() == "0":
            pad_value = 0
        elif self.pad_value.currentText() == "NaN":
            pad_value = np.nan 
        elif self.pad_value.currentText() == "background":
            # takes the mean of the backgorund
            pad_value = np.mean(current_selection.data[0][~mask.astype(bool)])
        cropped_images_names = [item.text() for item in self.join_imgs_selector.selectedItems()]
        cropped_images = [self.viewer.layers[item].data for item in cropped_images_names]

                
        results = arrange_cropped_images(cropped_images=[(img, None, None) for img in cropped_images], 
                                                arrangement=orientation, 
                                                padding_value=pad_value)

                                                
        self.add_result_img(result_img=results, 
                            operation_name="join_and_rearrange_views", 
                            custom_inputs= cropped_images_names,
                            custom_img_name=cropped_images_names[0],
                            method_name=arrange_cropped_images.__name__, 
                            custom_metadata= current_selection.metadata, 
                            sufix="Join",
                            parameters=None)
        print("lalala")
    
    def add_children_tree_widget(self, parent, dictionary):
        """Recursively add children to tree items."""
        if isinstance(dictionary, dict):
            for key, value in dictionary.items():
                child = QTreeWidgetItem(parent, [str(key), str(value) if not isinstance(value, (dict, list)) else ""])
                parent.addChild(child)
                self.add_children_tree_widget(child, value)  # Recursive call
        elif isinstance(dictionary, list):
            for i, item in enumerate(dictionary):
                child = QTreeWidgetItem(parent, [f"Item {i}", str(item) if not isinstance(item, (dict, list)) else ""])
                parent.addChild(child)
                self.add_children_tree_widget(child, item)










        


@magic_factory
def example_magic_widget(img_layer: "napari.layers.Image"):
    print(f"you have selected {img_layer}")


# Uses the `autogenerate: true` flag in the plugin manifest
# to indicate it should be wrapped as a magicgui to autogenerate
# a widget.
def example_function_widget(img_layer: "napari.layers.Image"):
    print(f"you have selected {img_layer}")


# class MyPopup(QWidget):
#     def __init__(self, napari_viewer):
#         super().__init__()
#         # QWidget.__init__(self)
#         self.viewer = napari_viewer
#         self.main_layout = QVBoxLayout()
#         self.setLayout(self.main_layout)

#         # self.tabs = QTabWidget()
#         # self.main_layout.addWidget(self.tabs)
#         ######## pre-processing tab ########
#         self.average_APs_widget = QWidget()
#         self._pre_processing_layout = QVBoxLayout()
#         self.average_APs_widget.setLayout(self._pre_processing_layout)
#         # self.tabs.addTab(self.pre_processing_tab, 'Average Trace')
#         # self.tabs.addTab(self.pre_processing_tab, 'Average Trace')
#         self.main_layout.addWidget(self.average_APs_widget)


#         ######## Pre-processing tab ########
#         ####################################
#         self._pre_processing_layout.setAlignment(Qt.AlignTop)
        
#         ######## pre-processing  group ########
#         self.pre_processing_group = VHGroup('Pre-porcessing', orientation='G')

#         ######## pre-processing btns ########
#         self.inv_and_norm_data_btn = QPushButton("Invert + Normalize (loc max)")        
#         self.pre_processing_group.glayout.addWidget(self.inv_and_norm_data_btn, 0, 1, 1, 1)

#         self.inv_data_btn = QPushButton("Invert signal")
#         self.inv_data_btn.setToolTip(("Invert the polarity of the signal"))
#         self.pre_processing_group.glayout.addWidget(self.inv_data_btn, 1, 1, 1, 1) 

#         self._pre_processing_layout.addWidget(self.pre_processing_group.gbox)

        
#         self.plot_grpup = VHGroup('Plot profile', orientation='G')
#         self.plot_widget =  BaseNapariMPLWidget(self.viewer) # this is the cleanest widget thatz does not have any callback on napari
#         self.plot_grpup.glayout.addWidget(self.plot_widget, 1, 1, 1, 2)
#         self.main_layout.addWidget(self.average_APs_widget)

class InterctiveWindowMapErode(QWidget):
    """
    This "window" is a QWidget. If it has no parent, it
    will appear as a free-floating window as we want.
    """
    def __init__(self, napari_viewer, OMAAS):
        self.viewer = napari_viewer
        self.o = OMAAS
        super().__init__()
        # layout = QVBoxLayout()
        # self.label = QLabel("Another Window % d" % randint(0,100))
        # layout.addWidget(self.label)
        # self.setLayout(layout)
        # self.main_widget = QWidget()
        self.InterctiveWindowMapErode_layout = QVBoxLayout()

        self.preview_map_erode_group = VHGroup('Erode map image edges', orientation='G')

        self.InterctiveWindowMapErode_layout.addWidget(self.preview_map_erode_group.gbox)

        # self.test_label = QLabel("Another Window % d" % randint(0,100))
        # self.preview_map_erode_group.glayout.addWidget(self.test_label, 0, 0, 1, 1)
        self.preview_plotter_widget =  BaseNapariMPLWidget(self.viewer) # this is the cleanest widget thatz does not have any callback on napari
        self.preview_plotter_widget.add_single_axes()
        self.map_data = self.o.map_data.copy()
        self.preview_plotter_widget.axes.imshow(self.map_data, cmap="turbo")
        # self.maps_plot_widget.axes.contour(data, 
        #                                 levels= self.colormap_n_levels.value(), 
        #                                 colors='k', origin="image", linewidths=1)
        # CSF = self.maps_plot_widget.axes.contourf(data, 
        #                                         levels= self.colormap_n_levels.value(), 
        #                                         cmap = "turbo", origin="image", 
        #                                         #   vmin = lower_limit, 
        #                                         #   vmax = upper_limit, 
        #                                         extend = "neither")
        self.preview_map_erode_group.glayout.addWidget(self.preview_plotter_widget, 0, 0, 1, 5)

        self.n_pixels_erode_label = QLabel("Number of Px:")
        self.preview_map_erode_group.glayout.addWidget(self.n_pixels_erode_label, 2, 0, 1, 1)
        
        self.n_pixels_erode_slider = QLabeledSlider()
        self.n_pixels_erode_slider.setRange(0, 100)
        self.preview_map_erode_group.glayout.addWidget(self.n_pixels_erode_slider, 2, 1, 1, 2)

        # self.apply_erosion_btn = QPushButton( "Apply changes")
        # self.preview_map_erode_group.glayout.addWidget(self.apply_erosion_btn, 2, 0, 1, 3)

        self.small_holes_size_label = QLabel("Small holes size")
        self.preview_map_erode_group.glayout.addWidget(self.small_holes_size_label, 2, 3, 1, 1)

        self.small_holes_size_map_spinbox = QSpinBox()
        self.small_holes_size_map_spinbox.setSingleStep(1)
        self.small_holes_size_map_spinbox.setValue(0)
        self.preview_map_erode_group.glayout.addWidget(self.small_holes_size_map_spinbox, 2, 4, 1, 1)
        

        # self.reset_erosion_btn = QPushButton("reset")
        # self.preview_map_erode_group.glayout.addWidget(self.reset_erosion_btn, 1, 5, 1, 1)

        self.gaussian_filter_label = QLabel("Gaussian Filter:")
        self.preview_map_erode_group.glayout.addWidget(self.gaussian_filter_label, 1, 0, 1, 1)
        
        self.gaussian_sigam_label = QLabel("Sigma")
        self.preview_map_erode_group.glayout.addWidget(self.gaussian_sigam_label, 1, 1, 1, 1)

        self.gaussian_sigma = QDoubleSpinBox()
        self.gaussian_sigma.setSingleStep(0.1)
        self.gaussian_sigma.setValue(0)
        self.preview_map_erode_group.glayout.addWidget(self.gaussian_sigma, 1, 2, 1, 1)
                  
        self.gaussian_radius_label = QLabel("Radius")
        self.preview_map_erode_group.glayout.addWidget(self.gaussian_radius_label, 1, 3, 1, 1)

        self.gaussian_radius = QSpinBox()
        self.gaussian_radius.setSingleStep(1)
        self.gaussian_radius.setValue(0)
        self.preview_map_erode_group.glayout.addWidget(self.gaussian_radius, 1, 4, 1, 1)

        # self.apply_gaussian_filt_btn = QPushButton( "View changes")
        # self.preview_map_erode_group.glayout.addWidget(self.apply_gaussian_filt_btn, 4, 0, 1, 3)

        self.accept_post_processing_changes_btn = QPushButton("Accept changes")
        self.preview_map_erode_group.glayout.addWidget(self.accept_post_processing_changes_btn, 3, 0, 1, 2)

        self.reset_all_postprocessing_map_btn = QPushButton("Reset")
        self.preview_map_erode_group.glayout.addWidget(self.reset_all_postprocessing_map_btn, 3, 2, 1, 2)
       
        self.close_postprocessing_map_window_btn = QPushButton("Close")
        self.preview_map_erode_group.glayout.addWidget(self.close_postprocessing_map_window_btn, 3, 4, 1, 1)

                  
        
        self.setLayout(self.InterctiveWindowMapErode_layout)
    
        ##############
        # Callbacks ##
        ##############
        self.accept_post_processing_changes_btn.clicked.connect(self._apply_postprocessing_methods_func)
        self.n_pixels_erode_slider.valueChanged.connect(self.n_pixels_erode_slider_func)
        self.small_holes_size_map_spinbox.valueChanged.connect(self.n_pixels_erode_slider_func)
        # self.apply_gaussian_filt_btn.clicked.connect(self._apply_gaussian_filt_btn_func)
        self.gaussian_sigma.valueChanged.connect(self._apply_gaussian_filt_on_map_func)
        self.gaussian_radius.valueChanged.connect(self._apply_gaussian_filt_on_map_func)
        # self.reset_erosion_btn.clicked.connect(self._reset_all_btn_func)
        self.reset_all_postprocessing_map_btn.clicked.connect(self._reset_all_btn_func)
        self.close_postprocessing_map_window_btn.clicked.connect(self._close_postprocessing_windows_func)


    def _apply_postprocessing_methods_func(self):

        try:
            self.n_pixels_erode_slider_func()
            img_items = [item.text() for item in self.o.map_imgs_selector.selectedItems()]
            current_map_image = self.viewer.layers[img_items[0]]
            # self.o.img_title
            # Step 1: Split the strings into parts
            split_strings = [s.split('_') for s in self.o._get_imgs2d_from_map_selector()]

            # Step 2: Identify the common part and collect APDMapXX parts
            common_parts = split_strings[0][:-1]  # Assume common parts are all parts except the last one
            apdmap_parts = []

            for split_str in split_strings:
                for part in split_str:
                    if "APDMap" in part:
                        apdmap_parts.append(part.replace("APDMap", ""))

            # Step 3: Reconstruct the string
            final_string = "_".join(common_parts) + "_APDMap" + "_".join(apdmap_parts)
            input_imgs = self.o._get_imgs2d_from_map_selector()

            eros_value = self.n_pixels_erode_slider.value()
            small_holes_s = self.small_holes_size_map_spinbox.value()
            sigma = self.gaussian_sigma.value()
            radius = self.gaussian_radius.value()
            
            params = {gaussian_filter_nan.__name__: {"paramters": {"radius": radius,
                      "sigma" : sigma}},
                      "Erosion_image" :{"parameters": {"small_holes_s" : small_holes_s,
                      "eros_value" : eros_value}}
                      }
            
            self.o.add_result_img(result_img=self.result_map_image, 
                                operation_name="Postprocessing_maps[test]", 
                                custom_img_name=f"{final_string}", 
                                method_name="crop_from_shape",
                                custom_inputs = input_imgs,
                                custom_metadata= current_map_image.metadata,
                                sufix="PostProMap", 
                                parameters=params)
            print("Image exported")
        except Exception as e:
            print(CustomException(e, sys))
    
    def n_pixels_erode_slider_func(self):
        try:

            # mask = segment_image_triangle(self.map_data)
            self.result_map_image = self.result_map_image if hasattr(self, "result_map_image") else self.map_data.copy()
            mask = np.invert(np.isnan(self.result_map_image))

            
            # mask = binary_closing(mask, 10)
            eros_value = self.n_pixels_erode_slider.value()
            small_holes_s = self.small_holes_size_map_spinbox.value()           

            if small_holes_s > 0:

                print(f"clossing small object of size = {small_holes_s}")
                mask = remove_small_objects(mask, small_holes_s)
                footprint=[(np.ones((small_holes_s, 1)), 1), (np.ones((1, small_holes_s)), 1)]
                mask = binary_closing(mask, footprint=footprint)
                self.result_map_image = closing(self.result_map_image, footprint=footprint)


            mask = erosion(mask, footprint=disk(eros_value))
            self.result_map_image[~mask] = None
            
            self.preview_plotter_widget.figure.clear()
            self.preview_plotter_widget.add_single_axes()
            self.preview_plotter_widget.axes.imshow(self.result_map_image, cmap="turbo")
            self.preview_plotter_widget.canvas.draw()

            # print("lalala")
        except Exception as e:
            raise CustomException(e, sys)
        

    def _apply_gaussian_filt_on_map_func(self):

        try:

            # self.n_pixels_erode_func()

            # self.result_map_image = self.result_map_image.copy()
            # NOTE: may be use intepolation method to refill holes in mask
            # after that you cna do erosion or gaussina filter.
            # need to try out.
            sigma = self.gaussian_sigma.value()
            radius = self.gaussian_radius.value()
            # self.result_map_image = self.result_map_image if hasattr(self, "result_map_image") else self.map_data.copy()
            self.result_map_image = self.map_data.copy()
            self.result_map_image = gaussian_filter_nan(self.result_map_image, sigma=sigma, radius=radius)

            self.preview_plotter_widget.figure.clear()
            self.preview_plotter_widget.add_single_axes()
            self.preview_plotter_widget.axes.imshow(self.result_map_image, cmap="turbo")
            self.preview_plotter_widget.canvas.draw()
            # print("applying Gaussian filter")
        
        except Exception as e:
            raise CustomException(e, sys)
    
    def _reset_all_btn_func(self):
        try:
            self.result_map_image = self.map_data.copy()

            self.preview_plotter_widget.figure.clear()
            self.preview_plotter_widget.add_single_axes()
            self.preview_plotter_widget.axes.imshow(self.result_map_image, cmap="turbo")
            self.preview_plotter_widget.canvas.draw()
        except Exception as e:
            raise CustomException(e, sys)
        
    def _close_postprocessing_windows_func(self):
        self.close()

        
    
    # def _reset_gauss_filt_btn_func(self):
    #     try:
    #         # self.result_map_image = self.map_data.copy()

    #         self.preview_plotter_widget.figure.clear()
    #         self.preview_plotter_widget.add_single_axes()
    #         self.preview_plotter_widget.axes.imshow(self.result_map_image, cmap="turbo")
    #         self.preview_plotter_widget.canvas.draw()
    #     except Exception as e:
    #         raise CustomException(e, sys)

        
        
        # self.InterctiveWindowMapErode.show()

        # print("preview")















        