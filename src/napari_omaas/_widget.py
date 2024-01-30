"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
from typing import TYPE_CHECKING

from magicgui import magic_factory
from superqt import QCollapsible, QLabeledSlider, QLabeledRangeSlider, QRangeSlider, QDoubleRangeSlider
from qtpy.QtWidgets import (
    QHBoxLayout, QPushButton, QWidget, QFileDialog, 
    QVBoxLayout, QGroupBox, QGridLayout, QTabWidget, QListWidget,
    QDoubleSpinBox, QLabel, QComboBox, QSpinBox, QLineEdit, QPlainTextEdit,
    QTreeWidget, QTreeWidgetItem, QCheckBox, QSlider, QTableView, QMessageBox, QToolButton, 
    )

from qtpy import QtWidgets
from warnings import warn
from qtpy.QtCore import Qt, QAbstractTableModel, QModelIndex, QRect
from qtpy.QtGui import QIntValidator
from numpy import ndarray as numpy_ndarray
from napari_matplotlib.base import BaseNapariMPLWidget
from napari.layers import Shapes, Image
from napari.utils import progress

import copy
import subprocess
import pandas as pd

from .utils import *
import os
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
        self._layers_processing_layout = QVBoxLayout()
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

        #########################################
        ######## Editing indivicual tabs ########
        #########################################

        ######## Pre-processing tab ########
        ####################################
        self._pre_processing_layout.setAlignment(Qt.AlignTop)
        
        ######## pre-processing  group ########
        self.pre_processing_group = VHGroup('Pre-porcessing', orientation='G')

        ######## pre-processing btns ########
        self.inv_and_norm_data_btn = QPushButton("Invert + Normalize (loc max)")        
        self.pre_processing_group.glayout.addWidget(self.inv_and_norm_data_btn, 1, 1, 1, 1)

        self.inv_data_btn = QPushButton("Invert signal")
        self.inv_data_btn.setToolTip(("Invert the polarity of the signal"))
        self.pre_processing_group.glayout.addWidget(self.inv_data_btn, 1, 2, 1, 1)

        self.loc_norm_data_btn = QPushButton("Normalize (loc max)")        
        self.pre_processing_group.glayout.addWidget(self.loc_norm_data_btn, 2, 1, 1, 1)


        # self.splt_chann_label = QLabel("Split Channels")
        # self.pre_processing_group.glayout.addWidget(self.splt_chann_label, 3, 6, 1, 1)
        self.splt_chann_btn = QPushButton("Split Channels")
        self.pre_processing_group.glayout.addWidget(self.splt_chann_btn, 1, 3, 1, 1)

        self.glob_norm_data_btn = QPushButton("Normalize (global)")
        self.pre_processing_group.glayout.addWidget(self.glob_norm_data_btn, 2, 2, 1, 1)
 
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

        self.butter_cutoff_freq_val = QSpinBox()
        self.butter_cutoff_freq_val.setSingleStep(5)
        self.butter_cutoff_freq_val.setValue(45)
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

        
        
        self.segmentation_group = VHGroup('Segment heart views', orientation='G')
        # self._pre_processing_layout.addWidget(self.filter_group.gbox)

        self._collapse_segmentation_group = QCollapsible('Segmentation', self)
        self._collapse_segmentation_group.addWidget(self.segmentation_group.gbox)

        self.segmentation_methods_lable = QLabel("Method")
        self.segmentation_group.glayout.addWidget(self.segmentation_methods_lable, 0, 0, 1, 1)

        self.segmentation_methods = QComboBox()
        self.segmentation_methods.addItems(["threshold_triangle", "GHT"])
        self.segmentation_group.glayout.addWidget(self.segmentation_methods, 0, 1, 1, 1)

        self.return_img_no_backg_btn = QCheckBox("Return image")
        self.return_img_no_backg_btn.setChecked(True)
        self.return_img_no_backg_btn.setToolTip(("Draw current selection as plot profile"))
        # self._plottingWidget_layout.addWidget(self.plot_profile_btn)
        self.segmentation_group.glayout.addWidget(self.return_img_no_backg_btn, 0, 2, 1, 1)

        self.apply_segmentation_btn = QPushButton("segment stack")
        self.segmentation_group.glayout.addWidget(self.apply_segmentation_btn, 0, 3, 1, 1)





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

        ######## Segmentation group ########
        self.segmentation_group = VHGroup('Segmentation', orientation='G')
        # self._pre_processing_layout.addWidget(self.segmentation_group.gbox)

        ######## Segmentation btns ########
        self.seg_heart_label = QLabel("Segment the heart shape")
        self.segmentation_group.glayout.addWidget(self.seg_heart_label, 3, 0, 1, 1)
        self.seg_heart_btn = QPushButton("apply")
        self.segmentation_group.glayout.addWidget(self.seg_heart_btn, 3, 1, 1, 1)

        self.sub_bkg_label = QLabel("Subtract Background")
        self.segmentation_group.glayout.addWidget(self.sub_bkg_label, 4, 0, 1, 1)
        self.sub_backg_btn = QPushButton("apply")
        self.segmentation_group.glayout.addWidget(self.sub_backg_btn, 4, 1, 1, 1)

        self.del_bkg_label = QLabel("Delete Background")
        self.segmentation_group.glayout.addWidget(self.del_bkg_label, 5, 0, 1, 1)
        self.rmv_backg_btn = QPushButton("apply")
        self.segmentation_group.glayout.addWidget(self.rmv_backg_btn, 5, 1, 1, 1)

        self.pick_frames_btn = QLabel("Pick frames")
        self.segmentation_group.glayout.addWidget(self.pick_frames_btn, 6, 0, 1, 1)
        self.pick_frames_btn = QPushButton("apply")
        self.segmentation_group.glayout.addWidget(self.pick_frames_btn, 6, 1, 1, 1)

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
        
        self.clip_trace_btn = QPushButton("Clip Trace")
        self._plotting_profile_tabs_layout.glayout.addWidget(self.clip_trace_btn, 2, 0, 1, 1)
        
        self.clip_label_range = QCheckBox("Show range")
        self._plotting_profile_tabs_layout.glayout.addWidget(self.clip_label_range, 2, 1, 1, 1)

        self.double_slider_clip_trace = QDoubleRangeSlider(Qt.Orientation.Horizontal)
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
        self._layers_processing_layout.addWidget(self.copy_rois_group.gbox)
        
        self.ROI_selection_1 = QComboBox()
        self.ROI_1_label = QLabel("From layer")
        self.copy_rois_group.glayout.addWidget(self.ROI_1_label, 3, 0, 1, 1)
        # self.ROI_selection_1.setAccessibleName("From layer")
        self.ROI_selection_1.addItems(self.get_rois_list())
        self.copy_rois_group.glayout.addWidget(self.ROI_selection_1, 3, 1, 1, 1)
        
        self.ROI_selection_2 = QComboBox()
        self.ROI_2_label = QLabel("To layer")
        self.copy_rois_group.glayout.addWidget(self.ROI_2_label, 4, 0, 1, 1)
        # self.ROI_selection_2.setAccessibleName("To layer")
        self.ROI_selection_2.addItems(self.get_rois_list())
        self.copy_rois_group.glayout.addWidget(self.ROI_selection_2, 4, 1, 1, 1)

        self.copy_ROIs_btn = QPushButton("Transfer ROIs")
        self.copy_ROIs_btn.setToolTip(("Transfer ROIs from one 'Shape' layer to another 'Shape' layer"))
        self.copy_rois_group.glayout.addWidget(self.copy_ROIs_btn, 5, 0, 1, 2)

        
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
        self.mot_correction_group = VHGroup('Apply image registration (motion correction)', orientation='G')
        self._motion_correction_layout.addWidget(self.mot_correction_group.gbox)


        self.fottprint_size_label = QLabel("Foot print size")
        self.fottprint_size_label.setToolTip(("Footprint size for local normalization"))
        self.mot_correction_group.glayout.addWidget(self.fottprint_size_label, 3, 0, 1, 1)

        self.use_GPU_label = QLabel("Use GPU")
        self.mot_correction_group.glayout.addWidget(self.use_GPU_label, 3, 2, 1, 1)
        
        self.use_GPU = QCheckBox()
        try:
            subprocess.check_output('nvidia-smi')
            warn('Nvidia GPU detected!, setting to default GPU use.\nSet GPU use as default')
            self.use_GPU.setChecked(True)
        except Exception: # this command not being found can raise quite a few different errors depending on the configuration
            warn('No Nvidia GPU in system!, setting to default CPU use')
            self.use_GPU.setChecked(False)
        
        self.mot_correction_group.glayout.addWidget(self.use_GPU,  3, 3, 1, 1)



        
        self.footprint_size = QSpinBox()
        self.footprint_size.setSingleStep(1)
        self.footprint_size.setValue(10)
        self.mot_correction_group.glayout.addWidget(self.footprint_size, 3, 1, 1, 1)

        self.radius_size_label = QLabel("Radius size")
        self.radius_size_label.setToolTip(("Radius of the window considered around each pixel for image registration"))
        self.mot_correction_group.glayout.addWidget(self.radius_size_label, 4, 0, 1, 1)
        
        self.radius_size = QSpinBox()
        self.radius_size.setSingleStep(1)
        self.radius_size.setValue(7)
        self.mot_correction_group.glayout.addWidget(self.radius_size, 4, 1, 1, 1)

        self.n_warps_label = QLabel("Number of warps")
        self.mot_correction_group.glayout.addWidget(self.n_warps_label, 5, 0, 1, 1)
        
        self.n_warps = QSpinBox()
        self.n_warps.setSingleStep(1)
        self.n_warps.setValue(8)
        self.mot_correction_group.glayout.addWidget(self.n_warps, 5, 1, 1, 1)

        self.ref_frame_label = QLabel("Reference frame")
        self.mot_correction_group.glayout.addWidget(self.ref_frame_label, 4, 2, 1, 1)
        
        self.ref_frame_val = QLineEdit()
        self.ref_frame_val.setValidator(QIntValidator()) 
        self.ref_frame_val.setText("0")
        self.mot_correction_group.glayout.addWidget(self.ref_frame_val, 4, 3, 1, 1)

        self.apply_mot_correct_btn = QPushButton("apply")
        self.apply_mot_correct_btn.setToolTip(("apply registration method to correct the image for motion artefacts"))
        self.mot_correction_group.glayout.addWidget(self.apply_mot_correct_btn, 6, 0, 1, 1)


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

        self.ref_frame_label = QLabel("Ref Frame")
        # self.c_kernels_label.setToolTip((""))
        self.mot_correction_group_optimap.glayout.addWidget(self.ref_frame_label, 2, 0, 1, 1)

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
        
        self.slider_APD_percentage = QSlider(Qt.Orientation.Horizontal)
        self.slider_APD_percentage.setRange(10, 100)
        self.slider_APD_percentage.setValue(75)
        self.slider_APD_percentage.setSingleStep(5)
        self.APD_plot_group.glayout.addWidget(self.slider_APD_percentage, 4, 3, 1, 1)
        
        self.slider_APD_perc_label = QLabel(f"APD percentage: {self.slider_APD_percentage.value()}")
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
        self.mv_left_AP_btn.setArrowType(QtCore.Qt.LeftArrow)
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
        self.slider_APD_map_percentage.setRange(5, 100)
        self.slider_APD_map_percentage.setValue(75)
        self.slider_APD_map_percentage.setSingleStep(5)
        self.average_trace_group.glayout.addWidget(self.slider_APD_map_percentage, 6, 1, 1, 1)
        
        self.make_maps_btn = QPushButton("Create Maps")
        self.average_trace_group.glayout.addWidget(self.make_maps_btn, 6, 2, 1, 3)

        self.average_roi_on_map_btn = QPushButton("ROI mean")
        self.average_trace_group.glayout.addWidget(self.average_roi_on_map_btn, 7, 0, 1, 1)






        ######## Settings tab ########
        ####################################

        ######## Macro record group ########
        self._settings_layout.setAlignment(Qt.AlignTop)
        self.macro_group = VHGroup('Record the scrips for analyis', orientation='G')
        self._settings_layout.addWidget(self.macro_group.gbox)

        self.record_script_label = QLabel("Your current actions")
        self.record_script_label.setToolTip('Display bellow the recorded set of actions of your processing pipeline.')
        self.macro_group.glayout.addWidget(self.record_script_label, 1, 0, 1, 4)
       
        self.macro_box_text = QPlainTextEdit()
        self.macro_box_text.setStyleSheet("border: 1px solid black;") 
        self.macro_box_text.setPlaceholderText("###### Start doing operations to populate your macro ######")
        self.macro_group.glayout.addWidget(self.macro_box_text, 2, 0, 1, 4)

        self.activate_macro_label = QLabel("Enable/disable Macro recording")
        self.activate_macro_label.setToolTip('Set on if you want to keep track of the script for reproducibility or further reuse in batch processing')
        self.macro_group.glayout.addWidget(self.activate_macro_label, 3, 0, 1, 1)
        
        self.record_macro_check = QCheckBox()
        self.record_macro_check.setChecked(True) 
        self.macro_group.glayout.addWidget(self.record_macro_check,  3, 1, 1, 1)

        self.clear_last_step_macro_btn = QPushButton("Delete last step")
        self.macro_group.glayout.addWidget(self.clear_last_step_macro_btn,  3, 2, 1, 1)
        
        self.clear_macro_btn = QPushButton("Clear Macro")
        self.macro_group.glayout.addWidget(self.clear_macro_btn,  3, 3, 1, 1)       


        
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

        self.export_processing_steps_btn = QPushButton("Export processing steps")
        self.metadata_display_group.glayout.addWidget(self.export_processing_steps_btn,  1, 3, 1, 1)
        
        self.export_image_btn = QPushButton("Export Image + meatadata")
        self.metadata_display_group.glayout.addWidget(self.export_image_btn,  1, 2, 1, 1)
        # self.layout().addWidget(self.metadata_display_group.gbox) # temporary silence hide the metadatda

        # self._settings_layout.setAlignment(Qt.AlignTop)
        # self.macro_group = VHGroup('Record the scrips for analyis', orientation='G')
        self._settings_layout.addWidget(self.metadata_display_group.gbox)


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
        self.loc_norm_data_btn.clicked.connect(self._on_click_norm_data_btn)
        self.inv_and_norm_data_btn.clicked.connect(self._on_click_inv_and_norm_data_btn)
        self.splt_chann_btn.clicked.connect(self._on_click_splt_chann)
        self.glob_norm_data_btn.clicked.connect(self._on_click_glob_norm_data_btn)
        self.rmv_backg_btn.clicked.connect(self._on_click_seg_heart_btn)

        self.apply_spat_filt_btn.clicked.connect(self._on_click_apply_spat_filt_btn)
        # self.filter_types.activated.connect(self._filter_type_change)
        # rmv_backg_btn.clicked.connect(self._on_click_rmv_backg_btn)
        # sub_backg_btn.clicked.connect(self._on_click_sub_backg_btn)
        self.pick_frames_btn.clicked.connect(self._on_click_pick_frames_btn)
        # inv_and_norm_btn.clicked.connect(self._on_click_inv_and_norm_btn)
        # inv_and_norm_btn.clicked.connect(self._on_click_inv_data_btn, self._on_click_norm_data_btn)
        # load_ROIs_btn.clicked.connect(self._on_click_load_ROIs_btn)
        # save_ROIs_btn.clicked.connect(self._on_click_save_ROIs_btn)
        # self.ROI_selection.currentIndexChanged.connect(self.???)
        self.ROI_selection_1.activated.connect(self._get_ROI_selection_1_current_text)
        self.ROI_selection_2.activated.connect(self._get_ROI_selection_2_current_text)
        self.copy_ROIs_btn.clicked.connect(self._on_click_copy_ROIS)
        self.apply_mot_correct_btn.clicked.connect(self._on_click_apply_mot_correct_btn)
        # self.transform_to_uint16_btn.clicked.connect(self._on_click_transform_to_uint16_btn)
        self.apply_temp_filt_btn.clicked.connect(self._on_click_apply_temp_filt_btn)
        self.compute_APD_btn.clicked.connect(self._get_APD_call_back)
        self.clear_plot_APD_btn.clicked.connect(self._clear_APD_plot)
        self.slider_APD_detection_threshold.valueChanged.connect(self._get_APD_thre_slider_vlaue_func)
        self.slider_APD_detection_threshold_2.valueChanged.connect(self._get_APD_thre_slider_vlaue_func)
        self.slider_APD_percentage.valueChanged.connect(self._get_APD_percent_slider_vlaue_func)
        self.clear_macro_btn.clicked.connect(self._on_click_clear_macro_btn)
        self.clear_last_step_macro_btn.clicked.connect(self._on_click_clear_last_step_macro_btn)
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
        self.apply_segmentation_btn.clicked.connect(self._on_click_apply_segmentation_btn_fun)
        self.average_roi_on_map_btn.clicked.connect(self._on_click_average_roi_on_map_btn_fun)
        self.plot_histogram_btn.clicked.connect(self._on_click_plot_histogram_btn_func)
        self.clear_histogram_btn.clicked.connect(self._on_click_clear_histogram_btn_func)
        self.clip_trace_btn.clicked.connect(self._on_click_clip_trace_btn_func)
        self.clip_label_range.stateChanged.connect(self._dsiplay_range_func)
        self.double_slider_clip_trace.valueChanged.connect(self._double_slider_clip_trace_func)
        self.export_processing_steps_btn.clicked.connect(self._export_processing_steps_btn_func)
        self.export_image_btn.clicked.connect(self._export_image_btn_func)
        self.apply_optimap_mot_corr_btn.clicked.connect(self._apply_optimap_mot_corr_btn_func)
        
        
        
        
        ##### handle events #####
        # self.viewer.layers.events.inserted.connect(self._shapes_layer_list_changed_callback)
        # self.viewer.layers.events.removed.connect(self._shapes_layer_list_changed_callback)
        # self.viewer.layers.events.reordered.connect(self._shapes_layer_list_changed_callback)
        self.viewer.layers.selection.events.active.connect(self._retrieve_metadata_call_back)
        # self.plot_widget.plotter.selector.model().itemChanged.connect(self._get_current_selected_TSP_layer_callback)
        # callback for insert /remove layers
        self.viewer.layers.events.inserted.connect(self._layer_list_changed_callback)
        self.viewer.layers.events.removed.connect(self._layer_list_changed_callback)
        # callback for selection of layers in the selectors
        self.listShapeswidget.itemClicked.connect(self._data_changed_callback)
        self.listImagewidget.itemClicked.connect(self._data_changed_callback)
        # updtae FPS label
        self.viewer.window.qt_viewer.canvas.measure_fps(callback = self.update_fps)
        # callback for trace plotting
        # self.plot_profile_btn.clicked.connect(self._on_click_plot_profile_btn_func)
        self.plot_profile_btn.stateChanged.connect(self._on_click_plot_profile_btn_func)
        # self.selection_layer.events.data.connect()
        

    def _on_click_inv_data_btn(self):
        current_selection = self.viewer.layers.selection.active

        if isinstance(current_selection, Image):
            print(f'computing "invert_signal" to image {current_selection}')
            results =invert_signal(current_selection.data)
            self.add_result_img(result_img=results, single_label_sufix="Inv", add_to_metadata = "inv_signal")
            self.add_record_fun()
        else:
           return warn(f"Select an Image layer to apply this function. \nThe selected layer: '{current_selection}' is of type: '{current_selection._type_string}'")


    def _on_click_norm_data_btn(self):
        current_selection = self.viewer.layers.selection.active

        if isinstance(current_selection, Image):
            print(f'computing "local_normal_fun" to image {current_selection}')
            results = local_normal_fun(current_selection.data)
            self.add_result_img(result_img=results, single_label_sufix="LocNor", add_to_metadata = "Local_norm_signal")
            self.add_record_fun()
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

            for channel in range(len(my_splitted_images)):
                # self.viewer.add_image(my_splitted_images[channel],
                # colormap= "turbo", 
                # name= f"{curr_img_name}_ch{channel + 1}")
                self.add_result_img(result_img=my_splitted_images[channel], img_custom_name=curr_img_name, single_label_sufix=f"Ch{channel}", add_to_metadata = f"Splitted_Channel_f_Ch{channel}")
                self.add_record_fun()
        else:
            warn(f"Select an Image layer to apply this function. \nThe selected layer: '{current_selection}' is of type: '{current_selection._type_string}'")

    def _on_click_glob_norm_data_btn(self):
        current_selection = self.viewer.layers.selection.active
        
        if isinstance(current_selection, Image):
            results = global_normal_fun(current_selection.data)
            self.add_result_img(result_img=results, single_label_sufix="GloNor", add_to_metadata = "Global_norm_signal")

        else:
            return warn(f"Select an Image layer to apply this function. \nThe selected layer: '{current_selection}' is of type: '{current_selection._type_string}'")


    
    def get_rois_list(self):

        shape_layer_list = [layer.name for layer in self.viewer.layers if layer._type_string == 'shapes']
        
        return shape_layer_list

    def update_roi_list(self):

        self.clear()
        self.addItems(self.get_rois_list())
    
    def _shapes_layer_list_changed_callback(self, event):
         if event.type in ['inserted', 'removed']:
            value = event.value
            etype = event.type
            if value._type_string == 'shapes' :
                if value:
                    if etype == 'inserted':  # add layer to model
                        # print("you  enter the event loop")
                        self.ROI_selection_1.clear()
                        self.ROI_selection_1.addItems(self.get_rois_list()) 
                        self.ROI_selection_2.clear()
                        self.ROI_selection_2.addItems(self.get_rois_list())
                        
                    elif etype == 'removed':  # remove layer from model
                        self.ROI_selection_1.clear()
                        self.ROI_selection_1.addItems(self.get_rois_list())
                        self.ROI_selection_2.clear()
                        self.ROI_selection_2.addItems(self.get_rois_list())
                        

                    elif etype == 'reordered':  # remove layer from model
                        self.ROI_selection_1.clear()
                        self.ROI_selection_1.addItems(self.get_rois_list())
                        self.ROI_selection_2.clear()
                        self.ROI_selection_2.addItems(self.get_rois_list())
    
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


    def _on_click_apply_spat_filt_btn(self):
        current_selection = self.viewer.layers.selection.active
        if isinstance(current_selection, Image):
        
            filter_type = self.spat_filter_types.currentText()
            all_my_filters = [self.spat_filter_types.itemText(i) for i in range(self.spat_filter_types.count())]
            sigma = self.sigma_filt_spatial_value.value()
            kernel_size = self.filt_kernel_value.value()
            sigma_col = self.sigma_filt_color_value.value()
            
            if filter_type == all_my_filters[0]:
                print(f'applying "{filter_type}" filter to image {current_selection}')
                results = apply_gaussian_func(current_selection.data, 
                                            sigma= sigma, 
                                            kernel_size=kernel_size)
                self.add_result_img(results, single_label_sufix = f"Filt{filter_type}", KrnlSiz = kernel_size, Sgma = sigma, add_to_metadata = f"{filter_type}Filt_sigma{sigma}_ksize{kernel_size}")

            
            elif filter_type == all_my_filters[3]:
                print(f'applying "{filter_type}" filter to image {current_selection}')
                results = apply_median_filt_func(current_selection.data, kernel_size)
                self.add_result_img(results, single_label_sufix = f"Filt{filter_type}", MednFilt = kernel_size, add_to_metadata = f"{filter_type}Filt_ksize{kernel_size}")

            elif filter_type == all_my_filters[1]:
                print(f'applying "{filter_type}" filter to image {current_selection}')
                results = apply_box_filter(current_selection.data, kernel_size)
                self.add_result_img(results, single_label_sufix = f"Filt{filter_type}", BoxFilt = kernel_size, add_to_metadata = f"{filter_type}Filt_ksize{kernel_size}")
            
            elif filter_type == all_my_filters[2]:
                print(f'applying "{filter_type}" filter to image {current_selection}')
                results = apply_laplace_filter(current_selection.data, kernel_size=kernel_size, sigma=sigma)
                self.add_result_img(results, single_label_sufix = f"Filt{filter_type}", KrnlSiz = kernel_size, Widht = sigma, add_to_metadata = f"{filter_type}Filt_sigma{sigma}_ksize{kernel_size}")
            
            elif filter_type == all_my_filters[4]:
                print(f'applying "{filter_type}" filter to image {current_selection}')
                results = apply_bilateral_filter(current_selection.data, sigma_spa=sigma, sigma_col = sigma_col, wind_size = kernel_size)
                self.add_result_img(results, single_label_sufix = f"Filt{filter_type}", WindSiz = kernel_size, sigma_spa = sigma,  sigma_col = sigma_col, add_to_metadata = f"{filter_type}WindSiz{kernel_size}_sigma_spa{sigma}_sigma_col_{sigma_col}")
            
            self.add_record_fun()

        else:
            warn(f"Select an Image layer to apply this function. \nThe selected layer: '{current_selection}' is of type: '{current_selection._type_string}'")
                
    
    
    
    def add_result_img(self, result_img, single_label_sufix = None, metadata = True, add_to_metadata = None, colormap="turbo", img_custom_name = None, **label_and_value_sufix):
        
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
            self.viewer.add_image(result_img, 
                        metadata = self.curr_img_metadata,
                        colormap = colormap,
                        name = img_name)

        else: 
            self.viewer.add_image(result_img,
                        colormap = colormap,
                        name = img_name)
        


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
        print(f"Current layer 1 is {ctext}")

    def _get_ROI_selection_2_current_text(self, _): # We receive the index, but don't use it.
        ctext = self.ROI_selection_2.currentText()
        print(f"Current layer 2 is {ctext}")

    
    def _on_click_apply_mot_correct_btn(self):
        foot_print = self.footprint_size.value()
        radius_size = self.radius_size.value()
        n_warps = self.n_warps.value()
        try:
            subprocess.check_output('nvidia-smi')
            print('Nvidia GPU detected!')
        except Exception: # this command not being found can raise quite a few different errors depending on the configuration
            warn('No Nvidia GPU in system!, setting to default CPU use')
            self.use_GPU.setChecked(False)
        
        gpu_use = self.use_GPU.isChecked() # put this in the GUI         
        ref_frame_indx = int(self.ref_frame_val.text()) # put this in the GUI
        current_selection = self.viewer.layers.selection.active
        raw_data = current_selection.data
        if gpu_use == True:
            raw_data = cp.asarray(raw_data)

        if current_selection._type_string == "image":
                
            scaled_img = scaled_img_func(raw_data, 
                                        foot_print_size=foot_print)
                
            results = register_img_func(scaled_img, orig_data= raw_data, radius_size=radius_size, num_warp=n_warps, ref_frame=ref_frame_indx)
            
            if not isinstance(results, numpy_ndarray):
                results =  results.get()    

            self.add_result_img(results, MotCorr_fp = foot_print, rs = radius_size, nw=n_warps)
        
            self.add_record_fun()

            
        else:
            warn(f"Select an Image layer to apply this function. \nThe selected layer: '{current_selection}' is of type: '{current_selection._type_string}'")

        
        
    def _on_click_transform_to_uint16_btn(self):
        
        results = transform_to_unit16_func(self.viewer.layers.selection)
        # print( "is doing something")

        self.viewer.add_image(results, 
            colormap = "turbo",
         # colormap= "twilight_shifted", 
            name= f"{self.viewer.layers.selection.active}_uint16")

    def _on_click_apply_temp_filt_btn(self):
        current_selection = self.viewer.layers.selection.active
        
        if isinstance(current_selection, Image):
            filter_type = self.temp_filter_types.currentText()
            all_my_filters = [self.temp_filter_types.itemText(i) for i in range(self.temp_filter_types.count())]
            cutoff_freq_value = self.butter_cutoff_freq_val.value()
            order_value = self.butter_order_val.value()
            fps_val = float(self.fps_val.text())

            if filter_type == all_my_filters[0]:

                print(f'applying "{filter_type}" filter to image {current_selection}')
                results = apply_butterworth_filt_func(current_selection.data, 
                                                    ac_freq=fps_val, 
                                                    cf_freq= cutoff_freq_value, 
                                                    fil_ord=order_value)

                # self.add_result_img(results, buttFilt_fre = cutoff_freq_value, ord = order_value, fps=round(fps_val), add_to_metadata=f"ButterworthFilt_acfreq{fps_val}_cffreq{cutoff_freq_value}_filtord{order_value}")
                self.add_result_img(results, single_label_sufix = f"Filt{filter_type}", cffreq = cutoff_freq_value, ord = order_value, fps=round(fps_val), add_to_metadata = f"{filter_type}Filt_acfreq{fps_val}_cffreq{cutoff_freq_value}_ord{order_value}")
                
            
            elif filter_type == all_my_filters[1]:

                return warn("Current filter '{filter_type}' is not supported.")
            
            self.add_record_fun()

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

    def _get_current_selected_TSP_layer_callback(self, event):
        # this object is a list of image(s) selected from the Time_series_plotter pluggin layer selector
                try:
                    self.current_seleceted_layer_from_TSP = self.main_plot_widget.plotter.selector.model().get_checked()[0].name
                except:
                    self.current_seleceted_layer_from_TSP = "ImageID"
                
                self.table_rstl_name.setPlaceholderText(f"{self.current_seleceted_layer_from_TSP}_APD_rslts")
    
    def _retrieve_metadata_call_back(self, event):

        if event.type in ['active']:
            value = event.value
            if isinstance(value, Image):
                self.img_metadata_dict = self.viewer.layers.selection.active.metadata
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
                    self.fps_val.setText(str(round(1/self.img_metadata_dict["CycleTime"], 2)))
                else:
                    # self.xscale = 1
                    self.fps_val.setText("Unknown sampling frequency (fps)")
                
            if not isinstance(value, Image):
                self.fps_val.setText("")
                self.metadata_tree.clear()



    def _get_APD_call_back(self, event):

        # assert that there is a trace in the main plotting canvas
        if len(self.main_plot_widget.figure.axes) > 0 :

            self._APD_plot_widget.figure.clear()
            self._APD_plot_widget.add_single_axes()
            traces = self.data_main_canvas["y"]
            time = self.data_main_canvas["x"]
            rmp_method = self.APD_computing_method.currentText()
            apd_percentage = self.slider_APD_percentage.value()
            # self.prominence = self.slider_APD_detection_threshold.value() / (self.slider_APD_thres_max_range)
            
            APD_props = []
            # get selection of images iand shape from the selector
            selected_img_list, selected_shps_list = self._get_imgs_and_shpes_items(return_img=True)

            for img_indx, img in enumerate(selected_img_list):

                for shape_indx, shape in enumerate(selected_shps_list[0].data):

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
                    self._APD_plot_widget.axes.plot(time[img_indx + shape_indx], traces[img_indx + shape_indx], label=f'ROI-{shape_indx}', alpha=0.8)

                    ##### catch error here and exit nicely for the user with a warning or so #####
                    try:

                        self.APs_props = compute_APD_props_func(traces[img_indx + shape_indx],
                                                        curr_img_name = img.name, 
                                                        # cycle_length_ms= self.curr_img_metadata["CycleTime"],
                                                        cycle_length_ms= self.xscale,
                                                        rmp_method = rmp_method, 
                                                        apd_perc = apd_percentage, 
                                                        promi=self.prominence, 
                                                        roi_indx=shape_indx)
                        # collect indexes of AP for plotting AP boudaries: ini, end, baseline
                        ini_indx = self.APs_props[-3]
                        peak_indx = self.APs_props[-2]
                        end_indx = self.APs_props[-1]
                        dVdtmax = self.APs_props[5]
                        resting_V = self.APs_props[8]
                        y_min = resting_V

                        y_max = traces[img_indx + shape_indx][peak_indx]
                        # plot vline of AP start
                        self._APD_plot_widget.axes.vlines(time[img_indx + shape_indx][ini_indx], 
                                            ymin= y_min,
                                            ymax= y_max,
                                            linestyles='dashed', color = "green", 
                                            # label=f'AP_ini',
                                            lw = 0.5, alpha = 0.8)
                        # plot vline of AP end
                        self._APD_plot_widget.axes.vlines(time[img_indx + shape_indx][end_indx], 
                                            ymin= y_min,
                                            ymax= y_max,
                                            linestyles='dashed', color = "red", 
                                            # label=f'AP_end',
                                            lw = 0.5, alpha = 0.8)
                        # plot hline of AP baseline
                        self._APD_plot_widget.axes.hlines(resting_V,
                                            xmin = time[img_indx + shape_indx][ini_indx],
                                            xmax = time[img_indx + shape_indx][end_indx],
                                            linestyles='dashed', color = "grey", 
                                            # label=f'AP_base',
                                            lw = 0.5, alpha = 0.8)

                        APD_props.append(self.APs_props)
                        
                        print(f"APD computed on image '{img.name}' with roi: {shape_indx}")

                    except Exception as e:
                        # warn(f"ERROR: Computing APD parameters fails witht error: {repr(e)}.")
                        raise e

            colnames = [ "image_name",
                         "ROI_id",
                         "AP_id" ,
                         "APD_perc" ,
                         "APD",
                         "AcTime_dVdtmax",
                         "amp_Vmax",
                         "BasCycLength_bcl",
                         "resting_V",
                         "time_at_AP_upstroke",
                         "time_at_AP_peak",
                         "time_at_AP_end",
                         "indx_at_AP_upstroke",
                         "indx_at_AP_peak",
                         "indx_at_AP_end"]
            self._APD_plot_widget.axes.legend()
            self._APD_plot_widget.canvas.draw()


            self.APD_props_df = pd.DataFrame(APD_props, columns=colnames).explode(colnames).reset_index(drop=True)

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

            self.add_record_fun()
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
        if len(self.main_plot_widget.figure.axes) > 0 :
            traces = self.data_main_canvas["y"]
            selected_img_list, shapes = self._get_imgs_and_shpes_items(return_img=True)
            for img_indx, img_name in enumerate(selected_img_list):
                for shpae_indx, shape in enumerate(shapes[0].data):

                    try:

                        traces[img_indx + shpae_indx]
                        n_peaks = return_peaks_found_fun(promi=self.prominence, np_1Darray=traces[img_indx + shpae_indx])
                        self.APD_peaks_help_box_label.setText(f'[AP detected]: {n_peaks}')
                        self.APD_peaks_help_box_label_2.setText(f'[AP detected]: {n_peaks}')

                    except Exception as e:
                        print(f">>>>> this is a known error when computing peaks found while creating shapes interactively: '{e}'")

                break

    def _get_APD_percent_slider_vlaue_func(self, value):
        self.slider_APD_perc_label.setText(f'APD percentage: {value}')


    def _on_click_clear_macro_btn(self, event):
        self.macro_box_text.clear()
        macro.clear()

    def add_record_fun(self):
        self.macro_box_text.clear()
        self.macro_box_text.insertPlainText(repr(macro))
    
    def _on_click_clear_last_step_macro_btn(self):
        macro.pop()
        self.add_record_fun()
    
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
    
    def _get_imgs_and_shpes_items(self, return_img = False):
        """
        Helper function that retunr the names of imags and shapes picked in the selector
        """
        if not return_img:

            img_items = [item.text() for item in self.listImagewidget.selectedItems()]
            shapes_items = [item.text() for item in self.listShapeswidget.selectedItems()]
        elif return_img:
            img_items = [self.viewer.layers[item.text()] for item in self.listImagewidget.selectedItems()]
            shapes_items = [self.viewer.layers[item.text()] for item in self.listShapeswidget.selectedItems()]

        return img_items, shapes_items
    


    
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
                        warn(f"File '{filename}{file_format}' exported to: {file_path}.")

                    elif file_format == ".xlsx":
                        file_path = os.path.join(output_dir, f"{filename}{file_format}")
                        self.APD_props_df.to_excel(file_path, index=False)
                        warn(f"File '{filename}{file_format}' exported to: {file_path}.")
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
        Update the selector model on each layer list change to insert or remove items accordingly.
        """
        
        value = event.value
        etype = event.type
        # control selection of Shape layers
        if isinstance(value, Shapes):
            if etype  == 'inserted':
                item = QtWidgets.QListWidgetItem(value.name)
                self.listShapeswidget.addItem(item)
            if event.type  == 'removed':
                item = self.listShapeswidget.findItems(value.name, Qt.MatchExactly)
                item_row = self.listShapeswidget.row(item[0])
                curr_item = self.listShapeswidget.takeItem(item_row)
                del curr_item
       # control selection of Image Layers
        elif isinstance(value, Image) and value.ndim > 2:
            if etype  == 'inserted':
                item = QtWidgets.QListWidgetItem(value.name)
                self.listImagewidget.addItem(item)
            if event.type  == 'removed':
                item = self.listImagewidget.findItems(value.name, Qt.MatchExactly)
                item_row = self.listImagewidget.row(item[0])
                curr_item = self.listImagewidget.takeItem(item_row)
                del curr_item    

    
    def update_fps(self, fps):
        """Update fps."""
        self.viewer.text_overlay.text = f"Currently rendering at: {fps:1.1f} FPS"
    
    def _on_click_plot_profile_btn_func(self):
        state = self.plot_profile_btn.isChecked()
        
        if state == True:
            # print('Checked')
            img_items = [item.text() for item in self.listImagewidget.selectedItems()]
            shapes_items = [item.text() for item in self.listShapeswidget.selectedItems()]
            img_items, shapes_items = self._get_imgs_and_shpes_items(return_img=False)
            
            if not shapes_items and img_items:
                warn("Please create and Select a SHAPE from the Shape selector to plot profile")
            if not img_items and shapes_items:
                warn("Please open and Select an IMAGE from the Image selector to plot profile")
            if not img_items and not shapes_items:
                warn("Please select a SHAPE & IMAGE from the Shape and Image selectors")
            if img_items and shapes_items:
                try:
                    # img_layer = self.viewer.layers[img_items[0]]
                    img_layer = [self.viewer.layers[layer] for layer in img_items]
                    self.shape_layer = self.viewer.layers[shapes_items[0]]
                    n_shapes = len(self.shape_layer.data)
                    if n_shapes == 0:
                        warn("Draw a new square shape to plot profile in the current selected shape")
                    else:
                        self.main_plot_widget.figure.clear()
                        self.main_plot_widget.add_single_axes()
                        # define container for data
                        self.data_main_canvas = {"x": [], "y": []}
                        # take a list of the images that contain "CycleTime" metadata
                        fps_metadata = [image.metadata["CycleTime"] for image in img_layer if "CycleTime" in image.metadata ]
                        imgs_metadata_names = [image.name for image in img_layer if "CycleTime" in image.metadata ]
                        
                        # check that all images contain contain compatible "CycleTime" metadataotherwise trow error
                        if fps_metadata and not (len(img_layer) == len(fps_metadata)):

                            return warn(f"Imcompatible metedata for plotting. Not all images seem to have the same fps metadata as 'CycleTime'. Check that the images have same 'CycleTime'. Current 'CycleTime' values are: {fps_metadata} for images : {imgs_metadata_names}")
                            
                        elif not all(fps == fps_metadata[0] for fps in fps_metadata):

                            return warn(f"Not all images seem to have the same 'CycleTime'. Check that the images have same 'CycleTime'. Current 'CycleTime' values are: {fps_metadata}")
                        else:
                            self.img_metadata_dict = img_layer[0].metadata                        
                        

                        if "CycleTime" in self.img_metadata_dict:
                            self.main_plot_widget.axes.set_xlabel("Time (ms)")
                            self.xscale = self.img_metadata_dict["CycleTime"] * 1000 
                        else:
                            self.main_plot_widget.axes.set_xlabel("Frames")
                            self.xscale = 1

                        # loop over images
                        for img in img_layer:
                            # loop over shapes
                            for roi in range(n_shapes):
                                img_label = f"{img.name}_{shapes_items}_ROI:{roi}"
                                x, y = extract_ROI_time_series(img_layer = img, shape_layer = self.shape_layer, idx_shape = roi, roi_mode="Mean", xscale = self.xscale)
                                if len(img_label) > 40:
                                    img_label = img.name[4:][:12] + "..." + img.name[-12:]
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
                    print(f"You have the following error: --->> {e} <----")
        else:
            # print('Unchecked')
            self.main_plot_widget.figure.clear()
            self.draw()
            # reset some variables
            if hasattr(self, "data_main_canvas"):
                del self.data_main_canvas

        
    
    def _data_changed_callback(self, event):

        # self.prominence = self.slider_APD_detection_threshold.value() / (self.slider_APD_thres_max_range)
        self._get_APD_thre_slider_vlaue_func(value=self.prominence * self.slider_APD_thres_max_range)
        self._retrieve_metadata_call_back(event)
        state = self.plot_profile_btn.isChecked()
        if state:
            self._on_click_plot_profile_btn_func()
            self.main_plot_widget.canvas.draw()
        else:
            # warn("Please Check on 'Plot profile' to creaate the plot")
            return
        
    def _preview_multiples_traces_func(self):

        # assert that there is a trace in the main plotting canvas
        if len(self.main_plot_widget.figure.axes) > 0 :

            # self._data_changed_callback(event)
            self.shape_layer.events.data.connect(self._data_changed_callback)
            # prominence = self.slider_label_current_value / (self.slider_APD_thres_max_range)
            
            traces = self.data_main_canvas["y"][0]
            time = self.data_main_canvas["x"][0]

            try:
                self.ini_i_spl_traces, _, self.end_i_spl_traces = return_AP_ini_end_indx_func(my_1d_array = traces, 
                                                                                    #    cycle_length_ms = self.xscale, 
                                                                                    promi= self.prominence)
            except Exception as e:
                print(f"You have the following error: --->> {e} <----")
                return

            self.slider_N_APs.setRange(0, len(self.ini_i_spl_traces) - 1)
            
            # re-create canvas
            self.average_AP_plot_widget.figure.clear()
            self.average_AP_plot_widget.add_single_axes()
            
            if self.ini_i_spl_traces.size == 1:
                self.average_AP_plot_widget.axes.plot(time, traces, "--", label = f"AP [{0}]", alpha = 0.8)
                # remove splitted_stack value if exists
                try:
                    if hasattr(self, "splitted_stack"):
                        # del self.splitted_stack
                        self.splitted_stack = traces
                    else:
                        raise AttributeError
                except Exception as e:
                    print(f">>>>> this is your error: {e}")
                

                print("Preview trace created")
            elif self.ini_i_spl_traces.size > 1:

                # NOTE: need to fix this function
                self.splitted_stack = split_AP_traces_func(traces, self.ini_i_spl_traces, self.end_i_spl_traces, type = "1d", return_mean=False)
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
                
            
            

                # first create remove the attributes if they already exist
                # self._remove_attribute_widget()
                # self.slider_N_APs_label = QLabel("Slide to select your current AP")
                # self.average_trace_group.glayout.addWidget(self.slider_N_APs_label, 5, 0, 1, 1)

                # self.slider_N_APs = QLabeledSlider(Qt.Orientation.Horizontal)
                
                # self.slider_N_APs.setValue(0)
                # self.average_trace_group.glayout.addWidget(self.slider_N_APs, 5, 1, 1, 1)

                # self.remove_mean_label = QLabel("remove mean")
                # self.average_trace_group.glayout.addWidget(self.slider_N_APs_label, 5, 2, 1, 1)

                # self.remove_mean_check = QCheckBox()
                # self.average_trace_group.glayout.addWidget(self.slider_N_APremove_mean_checks_label, 5, 3, 1, 1)
                
                print("Preview trace created")

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
    def _remove_attribute_widget(self):
        my_attr_list = ["slider_N_APs", "slider_N_APs_label"]
        for attr in my_attr_list:
            if hasattr(self, attr):
                for widget in list_of_widgets:
                    widget.destroy() 

        
        
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
        #     self.add_record_fun()

        
        # assert that you have content in the canvas
        if len(self.average_AP_plot_widget.figure.axes) != 0 and hasattr(self, "data_main_canvas"):            

            ini_i, _, end_i = return_AP_ini_end_indx_func(my_1d_array = self.data_main_canvas["y"][0], promi= self.prominence)

            if ini_i.size > 1:

                img_items, _ = self._get_imgs_and_shpes_items(return_img=True)
                results= split_AP_traces_func(img_items[0].data, ini_i, end_i, type = "3d", return_mean=True)
                self.add_result_img(result_img=results, img_custom_name=img_items[0].name, single_label_sufix="Ave", add_to_metadata = f"Average stack of {len(ini_i)} AP traces")
                print("Average trace created")
                self.add_record_fun()

            elif ini_i.size == 1:
                return warn(f"Only {ini_i.size} AP detected. No average computed.")
            elif ini_i.size < 1:
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

        
        # assert that a profile was created
        if hasattr(self, "data_main_canvas"):

            time = self.data_main_canvas["x"][0]
            _, AP_peaks_indx, _ = return_AP_ini_end_indx_func(self.data_main_canvas["y"][0], promi=self.prominence)
            # assert that you have a single AP detected
            if len(AP_peaks_indx) == 1:

                #########################
                #  start computing maps #
                #########################

                percentage = self.slider_APD_map_percentage.value()
                current_img_selection_name = self.listImagewidget.selectedItems()[0].text()
                current_img_selection = self.viewer.layers[current_img_selection_name]

                # NOTE: 2 states for map type: 0 for Act maps and 2 for APD maps
                map_type = self.toggle_map_type.checkState()
                
                # check for "CycleTime" in metadtata
                if "CycleTime" in self.img_metadata_dict:
                    cycl_t = self.img_metadata_dict["CycleTime"]
                else:
                    cycl_t = 1
                
                is_interpolated = self.make_interpolation_check.isChecked()

                results = return_maps(current_img_selection.data, 
                                    cycle_time=cycl_t,  
                                    interpolate_df = is_interpolated, 
                                    map_type = map_type, 
                                    percentage = percentage)
                
                if map_type == 0:
                    self.add_result_img(result_img=results, 
                                    img_custom_name=current_img_selection.name, 
                                    single_label_sufix=f"ActMap_Interp{str(is_interpolated)[0]}", 
                                    add_to_metadata = f"Activattion Map cycle_time={round(cycl_t, 4)}, interpolate={self.make_interpolation_check.isChecked()}")
                elif map_type == 2:
                    self.add_result_img(result_img=results, 
                                    img_custom_name=current_img_selection.name, 
                                    single_label_sufix=f"APDMap{percentage}_Interp{str(is_interpolated)[0]}", 
                                    add_to_metadata = f"APD{percentage} Map cycle_time={round(cycl_t, 4)}, interpolate={self.make_interpolation_check.isChecked()}")


                self.add_record_fun()
                print("Map generated")
            else:
                return warn("Either non or more than 1 AP detected. Please average your traces, clip 1 AP or make sure you have at least one AP detected by changing the 'Sensitivity threshold'.") 
       
        else:
            return warn("Make first a Preview of the APs detected using the 'Preview traces' button.") 


    def _on_click_average_roi_on_map_btn_fun(self):
        
        _, shapes_items = self._get_imgs_and_shpes_items(return_img=True)
        current_img_selected = self.viewer.layers.selection.active

        if isinstance(current_img_selected, Image) and current_img_selected.ndim == 2:
            ndim = current_img_selected.ndim
            dshape = current_img_selected.data.shape
            masks = shapes_items[0].to_masks(dshape)
            
            if masks.ndim == 3:
                for indx_roi, roi in enumerate(masks):
                    
                    results = np.nanmean(current_img_selected.data[roi])
                    print(f"mean of roi: '{indx_roi}' in shape '{shapes_items[0].name}' for image '{current_img_selected.name}' is: \n{round(results, 2)}")
                return            
            
            elif masks.ndim == 2:
                
                results = np.nanmean(current_img_selected.data[masks])
                return print(f"mean of roi in shape '{shapes_items[0].name}' for image '{current_img_selected.name}' is: \n{round(results, 2)}")
            else:
                warn(f" you seem to have an issue with your shapes")
        
        else:
            
            return warn(f"No an image selected or image: '{current_img_selected.name}' has ndim = {current_img_selected.ndim }. Select an 2d image retunred from a map.")



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
        current_img_selection_name = self.listImagewidget.selectedItems()[0].text()
        current_img_selection = self.viewer.layers[current_img_selection_name]
        dim_shape = current_img_selection.data.shape[1:]
        results = np.gradient(current_img_selection.data, axis=0)
        # make activation time mask using the gradien
        # act_map_mask = results == results.max(axis = 0)
        # act_map_rslt = current_img_selection.data[act_map_mask]
        # act_map_rslt = act_map_rslt.reshape( dim_shape[0], dim_shape[1])
        
        self.add_result_img(result_img=results, 
                            img_custom_name=current_img_selection.name, 
                            single_label_sufix="ActTime", 
                            add_to_metadata = f"Activation Time")
        

    def _on_click_apply_segmentation_btn_fun(self):
        current_selection = self.viewer.layers.selection.active
        if isinstance(current_selection, Image):
        
            segmentation_method_selected = self.segmentation_methods.currentText()
            segmentation_methods = [self.segmentation_methods.itemText(i) for i in range(self.segmentation_methods.count())]

            sigma = self.sigma_filt_spatial_value.value()
            kernel_size = self.filt_kernel_value.value()
            sigma_col = self.sigma_filt_color_value.value()
            
            if segmentation_method_selected == segmentation_methods[0]:
                print(f'applying "{segmentation_method_selected}" method to image {current_selection}')
                mask = segment_image_triangle(current_selection.data)
                mask = polish_mask(mask)

                
            elif segmentation_method_selected == segmentation_methods[1]:
                mask, threshold = segment_image_GHT(current_selection.data, return_threshold=True)
                mask = polish_mask(mask)
                print(f'Segmenting using "{segmentation_method_selected}" method to image {current_selection} with threshold: {threshold}')

                
            else:
                return warn( f"selected filter '{segmentation_method_selected}' no known.")
            
            
            # return results

            # self.viewer.add_labels(mask,
            #                        name = f"Heart_labels", 
            #                        metadata = current_selection.metadata)
            self.add_result_label(mask, 
                                    img_custom_name="Heart_labels", 
                                    single_label_sufix = f"NullBckgrnd", 
                                    add_to_metadata = f"Background image masked")

            if self.return_img_no_backg_btn.isChecked():
                # 8. remove background using mask
                n_frames =current_selection.data.shape[0]
                masked_image = current_selection.data.copy()
                masked_image[~np.tile(mask.astype(bool), (n_frames, 1, 1))] = None

                # 9. subtract bacground from original image 
                background = np.nanmean(masked_image)
                
                masked_image = masked_image - background

                self.add_result_img(masked_image, 
                                    img_custom_name=current_selection.name, 
                                    single_label_sufix = f"NullBckgrnd",
                                    add_to_metadata = f"Background subtracted")
                 
                    
            
            self.add_record_fun()

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
                print(f"Histogram of frame: '{time_point}' created ")

            else:
                data = layer.data
                # bins = np.linspace(np.min(data), np.max(data), n_bins)
                self.histogram_plot_widget.axes.hist(data.ravel(), 
                                                        bins=n_bins, 
                                                        #  histtype="step",
                                                        edgecolor='white',
                                                        #  linewidth=1.2,
                                                        label=layer.name)
                
                print(f"Histogram of full stack ({data.shape[0]} frames) created ")
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
        # assert that there is a trace in the main plotting canvas
        if len(self.main_plot_widget.figure.axes) > 0 :
            
            if self.clip_label_range.isChecked():
                
                time = self.data_main_canvas["x"]
                selected_img_list, _ = self._get_imgs_and_shpes_items(return_img=True)
                for image in selected_img_list:
                    results = image.data[start_indx:end_indx]
                    self.add_result_img(result_img=results, img_custom_name = image.name, single_label_sufix="TimeCrop", add_to_metadata = f"TimeCrop_at_Indx_[{start_indx}:{end_indx}]")
                    # self.add_record_fun()
                    # self.plot_profile_btn.setChecked(False)
                    self.clip_label_range.setChecked(False)
                    print(f"image '{image.name}' clipped from {round(start_indx * self.xscale, 2)} to {round(end_indx * self.xscale, 2)}")
            else:
                return warn("Preview the clipping range firts by ticking the 'Show region'.")
        else:
            return warn("Create a trace first by clicking on 'Display Profile'") 
    
    def _dsiplay_range_func(self):
        state = self.clip_label_range.isChecked()
        
        if state == True and self.plot_profile_btn.isChecked() :
            start_indx, end_indx = self.double_slider_clip_trace.value()
            # assert that there is a trace in the main plotting canvas
            if len(self.main_plot_widget.figure.axes) > 0 :
                time = self.data_main_canvas["x"]
                selected_img_list, _ = self._get_imgs_and_shpes_items(return_img=True)
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
        state = self.clip_label_range.isChecked()
        
        if state == True and self.plot_profile_btn.isChecked():
            # self._dsiplay_range_func()

            for i in range(2):
                self.main_plot_widget.figure.axes[0].lines[-1].remove()
            
            self._dsiplay_range_func()


        else:
            return
    
    def _export_processing_steps_btn_func(self):
        
        current_selection = self.viewer.layers.selection.active
        
        if isinstance(current_selection, Image):
            metadata = current_selection.metadata
            key = "ProcessingSteps"

            if key in metadata.keys():

                fileName, _ = QFileDialog.getSaveFileName(self,
                                                    "Save File",
                                                        "",
                                                        "Hierarchical Data Format (*.h5 *.hdf5);;Text Files (*.txt)")
                if not len(fileName) == 0:
                
                    with h5py.File(fileName, "w") as hf:

                        # NOTE: may be add more information: original image name, date, etc?
                        hf.attrs.update({key:metadata[key]})
                        
                    print(f"Processing steps for image {current_selection.name} exported")
            else:
                return warn("No 'Preprocessing' steps detected.")
        else:
            return warn("Please select an image leyer.")
    

    def _export_image_btn_func(self):

        current_selection = self.viewer.layers.selection.active
        
        if isinstance(current_selection, Image):
            
            options = QFileDialog.Options()
            # options |= QFileDialog.DontUseNativeDialog                        
            fileName, extension_ = QFileDialog.getSaveFileName(self,
                                                    "Save File",
                                                        "",
                                                        "OME-TIFF image format (*ome.tif);;All Files (*)",
                                                        options=options)
            if fileName:
                if not len(fileName) == 0:
                    file_basename = os.path.basename(fileName)
                    file_dir = os.path.dirname(fileName)
                    # remove extension if exists and preserve only first part
                    splitted_file_basename = file_basename.split(".")
                    fileName = os.path.join(file_dir, splitted_file_basename[0] + ".ome.tif" ) # here you can eventually to change 

                    metadata = current_selection.metadata

                    # NOTE: still not able to export the metadata correctly with this method
                    with tifffile.TiffWriter(fileName) as tif:
                        
                        metadata_tif = {
                            'axes': 'TYX',
                            'fps': 1/metadata['CycleTime']
                            # 'comment': metadata
                            # 'shape': (metadata['NumberOfFrames'], metadata['DetectorDimensions'][0], metadata['DetectorDimensions'][1])
                        }
                        options = dict(photometric='minisblack',
                                    #    tile=(128, 128),
                                    #    compression='jpeg',
                                    #    resolutionunit='CENTIMETER',
                                    #    maxworkers=2
                                    )
                        
                        tif.write(current_selection.data, 
                                #   metadata =  current_selection.metadata,
                                metadata =  metadata_tif,
                                **options)

                    
                    print(f"Image '{current_selection.name}' exported")
                else:
                    return
        else:
            return warn("Please select an image leyer.")
    

    def _apply_optimap_mot_corr_btn_func(self):
        
        current_selection = self.viewer.layers.selection.active
        
        if isinstance(current_selection, Image) and current_selection.ndim == 3:
            c_k = self.c_kernels.value()
            pre_smooth_t=self.pre_smooth_temp.value()
            pre_smooth_s=self.pre_smooth_spat.value()

            print("running motion stabilization")
            results = optimap_mot_correction(current_selection.data, 
                                             c_k = c_k,
                                             pre_smooth_t= pre_smooth_t,
                                             proe_smooth_s= pre_smooth_s)
            
            self.add_result_img(result_img=results, 
                                img_custom_name = current_selection.name,
                                single_label_sufix= f'MotStab_ck{c_k}_PresmT{pre_smooth_t}_PresmS{pre_smooth_s}', 
                                add_to_metadata = f'Motion_correction_optimap_ck{c_k}_PresmT{pre_smooth_t}_PresmS{pre_smooth_s}')
            
            self.add_record_fun()

        else:
            
            return warn(f"No an image selected or image: '{current_selection.name}' has ndim = {current_selection.ndim }. Select an temporal 3d image stack")










        


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
















        