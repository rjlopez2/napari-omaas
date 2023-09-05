"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
from typing import TYPE_CHECKING

from magicgui import magic_factory
from qtpy.QtWidgets import (
    QHBoxLayout, QPushButton, QWidget, QFileDialog, 
    QVBoxLayout, QGroupBox, QGridLayout, QTabWidget, 
    QDoubleSpinBox, QLabel, QComboBox, QSpinBox, QLineEdit, QPlainTextEdit,
    QTreeWidget, QTreeWidgetItem, QCheckBox, QSlider, QTableView, QMessageBox
    )
from warnings import warn
from qtpy.QtCore import Qt, QAbstractTableModel, QModelIndex
from PyQt5.QtGui import QIntValidator
from numpy import ndarray as numpy_ndarray
import pyqtgraph as pg
from napari_time_series_plotter import TSPExplorer
from napari_matplotlib.base import NapariMPLWidget
import matplotlib.pyplot as plt

import copy
import subprocess
import pandas as pd

from .utils import *
import os
from pathlib import Path


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
        self.tabs.addTab(self.layers_processing, 'Shapes')

        ######## Mot-Correction tab ########
        self.motion_correction = QWidget()
        self._motion_correction_layout = QVBoxLayout()
        self.motion_correction.setLayout(self._motion_correction_layout)
        self.tabs.addTab(self.motion_correction, 'Mot-Correction')

        ######## APD analysis tab ########
        self.APD_analysis = QWidget()
        self._APD_analysis_layout = QVBoxLayout()
        self.APD_analysis.setLayout(self._APD_analysis_layout)
        self.tabs.addTab(self.APD_analysis, 'APD analysis')

        ######## Settings tab ########
        self.settings = QWidget()
        self._settings_layout = QVBoxLayout()
        self.settings.setLayout(self._settings_layout)
        self.tabs.addTab(self.settings, 'Settings')

        #########################################
        ######## Editing indivicual tabs ########
        #########################################

        ######## Pre-processing tab ########
        ####################################
        self._pre_processing_layout.setAlignment(Qt.AlignTop)
        
        ######## pre-processing  group ########
        self.pre_processing_group = VHGroup('Pre-porcessing', orientation='G')
        self._pre_processing_layout.addWidget(self.pre_processing_group.gbox)

        ######## pre-processing btns ########
        self.inv_and_norm_data_btn = QPushButton("Invert + Normalize (loc max)")        
        self.pre_processing_group.glayout.addWidget(self.inv_and_norm_data_btn, 3, 1, 1, 1)

        self.inv_data_btn = QPushButton("Invert signal")
        self.inv_data_btn.setToolTip(("Invert the polarity of the signal"))
        self.pre_processing_group.glayout.addWidget(self.inv_data_btn, 3, 2, 1, 1)

        self.norm_data_btn = QPushButton("Normalize (loc max)")        
        self.pre_processing_group.glayout.addWidget(self.norm_data_btn, 3, 3, 1, 1)


        # self.splt_chann_label = QLabel("Split Channels")
        # self.pre_processing_group.glayout.addWidget(self.splt_chann_label, 3, 6, 1, 1)
        self.splt_chann_btn = QPushButton("Split Channels")
        self.pre_processing_group.glayout.addWidget(self.splt_chann_btn, 3, 4, 1, 1)
 
        ######## Filters group ########
        self.filter_group = VHGroup('Filter Image', orientation='G')
        self._pre_processing_layout.addWidget(self.filter_group.gbox)


        ####### temporal filter subgroup #######     
        self.temp_filter_group = VHGroup('Temporal Filters', orientation='G')
        self.filter_group.glayout.addWidget(self.temp_filter_group.gbox)

        ######## temporal Filters btns ########
        self.temp_filter_types = QComboBox()
        self.temp_filter_types.addItems(["Butterworth", "FIR"])
        self.temp_filter_group.glayout.addWidget(self.temp_filter_types, 3, 0, 1, 1)

        self.cutoff_freq_label = QLabel("Cutoff frequency")
        self.temp_filter_group.glayout.addWidget(self.cutoff_freq_label, 3, 1, 1, 1)

        self.butter_cutoff_freq_val = QSpinBox()
        self.butter_cutoff_freq_val.setSingleStep(5)
        self.butter_cutoff_freq_val.setValue(30)
        self.temp_filter_group.glayout.addWidget(self.butter_cutoff_freq_val, 3, 2, 1, 1)
        
        self.filt_order_label = QLabel("Filter order")
        self.temp_filter_group.glayout.addWidget(self.filt_order_label, 3, 3, 1, 1)

        self.butter_order_val = QSpinBox()
        self.butter_order_val.setSingleStep(1)
        self.butter_order_val.setValue(5)
        self.temp_filter_group.glayout.addWidget(self.butter_order_val, 3, 4, 1, 1)

        self.fps_label = QLabel("Sampling Freq (Hz)")
        self.temp_filter_group.glayout.addWidget(self.fps_label, 3, 5, 1, 1)
        
        self.fps_val = QLineEdit()
        self.fps_val.setText("")
        self.temp_filter_group.glayout.addWidget(self.fps_val, 3, 6, 1, 1)

        self.apply_temp_filt_btn = QPushButton("apply")
        self.temp_filter_group.glayout.addWidget(self.apply_temp_filt_btn, 3, 7, 1, 1)





        ####### spacial filter subgroup #######
        self.spac_filter_group = VHGroup('Spacial Filters', orientation='G')
        self.filter_group.glayout.addWidget(self.spac_filter_group.gbox)

        
        ######## spacial Filters btns ########
        # self.filters_label = QLabel("Gaussian filter")
        # self.filter_group.glayout.addWidget(self.filters_label, 3, 0, 1, 1)

        self.spat_filter_types = QComboBox()
        self.spat_filter_types.addItems(["Gaussian", "Box Filter", "Laplace Filter", "Median"])
        self.spac_filter_group.glayout.addWidget(self.spat_filter_types, 3, 1, 1, 1)

        self.sigma_label = QLabel("Sigma")
        self.spac_filter_group.glayout.addWidget(self.sigma_label, 3, 2, 1, 1)

        self.sigma_filt_param = QDoubleSpinBox()
        self.sigma_filt_param.setSingleStep(1)
        self.sigma_filt_param.setSingleStep(0.1)
        self.sigma_filt_param.setValue(0.5)
        self.spac_filter_group.glayout.addWidget(self.sigma_filt_param, 3, 3, 1, 1)

        self.kernels_label = QLabel("Kernel size")
        self.spac_filter_group.glayout.addWidget(self.kernels_label, 3, 4, 1, 1)

        self.filt_kernel_value = QSpinBox()
        self.filt_kernel_value.setSingleStep(1)
        self.filt_kernel_value.setSingleStep(1)
        self.filt_kernel_value.setValue(3)
        self.spac_filter_group.glayout.addWidget(self.filt_kernel_value, 3, 5, 1, 1)


        self.apply_spat_filt_btn = QPushButton("apply")
        self.apply_spat_filt_btn.setToolTip(("apply selected filter to the image"))
        self.spac_filter_group.glayout.addWidget(self.apply_spat_filt_btn, 3, 6, 1, 2)

       
        ######## Load spool data btns Group ########
        self.load_spool_group = VHGroup('Load Spool data', orientation='G')
        self.filter_group.glayout.addWidget(self.load_spool_group.gbox)

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
        self.transform_group = VHGroup('Transform Image data', orientation='G')
        self._motion_correction_layout.addWidget(self.transform_group.gbox)

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


        ######## APD-analysis tab ########
        # ####################################
        self._APD_analysis_layout.setAlignment(Qt.AlignTop)

        ##### APD_plot_group ########
        self.APD_plot_group = VHGroup('APD plot group', orientation='G')
        self._APD_analysis_layout.addWidget(self.APD_plot_group.gbox)

        # self._APD_widget_TSP = TSPExplorer(self.viewer)
        # self.APD_plot_group.glayout.addWidget(self._APD_widget_TSP, 3, 0, 1, 1)
        
        # self._APD_TSP = NapariMPLWidget(self.viewer)
        # self.APD_plot_group.glayout.addWidget(self._APD_TSP, 3, 0, 1, 8)
        # self.APD_axes = self._APD_TSP.canvas.figure.subplots()

        self.compute_APD_btn = QPushButton("Compute APDs")
        self.compute_APD_btn.setToolTip(("PLot the current traces displayed in main plotter"))
        self.APD_plot_group.glayout.addWidget(self.compute_APD_btn, 4, 0, 1, 1)

        self.clear_plot_APD_btn = QPushButton("Clear traces")
        self.clear_plot_APD_btn.setToolTip(("PLot the current traces displayed in main plotter"))
        self.APD_plot_group.glayout.addWidget(self.clear_plot_APD_btn, 4, 1, 1, 1)

        self.APD_computing_method_label = QLabel("AP detection method")
        self.APD_computing_method_label.setToolTip(("""        
        Select the method to compute the resting (membrane) to detect the AP. 
         Methods are : 
        - bcl_to_bcl: from BCL (Basal cycle length) to BCL.
        - pre_upstroke_min: minimum value Pre-upstroke, 
        - post_AP_min: minimum value after AP,
        - ave_pre_post_min: average the minimum value before and after stroke.
         """))
        self.APD_plot_group.glayout.addWidget(self.APD_computing_method_label, 4, 2, 1, 1)
        
        self.APD_computing_method = QComboBox()
        self.APD_computing_method.addItems(["bcl_to_bcl", "pre_upstroke_min", "post_AP_min", "ave_pre_post_min"])
        self.APD_plot_group.glayout.addWidget(self.APD_computing_method, 4, 3, 1, 1)
        
        self.slider_APD_detection_threshold = QSlider(Qt.Orientation.Horizontal)
        self.slider_APD_thres_max_range = 10000
        self.slider_APD_detection_threshold.setRange(1, 1000)
        self.slider_APD_detection_threshold.setValue(500)
        self.APD_plot_group.glayout.addWidget(self.slider_APD_detection_threshold, 4, 6, 1, 1)
        
        self.slider_label_current_value = QLabel(f"Sensitivity threshold: {self.slider_APD_detection_threshold.value() / (self.slider_APD_thres_max_range )}")
        self.slider_label_current_value.setToolTip('Change the threshold sensitivity for the APD detection base on peak "prominence"')
        self.APD_plot_group.glayout.addWidget(self.slider_label_current_value, 4, 4, 1, 1)
        
        self.APD_peaks_help_box_label_def_value = 0
        self.APD_peaks_help_box_label = QLabel(f"[detected]: {self.APD_peaks_help_box_label_def_value}")
        self.APD_peaks_help_box_label.setToolTip('Display number of peaks detected as you scrol over the "Sensitivity threshold')
        self.APD_plot_group.glayout.addWidget(self.APD_peaks_help_box_label, 4, 5, 1, 1)
        
        self.slider_APD_percentage = QSlider(Qt.Orientation.Horizontal)
        self.slider_APD_percentage.setRange(10, 100)
        self.slider_APD_percentage.setValue(75)
        self.slider_APD_percentage.setSingleStep(5)
        self.APD_plot_group.glayout.addWidget(self.slider_APD_percentage, 4, 8, 1, 1)
        
        self.slider_APD_perc_label = QLabel(f"APD percentage: {self.slider_APD_percentage.value()}")
        self.slider_APD_perc_label.setToolTip('Change the APD at the given percentage')
        self.APD_plot_group.glayout.addWidget(self.slider_APD_perc_label, 4, 7, 1, 1)
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
        self.APD_plot_group.glayout.addWidget(self.APD_propert_table, 5, 0, 1, 8)


         ##### APD export results ########
        self.APD_export_group = VHGroup('Export results', orientation='G')
        self._APD_analysis_layout.addWidget(self.APD_export_group.gbox)


        
        self.APD_rslts_export_file_format_label = QLabel("File format")
        self.APD_export_group.glayout.addWidget(self.APD_rslts_export_file_format_label, 7, 0, 1, 1)
        
        self.APD_rslts_export_file_format = QComboBox()
        self.APD_rslts_export_file_format.addItems([".csv", ".xlsx"])
        self.APD_export_group.glayout.addWidget(self.APD_rslts_export_file_format, 7, 1, 1, 1)

        self.search_dir_APD_rslts_btn = QPushButton("change directory")
        self.search_dir_APD_rslts_btn.setToolTip(("Navigate to change the current directory to save your APD results"))
        self.APD_export_group.glayout.addWidget(self.search_dir_APD_rslts_btn, 7, 2, 1, 1)

        self.APD_rslt_dir_btn_label = QLabel("Current Directory")
        self.APD_rslt_dir_btn_label.setToolTip("Drag and drop folders here to change the current directory to save your APD results")
        self.APD_export_group.glayout.addWidget(self.APD_rslt_dir_btn_label, 7, 4, 1, 1)

        self.APD_rslts_dir_box_text = QLineEdit()
        self.APD_rslts_dir_box_text.installEventFilter(self)
        self.APD_rslts_dir_box_text.setAcceptDrops(True)
        self.APD_rslts_dir_box_text.setPlaceholderText(os.getcwd())
        self.APD_export_group.glayout.addWidget(self.APD_rslts_dir_box_text, 7, 5, 1, 1)
        
        self.copy_APD_rslts_btn = QPushButton("Copy table")
        self.copy_APD_rslts_btn.setToolTip(("Copy to clipboard the current APD results."))
        self.APD_export_group.glayout.addWidget(self.copy_APD_rslts_btn, 6, 0, 1, 2)

        self.save_APD_rslts_btn = QPushButton("Export table")
        self.save_APD_rslts_btn.setToolTip(("Export current APD results to a directory in .csv format."))
        self.APD_export_group.glayout.addWidget(self.save_APD_rslts_btn, 6, 2, 1, 2)

        self.label_rstl_name = QLabel("Rename results")
        self.label_rstl_name.setToolTip(("Set the name for the resulting table"))
        self.APD_export_group.glayout.addWidget(self.label_rstl_name, 6, 4,  1, 1)
        
        self.table_rstl_name = QLineEdit()
        self.table_rstl_name.setToolTip(("Drag and drop or copy/paste a directory path to export your results"))
        self.APD_export_group.glayout.addWidget(self.table_rstl_name, 6, 5, 1, 1)

        ######## Settings tab ########
        ####################################

        ######## Macro record group ########
        self._settings_layout.setAlignment(Qt.AlignTop)
        self.macro_group = VHGroup('Record the scrips for analyis', orientation='G')
        self._settings_layout.addWidget(self.macro_group.gbox)

        self.record_script_label = QLabel("Macro")
        self.record_script_label.setToolTip('Set on if you want to keep track of the script for reproducibility or further reuse in batch processing')
        self.macro_group.glayout.addWidget(self.record_script_label, 3, 2, 1, 1)
        
        self.record_macro_check = QCheckBox()
        self.record_macro_check.setChecked(True) 
        self.macro_group.glayout.addWidget(self.record_macro_check,  3, 3, 1, 1)

        self.clear_last_step_macro_btn = QPushButton("Delete last step")
        self.macro_group.glayout.addWidget(self.clear_last_step_macro_btn,  3, 4, 1, 1)
        
        self.clear_macro_btn = QPushButton("Clear Macro")
        self.macro_group.glayout.addWidget(self.clear_macro_btn,  3, 5, 1, 1)       
       
        self.macro_box_text = QPlainTextEdit()
        self.macro_box_text.setPlaceholderText("###### Start doing operations to populate your macro ######")
        self.macro_group.glayout.addWidget(self.macro_box_text, 4, 2, 1, 1)
        

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
        self.metadata_display_group.glayout.addWidget(self.metadata_tree)
        # self.layout().addWidget(self.metadata_display_group.gbox) # temporary silence hide the metadatda

        # self._settings_layout.setAlignment(Qt.AlignTop)
        # self.macro_group = VHGroup('Record the scrips for analyis', orientation='G')
        self._settings_layout.addWidget(self.metadata_display_group.gbox)


        ######################
        ##### Plotters ######
        ######################

        ##### using pyqtgraph ######
        self.plotting_group = VHGroup('Plot profile', orientation='G')
        # self.layout().addWidget(self.plotting_group.gbox)

        ######## pre-processing btns ########
        # self.inv_img_label = QLabel("Invert image")
        # self.pre_processing_group.glayout.addWidget(self.inv_img_label, 3, 0, 1, 1)
        # self.inv_data_btn = QPushButton("Apply")
        # self.pre_processing_group.glayout.addWidget(self.inv_data_btn, 3, 1, 1, 1)

        # self.plotting_group = VHGroup('Pre-porcessing', orientation='G')
        # self._pre_processing_layout.addWidget(self.plotting_group.gbox)







        # graph_container = QWidget()

        # histogram view
        self._graphics_widget = pg.GraphicsLayoutWidget()
        self._graphics_widget.setBackground("w")


        self.plotting_group.glayout.addWidget(self._graphics_widget, 3, 0, 1, 1)

        hour = [1,2,3,4,5,6,7,8,9,10]
        temperature = [30,32,34,32,33,31,29,32,35,45]

        self.p2 = self._graphics_widget.addPlot()
        axis = self.p2.getAxis('bottom')
        axis.setLabel("Distance")
        axis = self.p2.getAxis('left')
        axis.setLabel("Intensity")

        self.p2.plot(hour, temperature, pen="red", name="test")

        # individual layers: legend
        # self._labels = QWidget()
        # self._labels.setLayout(QVBoxLayout())
        # # self._labels.layout().setSpacing(0)

        # # setup layout
        # self.setLayout(QVBoxLayout())

        # self.layout().addWidget(graph_container)
        # self.layout().addWidget(self._labels)


        ##### using TSPExplorer ######

        self._graphics_widget_TSP = TSPExplorer(self.viewer)
        self.layout().addWidget(self._graphics_widget_TSP, 1)




        ##################################################################
        ############################ callbacks ###########################
        ##################################################################
        
        self.inv_data_btn.clicked.connect(self._on_click_inv_data_btn)
        self.norm_data_btn.clicked.connect(self._on_click_norm_data_btn)
        self.inv_and_norm_data_btn.clicked.connect(self._on_click_inv_and_norm_data_btn)
        self.splt_chann_btn.clicked.connect(self._on_click_splt_chann)
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
        self.compute_APD_btn.clicked.connect(self._get_APD_params_call_back)
        self.clear_plot_APD_btn.clicked.connect(self._clear_APD_plot)
        self.slider_APD_detection_threshold.valueChanged.connect(self._get_APD_thre_slider_vlaue_func)
        self.slider_APD_percentage.valueChanged.connect(self._get_APD_percent_slider_vlaue_func)
        self.clear_macro_btn.clicked.connect(self._on_click_clear_macro_btn)
        self.clear_last_step_macro_btn.clicked.connect(self._on_click_clear_last_step_macro_btn)
        self.load_spool_dir_btn.clicked.connect(self._load_current_spool_dir_func)
        self.search_spool_dir_btn.clicked.connect(self._search_and_load_spool_dir_func)
        self.copy_APD_rslts_btn.clicked.connect(self._on_click_copy_APD_rslts_btn_func)
        self.search_dir_APD_rslts_btn.clicked.connect(self._on_click_search_new_dir_APD_rslts_btn_func)
        self.save_APD_rslts_btn.clicked.connect(self._on_click_save_APD_rslts_btn_func)
        
        
        
        ##### handle events #####
        self.viewer.layers.events.inserted.connect(self._shapes_layer_list_changed_callback)
        self.viewer.layers.events.removed.connect(self._shapes_layer_list_changed_callback)
        self.viewer.layers.events.reordered.connect(self._shapes_layer_list_changed_callback)
        self.viewer.layers.selection.events.active.connect(self._retrieve_metadata_call_back)
        self._graphics_widget_TSP.plotter.selector.model().itemChanged.connect(self._get_current_selected_TSP_layer_callback)
        

    def _on_click_inv_data_btn(self):
        current_selection = self.viewer.layers.selection.active

        if current_selection._type_string == "image":
            print(f'computing "invert_signal" to image {current_selection}')
            results =invert_signal(current_selection.data)
            self.add_result_img(result_img=results, single_label_sufix="Inv", add_to_metadata = "inv_signal")
            self.add_record_fun()
        else:
            warn(f"Select an Image layer to apply this function. \nThe selected layer: '{current_selection}' is of type: '{current_selection._type_string}'")


    def _on_click_norm_data_btn(self):
        current_selection = self.viewer.layers.selection.active

        if current_selection._type_string == "image":
            print(f'computing "local_normal_fun" to image {current_selection}')
            results = local_normal_fun(current_selection.data)
            self.add_result_img(result_img=results, single_label_sufix="Nor", add_to_metadata = "norm_signal")
            self.add_record_fun()
        else:
            warn(f"Select an Image layer to apply this function. \nThe selected layer: '{current_selection}' is of type: '{current_selection._type_string}'")


    def _on_click_inv_and_norm_data_btn(self):
        self._on_click_inv_data_btn()
        self._on_click_norm_data_btn()


    def _on_click_splt_chann(self):
        current_selection = self.viewer.layers.selection.active

        if current_selection._type_string == "image":
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
        if current_selection._type_string == "image":
        
            filter_type = self.spat_filter_types.currentText()
            sigma = self.sigma_filt_param.value()
            kernel_size = self.filt_kernel_value.value()
            
            if filter_type == "Gaussian":
                print(f'applying "apply_gaussian_func" to image {current_selection}')
                results = apply_gaussian_func(current_selection.data, 
                                            sigma= sigma, 
                                            kernel_size=kernel_size)
                self.add_result_img(results, KrnlSiz = kernel_size, Sgma = sigma)

            
            if filter_type == "Median":
                print(f'applying "apply_median_filt_func" to image {current_selection}')
                results = apply_median_filt_func(current_selection.data, kernel_size)
                self.add_result_img(results, MednFilt = kernel_size)

            if filter_type == "Box Filter":
                print(f'applying "apply_box_filter" to image {current_selection}')
                results = apply_box_filter(current_selection.data, kernel_size)
                self.add_result_img(results, BoxFilt = kernel_size)
            
            if filter_type == "Laplace Filter":
                print(f'applying "apply_laplace_filter" to image {current_selection}')
                results = apply_laplace_filter(current_selection.data, kernel_size=kernel_size, sigma=sigma)
                self.add_result_img(results, KrnlSiz = kernel_size, Widht = sigma)
            
            self.add_record_fun()

        else:
            warn(f"Select an Image layer to apply this function. \nThe selected layer: '{current_selection}' is of type: '{current_selection._type_string}'")
                
    
    
    
    def add_result_img(self, result_img, single_label_sufix = None, metadata = True, add_to_metadata = None, colormap="turbo", img_custom_name = None, **label_and_value_sufix):
        
        if img_custom_name is not None:
            img_name = img_custom_name
        else:
            img_name = self.viewer.layers.selection.active.name

        self.curr_img_metadata = copy.deepcopy(self.viewer.layers.selection.active.metadata)

        key_name = "Processing_method"
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
        if self.viewer.layers.selection.active._type_string == "image":

            cutoff_freq_value = self.butter_cutoff_freq_val.value()
            order_value = self.butter_order_val.value()
            fps_val = float(self.fps_val.text())

            results = apply_butterworth_filt_func(current_selection.data, 
                                                ac_freq=fps_val, 
                                                cf_freq= cutoff_freq_value, 
                                                fil_ord=order_value)

            self.add_result_img(results, buttFilt_fre = cutoff_freq_value, ord = order_value, fps=round(fps_val))
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
                    self.current_seleceted_layer_from_TSP = self._graphics_widget_TSP.plotter.selector.model().get_checked()[0].name
                except:
                    self.current_seleceted_layer_from_TSP = "ImageID"
                
                self.table_rstl_name.setPlaceholderText(f"{self.current_seleceted_layer_from_TSP}_APD_rslts")
    
    def _retrieve_metadata_call_back(self, event):

        if event.type in ['active']:
            value = event.value
            if value is not None and value._type_string == 'image' :
                self.img_metadata_dict = self.viewer.layers.selection.active.metadata
                if "CycleTime" in self.img_metadata_dict:
                    # print(f"getting image: '{self.viewer.layers.selection.active.name}'")
                    self.metadata_tree.clear()
                    metadata = self.img_metadata_dict
                    items = []
                    for key, values in metadata.items():
                        item = QTreeWidgetItem([key, str(values)])
                        items.append(item)
                
                    self.metadata_tree.insertTopLevelItems(0, items)  
                    self.xscale = metadata["CycleTime"]       
                    # update plotter x scale and x label expressed in ms
                    self._graphics_widget_TSP.options.xscale.setText(str(metadata["CycleTime"] * 1000))
                    self._graphics_widget_TSP.options.xaxis_label.setText("Time (ms)")
                    options = self._graphics_widget_TSP.options.plotter_options()  
                    self._graphics_widget_TSP.plotter.update_options(options) 
                    self.fps_val.setText(str(round(1/metadata["CycleTime"], 2)))
                
            if value is not None and value._type_string != 'image' :
                self.metadata_tree.clear()
                self._graphics_widget_TSP.options.xscale.setText(str(1))
                self._graphics_widget_TSP.options.xaxis_label.setText("Time")
                options = self._graphics_widget_TSP.options.plotter_options()
                self.fps_val.setText("")



    def _get_APD_params_call_back(self, event):
        if len(self._graphics_widget_TSP.plotter.data) > 0 :
                       
            # Clear the canvas before start plotting if plot exist
            # try:
            #     if hasattr(self, "APD_axes_main_canvas"):
            #         self.APD_axes_main_canvas.remove()
            # except Exception as e:
            #     print(f">>>>> this is your error: {e}")
            self._clear_APD_plot(self)

            self.APD_axes_main_canvas = self._graphics_widget_TSP.plotter.canvas.figure.subplots()
            # self.APD_axes_main_canvas = self._graphics_widget_TSP.plotter.axes

            # handles = []
            # print("lalala")
            traces = self._graphics_widget_TSP.plotter.data[1::2]
            shapes = self._graphics_widget_TSP.plotter.selection_layer.data
            time = self._graphics_widget_TSP.plotter.data[0]
            lname = self.viewer.layers.selection.active.name
            rmp_method = self.APD_computing_method.currentText()
            apd_percentage = self.slider_APD_percentage.value()
            prominence = self.slider_APD_detection_threshold.value() / (self.slider_APD_thres_max_range)
            # self.viewer.layers.select_previous()
            # self.img_metadata_dict = self.viewer.layers.selection.active.metadata
            APD_props = []
            selected_img_list = [img.name for img in  self._graphics_widget_TSP.plotter.selector.model().get_checked()]

            for img_indx, img_name in enumerate(selected_img_list):


                for shpae_indx, lalala in enumerate(shapes):

                    # update detected APs labels
                    self.APD_peaks_help_box_label.setText(f'[detected]: {return_peaks_found_fun(promi=prominence, np_1Darray=traces[img_indx + shpae_indx])}')

                    # self.APD_axes.plot(time, traces[img_indx + shpae_indx], label=f'{lname}_ROI-{shpae_indx}', alpha=0.5)
                    self.APD_axes_main_canvas.plot(time, traces[img_indx + shpae_indx], label=f'{lname}_ROI-{shpae_indx}', alpha=0.5)

                    # ##### catch error here and exit nicely for the user with a warning or so #####
                    try:

                        props = compute_APD_props_func(traces[img_indx + shpae_indx], 
                                                        curr_img_name = img_name, 
                                                        # cycle_length_ms= self.curr_img_metadata["CycleTime"],
                                                        cycle_length_ms= self.img_metadata_dict["CycleTime"],
                                                        rmp_method = rmp_method, 
                                                        apd_perc = apd_percentage, 
                                                        promi=prominence, 
                                                        roi_indx=shpae_indx)
                        
                        ini_indx = props[-3]
                        peak_indx = props[-2]
                        end_indx = props[-1]
                        dVdtmax = props[5]
                        resting_V = props[8]
                        # ini_indx = [props[val][-3] for val in range(len(props))]
                        # peak_indx = [props[val][-2] for val in range(len(props))]
                        # end_indx = [props[val][-1] for val in range(len(props))]               

                        # text_lalala = ["lalala" for i in range(len(props[0]))]
                        # props.append(text_lalala)
                        y_min = resting_V    
                        # y_min = traces[img_indx + shpae_indx][ini_indx]    
                        y_max = traces[img_indx + shpae_indx][peak_indx]

                        self.APD_axes_main_canvas.vlines(time[ini_indx], 
                                            ymin= y_min,
                                            ymax= y_max,
                                            linestyles='dashed', color = "green", label=f'AP_ini', lw = 0.5, alpha = 0.8)
                        
                        # self._graphics_widget_TSP
                        # self.APD_axes = self._APD_TSP.canvas.figure.subplots()


                        self.APD_axes_main_canvas.vlines(time[end_indx], 
                                            ymin= y_min,
                                            ymax= y_max,
                                            linestyles='dashed', color = "red", label=f'AP_end', lw = 0.5, alpha = 0.8)

                        self.APD_axes_main_canvas.hlines(resting_V,
                                            xmin = time[ini_indx],
                                            xmax = time[end_indx],
                                            linestyles='dashed', color = "grey", label=f'AP_end', lw = 0.5, alpha = 0.8)

                        APD_props.append(props)

                    except Exception as e:
                        # warn(f"ERROR: Computing APD parameters fails witht error: {repr(e)}.")
                        raise e



            # self._APD_TSP._draw()
            self._graphics_widget_TSP.plotter._draw()

            # print(acttime_peaks_indx, ini_peaks_indx )
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


            self.APD_props_df = pd.DataFrame(APD_props, columns=colnames).explode(colnames).reset_index(drop=True)

            # convert back to the correct type the numeric columns
            cols_to_keep = ["image_name", "ROI_id", "AP_id" ]
            cols_to_numeric = self.APD_props_df.columns.difference(cols_to_keep)

            self.APD_props_df[cols_to_numeric] = self.APD_props_df[cols_to_numeric].apply(pd.to_numeric, errors = "coerce")

            # convert numeric values to ms and round then
            self.APD_props_df = self.APD_props_df.apply(lambda x: np.round(x * 1000, 2) if x.dtypes == "float64" else x ) 

            
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

                

    
    def _clear_APD_plot(self, event):
        """
        Clear the canvas.
        """
        # self.APD_axes.clear()
        # self._graphics_widget_TSP.plotter.clear()
        # self._graphics_widget_TSP.plotter.canvas.figure.clear()
        # self.APD_axes_main_canvas.axes.clear()


        # if (self.APD_axes_main_canvas):
        #     del self.APD_axes_main_canvas
        # self._APD_TSP._draw()
        # plt.close(self.APD_axes_main_canvas)
        # plt.close(self._graphics_widget_TSP.plotter.canvas.figure)
        # self._graphics_widget_TSP.plotter.axes.remove()
        # self._graphics_widget_TSP.plotter.clear()
        # self._graphics_widget_TSP.plotter.axes.remove()
         # Clear the canvas before start plotting if plot exist
        try:
            if hasattr(self, "APD_axes_main_canvas"):
                self.APD_axes_main_canvas.remove()
        except Exception as e:
            print(f">>>>> this is your error: {e}")



        # self.APD_axes_main_canvas.remove()
        self._graphics_widget_TSP.plotter._draw()

        model = PandasModel(self.AP_df_default_val)
        self.APD_propert_table.setModel(model)

        # ---->>>> this return the data currently pltting -> self._graphics_widget_TSP.plotter.data
        # ----->>>>> this retrn the new cavas to plot on to -> self._APD_TSP.canvas.figure.subplots

    def _get_APD_thre_slider_vlaue_func(self, value):
        prominence = self.slider_APD_detection_threshold.value() / (self.slider_APD_thres_max_range)

        self.slider_label_current_value.setText(f'Sensitivity threshold: {prominence}')
        
        # check that you have content in the graphics panel
        if len(self._graphics_widget_TSP.plotter.data) > 0 :
            traces = self._graphics_widget_TSP.plotter.data[1::2]
            shapes = self._graphics_widget_TSP.plotter.selection_layer.data
            selected_img_list = [img.name for img in  self._graphics_widget_TSP.plotter.selector.model().get_checked()]
            for img_indx, img_name in enumerate(selected_img_list):
                for shpae_indx, trace in enumerate(shapes):
                    traces[img_indx + shpae_indx]
                    self.APD_peaks_help_box_label.setText(f'[detected]: {return_peaks_found_fun(promi=prominence, np_1Darray=traces[img_indx + shpae_indx])}')

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
                    # self.msg = QMessageBox()
                    # self.msg.setIcon(QMessageBox.Information)
                    # self.msg.setText("Error")
                    # self.msg.setInformativeText('More information')
                    # self.msg.setWindowTitle("Error")
                    # self.msg.exec_()
                    self.APD_props_df.to_clipboard(index=False) 
                    print(">>>>> data copied to clipboard <<<<<<")
                    warn("APD Table copied to clipboard")
                
                else:
                    warn("No data was copied! Make sure you have a APD reulst table and has len > 0")
            else:
                warn("No data was copied! Make sure you have a APD reulst table.")

                    
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

                    # output_dir = str(QFileDialog.getExistingDirectory(self, "Select Directory", self.APD_rslts_dir_box_text.placeholderText()))
                    file_format = self.APD_rslts_export_file_format.currentText()

                    if file_format == ".csv":
                        file_path = os.path.join(output_dir, f"{filename}{file_format}")
                        self.APD_props_df.to_csv(file_path, index=False)
                        print(f">>>>> File exported to: {file_path} <<<<<<")

                    elif file_format == ".xlsx":
                        file_path = os.path.join(output_dir, f"{filename}{file_format}")
                        self.APD_props_df.to_excel(file_path, index=False)
                        print(f">>>>> File exported to: {file_path} <<<<<<")
                else:
                    warn("No APD results table found or len of the table is < 0.")
            else:
                    warn("No APD results table found.")

        except Exception as e:
            print(f">>>>> this is your error: {e}")



@magic_factory
def example_magic_widget(img_layer: "napari.layers.Image"):
    print(f"you have selected {img_layer}")


# Uses the `autogenerate: true` flag in the plugin manifest
# to indicate it should be wrapped as a magicgui to autogenerate
# a widget.
def example_function_widget(img_layer: "napari.layers.Image"):
    print(f"you have selected {img_layer}")





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

