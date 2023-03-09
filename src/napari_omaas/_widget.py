"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
from typing import TYPE_CHECKING

from magicgui import magic_factory
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget, QFileDialog, QVBoxLayout, QGroupBox, QGridLayout, QTabWidget, QDoubleSpinBox, QLabel, QComboBox, QSpinBox
from qtpy.QtCore import Qt
import pyqtgraph as pg

from .utils import *

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
        self.inv_img_label = QLabel("Invert image")
        self.pre_processing_group.glayout.addWidget(self.inv_img_label, 3, 0, 1, 1)
        self.inv_data_btn = QPushButton("Apply")
        self.pre_processing_group.glayout.addWidget(self.inv_data_btn, 3, 1, 1, 1)

        self.norm_img_label = QLabel("Normalize image (loc max)")
        self.pre_processing_group.glayout.addWidget(self.norm_img_label, 4, 0, 1, 1)
        self.norm_data_btn = QPushButton("Apply")        
        self.pre_processing_group.glayout.addWidget(self.norm_data_btn, 4, 1, 1, 1)

        self.inv_and_norm_label = QLabel("Invert + Normalize (loc max)")
        self.pre_processing_group.glayout.addWidget(self.inv_and_norm_label, 5, 0, 1, 1)
        self.inv_and_norm_data_btn = QPushButton("Apply")        
        self.pre_processing_group.glayout.addWidget(self.inv_and_norm_data_btn, 5, 1, 1, 1)


        self.splt_chann_label = QLabel("Split Channels")
        self.pre_processing_group.glayout.addWidget(self.splt_chann_label, 6, 0, 1, 1)
        self.splt_chann_btn = QPushButton("Apply")
        self.pre_processing_group.glayout.addWidget(self.splt_chann_btn, 6, 1, 1, 1)
 
        ######## Filters group ########
        self.filter_group = VHGroup('Filter Image', orientation='G')
        self._pre_processing_layout.addWidget(self.filter_group.gbox)
        
        ######## Filters btns ########
        # self.filters_label = QLabel("Gaussian filter")
        # self.filter_group.glayout.addWidget(self.filters_label, 3, 0, 1, 1)

        self.filter_types = QComboBox()
        self.filter_types.addItems(["Gaussian", "Median"])
        self.filter_group.glayout.addWidget(self.filter_types, 3, 1, 1, 1)

        
        self.filt_param = QDoubleSpinBox()
        self.filt_param.setSingleStep(1)
        # self.filt_param.setMaximum(10)
        # self.filt_param.setMinimum(-10)
        self.filt_param.setValue(1)
        self.filter_group.glayout.addWidget(self.filt_param, 3, 2, 1, 1)

        self.apply_filt_btn = QPushButton("apply")
        self.apply_filt_btn.setToolTip(("apply selected filter to the image"))
        self.filter_group.glayout.addWidget(self.apply_filt_btn, 3, 3, 1, 1)

        ######## Transform group ########
        self.transform_group = VHGroup('Transform Image data', orientation='G')
        self._pre_processing_layout.addWidget(self.transform_group.gbox)

        ######## Transform btns ########
        self.inv_img_label = QLabel("Transform data to integer")
        self.transform_group.glayout.addWidget(self.inv_img_label, 3, 0, 1, 1)
        self.transform_to_uint16_btn = QPushButton("Apply")
        self.transform_to_uint16_btn.setToolTip(("Transform numpy array data to type integer np.uint16"))
        self.transform_group.glayout.addWidget(self.transform_to_uint16_btn, 3, 1, 1, 1)


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
        ######## Mot-Correction group ########
        self.mot_correction_group = VHGroup('Apply image registration (motion correction)', orientation='G')
        self._motion_correction_layout.addWidget(self.mot_correction_group.gbox)

        self.inv_and_norm_label = QLabel("Foot print size")
        self.mot_correction_group.glayout.addWidget(self.inv_and_norm_label, 3, 0, 1, 1)
        
        self.footprint_size = QSpinBox()
        self.footprint_size.setSingleStep(1)
        self.footprint_size.setValue(10)
        self.mot_correction_group.glayout.addWidget(self.footprint_size, 3, 1, 1, 1)

        self.radius_size_label = QLabel("Radius size")
        self.mot_correction_group.glayout.addWidget(self.radius_size_label, 4, 0, 1, 1)
        
        self.radius_size = QSpinBox()
        self.radius_size.setSingleStep(1)
        self.radius_size.setValue(7)
        self.mot_correction_group.glayout.addWidget(self.radius_size, 4, 1, 1, 1)

        self.n_warps_label = QLabel("Number of warps")
        self.mot_correction_group.glayout.addWidget(self.n_warps_label, 5, 0, 1, 1)
        
        self.n_warps = QSpinBox()
        self.n_warps.setSingleStep(1)
        self.n_warps.setValue(20)
        self.mot_correction_group.glayout.addWidget(self.n_warps, 5, 1, 1, 1)

        self.apply_mot_correct_btn = QPushButton("apply")
        self.apply_mot_correct_btn.setToolTip(("apply registration method to correct the image for motion artefacts"))
        self.mot_correction_group.glayout.addWidget(self.apply_mot_correct_btn, 6, 0, 1, 2)



        # sub_backg_btn = QPushButton("Subtract Background")
        # rmv_backg_btn = QPushButton("Delete Background")
        # pick_frames_btn = QPushButton("Pick frames")

        ##### instanciate buttons #####
        
        # segmentation
        

        # self.layout().addWidget(seg_heart_btn)
        # self.layout().addWidget(sub_backg_btn)
        # self.layout().addWidget(rmv_backg_btn)
        # self.layout().addWidget(pick_frames_btn)

        # plotter

        graph_container = QWidget()

        # histogram view
        self._graphics_widget = pg.GraphicsLayoutWidget()
        self._graphics_widget.setBackground("w")

        #graph_container.setMaximumHeight(100)
        graph_container.setLayout(QHBoxLayout())
        graph_container.layout().addWidget(self._graphics_widget)

        # individual layers: legend
        self._labels = QWidget()
        self._labels.setLayout(QVBoxLayout())
        self._labels.layout().setSpacing(0)

        # setup layout
        self.setLayout(QVBoxLayout())

        self.layout().addWidget(graph_container)
        self.layout().addWidget(self._labels)



        ######################
        ##### callbacks ######
        ######################
        
        self.inv_data_btn.clicked.connect(self._on_click_inv_data_btn)
        self.norm_data_btn.clicked.connect(self._on_click_norm_data_btn)
        self.inv_and_norm_data_btn.clicked.connect(self._on_click_inv_and_norm_data_btn)
        self.splt_chann_btn.clicked.connect(self._on_click_splt_chann)
        self.rmv_backg_btn.clicked.connect(self._on_click_seg_heart_btn)

        self.apply_filt_btn.clicked.connect(self._on_click_apply_filt_btn)
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
        self.transform_to_uint16_btn.clicked.connect(self._on_click_transform_to_uint16_btn)
        
        
        ##### handle events #####
        self.viewer.layers.events.inserted.connect(self._shapes_layer_list_changed_callback)
        self.viewer.layers.events.removed.connect(self._shapes_layer_list_changed_callback)
        self.viewer.layers.events.reordered.connect(self._shapes_layer_list_changed_callback)
        

    def _on_click_inv_data_btn(self):
        results =invert_signal(self.viewer.layers.selection)
        # print(type(results))
        self.viewer.add_image(results, 
        colormap = "turbo",
        # colormap= "twilight_shifted", 
        name= f"{self.viewer.layers.selection.active}_Inv")



    def _on_click_norm_data_btn(self):
        results = local_normal_fun(self.viewer.layers.selection)
        # print(type(results))
        # local_normal_fun(self.viewer.layers.selection)
        self.viewer.add_image(results, 
        # colormap= "twilight_shifted", 
        colormap = "turbo",
        name= f"{self.viewer.layers.selection.active}_Nor")

    def _on_click_inv_and_norm_data_btn(self):
        self._on_click_inv_data_btn()
        self._on_click_norm_data_btn()


    def _on_click_splt_chann(self):
        my_splitted_images = split_channels_fun(self.viewer.layers.selection)
        curr_img_name = self.viewer.layers.selection.active
        for channel in range(len(my_splitted_images)):
            self.viewer.add_image(my_splitted_images[channel], 
            colormap= "twilight_shifted", 
            name= f"{curr_img_name}_ch{channel + 1}")

    
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


    def _on_click_apply_filt_btn(self):
        # self.gaus_filt_value.value()
        
        ctext = self.filter_types.currentText()

        if ctext == "Gaussian":
            sigma = self.filt_param.value()
            # print(f"applying {ctext} with value: {str(sigma)} ")

            results = apply_gaussian_func(self.viewer.layers.selection, sigma)

            self.viewer.add_image(results, 
            colormap = "turbo",
         # colormap= "twilight_shifted", 
            name= f"{self.viewer.layers.selection.active}_GausFilt_{str(sigma)}")

        
        if ctext == "Median":
            param = self.filt_param.value()
            # print(f"applying {ctext} with value: {str(param)} ")

            results = apply_median_filt_func(self.viewer.layers.selection, param)

            self.viewer.add_image(results, 
            colormap = "turbo",
         # colormap= "twilight_shifted", 
            name= f"{self.viewer.layers.selection.active}_MednFilt_{str(param)}")




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

        results = motion_correction_func(self.viewer.layers.selection, 
                                        foot_print_size=foot_print, 
                                        radius_size=radius_size, num_warp=n_warps)
        self.viewer.add_image(results, 
            colormap = "turbo",
         # colormap= "twilight_shifted", 
            name= f"{self.viewer.layers.selection.active}_MotCorr_fp{str(foot_print)}_rs{str(radius_size)}_nw{str(n_warps)}")
        
    def _on_click_transform_to_uint16_btn(self):
        
        results = transform_to_unit16_func(self.viewer.layers.selection)
        print( "is doing something")

        self.viewer.add_image(results, 
            colormap = "turbo",
         # colormap= "twilight_shifted", 
            name= f"{self.viewer.layers.selection.active}_uint16")






        
                        
                        
                
                

        

    
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