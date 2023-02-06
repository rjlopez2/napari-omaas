"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
from typing import TYPE_CHECKING

from magicgui import magic_factory
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget, QFileDialog, QVBoxLayout, QGroupBox, QGridLayout, QTabWidget, QDoubleSpinBox, QLabel, QComboBox
from qtpy.QtCore import Qt


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

        # create tabs
        self.options_tab = QWidget()
        self._options_tab_layout = QVBoxLayout()
        self.options_tab.setLayout(self._options_tab_layout)
        self.tabs.addTab(self.options_tab, 'Pre-processing')

        self.layers_processing = QWidget()
        self._layers_processing_layout = QVBoxLayout()
        self.layers_processing.setLayout(self._layers_processing_layout)
        self.tabs.addTab(self.layers_processing, 'Processing Layers')

        #/////// Processing layers tab /////////
        self._layers_processing_layout.setAlignment(Qt.AlignTop)
        self.working_on_layers_groups = VHGroup('Copy ROIs from one layer to another', orientation='G')
        self._layers_processing_layout.addWidget(self.working_on_layers_groups.gbox)
        
        self.ROI_selection_1 = QComboBox()
        self.ROI_1_label = QLabel("From layer")
        self.working_on_layers_groups.glayout.addWidget(self.ROI_1_label, 3, 0, 1, 1)
        # self.ROI_selection_1.setAccessibleName("From layer")
        self.ROI_selection_1.addItems(self.get_rois_list())
        self.working_on_layers_groups.glayout.addWidget(self.ROI_selection_1, 3, 1, 1, 1)
        
        self.ROI_selection_2 = QComboBox()
        self.ROI_2_label = QLabel("To layer")
        self.working_on_layers_groups.glayout.addWidget(self.ROI_2_label, 4, 0, 1, 1)
        # self.ROI_selection_2.setAccessibleName("To layer")
        self.ROI_selection_2.addItems(self.get_rois_list())
        self.working_on_layers_groups.glayout.addWidget(self.ROI_selection_2, 4, 1, 1, 1)

        self.copy_ROIs_btn = QPushButton("Transfer ROIs")
        self.working_on_layers_groups.glayout.addWidget(self.copy_ROIs_btn, 5, 0, 1, 2)


        #/////// Options tab /////////
        self._options_tab_layout.setAlignment(Qt.AlignTop)
        self.options_group = VHGroup('Segmentation Options', orientation='G')
        self._options_tab_layout.addWidget(self.options_group.gbox)
        

        self.flow_threshold_label = QLabel("Gaussian filter")
        self.options_group.glayout.addWidget(self.flow_threshold_label, 3, 0, 1, 1)
        self.flow_threshold = QDoubleSpinBox()
        self.flow_threshold.setSingleStep(1)
        self.flow_threshold.setMaximum(10)
        self.flow_threshold.setMinimum(-10)
        self.flow_threshold.setValue(1)
        self.options_group.glayout.addWidget(self.flow_threshold, 3, 1, 1, 1)

        self.btn_select_options_file = QPushButton("apply")
        self.btn_select_options_file.setToolTip(("apply gaussina filter to selected image"))
        self.options_group.glayout.addWidget(self.btn_select_options_file, 4, 0, 1, 1)

        ##### instanciate buttons #####
        inv_data_btn = QPushButton("Invert image")
        norm_data_btn = QPushButton("Normalize (local max)")
        splt_chann_btn = QPushButton("Split Channels")
        # segmentation
        seg_heart_btn = QPushButton("Segment the heart shapes")
        sub_backg_btn = QPushButton("Subtract Background")
        rmv_backg_btn = QPushButton("Remove Background")
        pick_frames_btn = QPushButton("Pick frames")
        # inv_and_norm_btn = QPushButton("invert + Normaize (local max)")

        # load_ROIs_btn = QPushButton("load ROIs")
        # save_ROIs_btn = QPushButton("Save ROIs")

        ##### adding buttons in layout #####
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(inv_data_btn)
        self.layout().addWidget(norm_data_btn)
        self.layout().addWidget(splt_chann_btn)

        # self.run_group = VHGroup('Run analysis', orientation='G')
        # self._segmentation_layout.addWidget(self.run_group.gbox)

        self.layout().addWidget(seg_heart_btn)
        self.layout().addWidget(sub_backg_btn)
        self.layout().addWidget(rmv_backg_btn)
        self.layout().addWidget(pick_frames_btn)
        # self.layout().addWidget(inv_and_norm_btn)
        

        # self.layout().addWidget(load_ROIs_btn)
        # self.layout().addWidget(save_ROIs_btn)
        
        ##### callbacks #####
        inv_data_btn.clicked.connect(self._on_click_inv_data_btn)
        norm_data_btn.clicked.connect(self._on_click_norm_data_btn)
        splt_chann_btn.clicked.connect(self._on_click_splt_chann)
        rmv_backg_btn.clicked.connect(self._on_click_seg_heart_btn)
        # rmv_backg_btn.clicked.connect(self._on_click_rmv_backg_btn)
        # sub_backg_btn.clicked.connect(self._on_click_sub_backg_btn)
        pick_frames_btn.clicked.connect(self._on_click_pick_frames_btn)
        # inv_and_norm_btn.clicked.connect(self._on_click_inv_and_norm_btn)
        # inv_and_norm_btn.clicked.connect(self._on_click_inv_data_btn, self._on_click_norm_data_btn)
        # load_ROIs_btn.clicked.connect(self._on_click_load_ROIs_btn)
        # save_ROIs_btn.clicked.connect(self._on_click_save_ROIs_btn)
        # self.ROI_selection.currentIndexChanged.connect(self.???)
        self.ROI_selection_1.activated.connect(self._get_ROI_selection_1_current_text)
        self.ROI_selection_2.activated.connect(self._get_ROI_selection_2_current_text)
        self.copy_ROIs_btn.clicked.connect(self._on_click_copy_ROIS)
        
        
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
        print(type(results))
        local_normal_fun(self.viewer.layers.selection)
        # self.viewer.add_image(results, 
        # colormap= "twilight_shifted", 
        # name= f"{self.viewer.layers.selection.active}_Nor")

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
    
    def _get_ROI_selection_1_current_text(self, _): # We receive the index, but don't use it.
        ctext = self.ROI_selection_1.currentText()
        print(f"Current layer 1 is {ctext}")

    def _get_ROI_selection_2_current_text(self, _): # We receive the index, but don't use it.
        ctext = self.ROI_selection_2.currentText()
        print(f"Current layer 2 is {ctext}")
        
                        
                        
                
                

        

    
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