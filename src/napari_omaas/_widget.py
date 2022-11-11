"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
from typing import TYPE_CHECKING

from magicgui import magic_factory
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget, QFileDialog

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

        # create buttons
        inv_data_btn = QPushButton("invert image")
        norm_data_btn = QPushButton("normalize image")

        load_ROIs_btn = QPushButton("load ROIs")
        save_ROIs_btn = QPushButton("Save ROIs")


        self.setLayout(QHBoxLayout())
        self.layout().addWidget(inv_data_btn)
        self.layout().addWidget(norm_data_btn)
        self.layout().addWidget(load_ROIs_btn)
        self.layout().addWidget(save_ROIs_btn)
        
        # callbacks
        inv_data_btn.clicked.connect(self._on_click_inv_data_btn)
        norm_data_btn.clicked.connect(self._on_click_norm_data_btn)
        load_ROIs_btn.clicked.connect(self._on_click_load_ROIs_btn)
        save_ROIs_btn.clicked.connect(self._on_click_save_ROIs_btn)

    def _on_click_inv_data_btn(self):
        results =invert_signal(self.viewer.layers.selection)
        self.viewer.add_image(results, 
        colormap= "twilight_shifted", 
        name= f"{self.viewer.layers.selection.active}_Inv")

    def _on_click_norm_data_btn(self):
        results = local_normal_fun(self.viewer.layers.selection)
        self.viewer.add_image(results, 
        colormap= "twilight_shifted", 
        name= f"{self.viewer.layers.selection.active}_Nor")

    def _on_click_load_ROIs_btn(self, event=None, filename=None):
        if filename is None: filename, _ = QFileDialog.getOpenFileName(self, "Load ROIs", ".", "ImageJ ROIS(*.roi *.zip)")
        self.viewer.open(filename, plugin='napari_jroireader')
        
    
    def _on_click_save_ROIs_btn(self, event=None, filename=None):
        if filename is None: filename, _ = QFileDialog.getSaveFileName(self, "Save as .csv", ".", "*.csv")
        # self.viewer.layers.save(filename, plugin='napari_jroiwriter')
        self.viewer.layers.save(filename, plugin='napari')
    



@magic_factory
def example_magic_widget(img_layer: "napari.layers.Image"):
    print(f"you have selected {img_layer}")


# Uses the `autogenerate: true` flag in the plugin manifest
# to indicate it should be wrapped as a magicgui to autogenerate
# a widget.
def example_function_widget(img_layer: "napari.layers.Image"):
    print(f"you have selected {img_layer}")
