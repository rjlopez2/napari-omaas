"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
from typing import TYPE_CHECKING

from magicgui import magic_factory
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget

from .utils import *

if TYPE_CHECKING:
    import napari


class ExampleQWidget(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # in one of two ways:
    # 1. use a parameter called `napari_viewer`, as done here
    # 2. use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        inv_data_btn = QPushButton("invert image")
        norm_data_btn = QPushButton("normalize image")


        self.setLayout(QHBoxLayout())
        self.layout().addWidget(inv_data_btn)
        self.layout().addWidget(norm_data_btn)
        
        # callbacks
        inv_data_btn.clicked.connect(self._on_click_inv_data_btn)
        norm_data_btn.clicked.connect(self._on_click_norm_data_btn)

    def _on_click_inv_data_btn(self):
        results =invert_signal(self.viewer.layers.selection)
        # name= f'{self.viewer.layers.layers.Image}_norm')
        # self.viewer._add_layer_from_data(results)
        self.viewer.add_image(results, 
        colormap= "twilight_shifted", 
        # gamma= 0.2,
        name= f"{self.viewer.layers.selection.active}_Inv")
        # invert_signal(self.viewer.layers.selection)

    def _on_click_norm_data_btn(self):

        results = local_normal_fun(self.viewer.layers.selection)


        self.viewer.add_image(results, 
        colormap= "twilight_shifted", 
        # gamma= 0.2,
        name= f"{self.viewer.layers.selection.active}_Nor")




@magic_factory
def example_magic_widget(img_layer: "napari.layers.Image"):
    print(f"you have selected {img_layer}")


# Uses the `autogenerate: true` flag in the plugin manifest
# to indicate it should be wrapped as a magicgui to autogenerate
# a widget.
def example_function_widget(img_layer: "napari.layers.Image"):
    print(f"you have selected {img_layer}")
