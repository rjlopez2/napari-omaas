"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
from typing import TYPE_CHECKING

from magicgui import magic_factory
from qtpy.QtWidgets import QHBoxLayout, QVBoxLayout, QTabWidget, QPushButton, QWidget, QGroupBox, QGridLayout

from .widgets import *

if TYPE_CHECKING:
    import napari


class Omass(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # in one of two ways:
    # 1. use a parameter called `napari_viewer`, as done here
    # 2. use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        # self.btn = QPushButton("Click me!")
        # self.btn.clicked.connect(self._on_click)

        # self.setLayout(QHBoxLayout())
        # self.layout().addWidget(btn)

        ##################################################
        self.main_layout = QVBoxLayout()
        self.setLayout(self.main_layout)

        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)

        # Analysis tab
        self.analysis = QWidget()
        self._analysis_layout = QVBoxLayout()
        self.analysis.setLayout(self._analysis_layout)
        self.tabs.addTab(self.analysis, 'Analysis')

        # aquisition tab
        self.aquisition_tab = QWidget()
        self._aquisition_tab_layout = QVBoxLayout()
        self.aquisition_tab.setLayout(self._aquisition_tab_layout)
        self.tabs.addTab(self.aquisition_tab, 'Aquisition')

        # self.layout().addWidget(btn)



        #/////// analysis tab /////////
        # adding the plotting panel
        self.plotting_panel = VHGroup('Visualize image profile', orientation='G')
        self._analysis_layout.addWidget(self.plotting_panel.gbox)

        # experimental for adding plot canvas

        self.selector = LayerSelector(self.viewer)
        self.plotter = ProfilePlotter(self.viewer, self.selector)
        self.plotting_panel.glayout.addWidget(self.plotter, 0, 0, 1, 1)

        # add button for testing
        self.btn = QPushButton("Click me!")
        self.plotting_panel.glayout.addWidget(self.btn, 0, 0, 1, 2)
        # self.btn.clicked.connect(self._on_click)




        # adding the Analysis panel
        self.analysis_panel = VHGroup('Analysis', orientation='G')
        self._analysis_layout.addWidget(self.analysis_panel.gbox)
        self.normalize_btn = QPushButton("Normalize Image")
        self.analysis_panel.glayout.addWidget(self.normalize_btn, 0, 0, 1, 2)







# Add at the end the connections
        self.add_connections()

# define the callbacks function

    def add_connections(self):
         """Add callbacks"""
         self.btn.clicked.connect(self._on_click_test_btn)



# define teh logic for each button
    def _on_click_test_btn(self):
        print("napari has", len(self.viewer.layers), "layers_lalala")








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