from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtWidgets import *

from napari_matplotlib.base import NapariMPLWidget

from .utils import *


__all__ = ('LayerSelector', 'ProfilePlotter')

class LayerSelector(QtWidgets.QListView):
    """Subclass of QListView for selection of 3D/4D napari image layers.

    This widget contains a list of all 4D images layer currently in the napari viewer layers list. All items have a
    checkbox for selection. It is not meant to be directly docked to the napari viewer, but needs a napari viewer
    instance to work.

    Attributes:
        napari_viewer : napari.Viewer
        parent : Qt parent widget / window, default None
    """
    def __init__(self, napari_viewer, model=SelectorListModel(), parent=None):
        super(LayerSelector, self).__init__(parent)
        self.napari_viewer = napari_viewer
        self.setModel(model)
        self.update_model(None)

    def update_model(self, event):
        """
        Update the underlying model data (clear and rewrite) and emit an itemChanged event.
        The size of the widget is adjusted to the number of items displayed.

        :param event: Not used, just for napari event compatibility.
        """
        self.model().clear()
        for layer in get_valid_image_layers(self.napari_viewer.layers):
            item = SelectorListItem(layer)
            self.model().appendRow(item)
        self.setMaximumHeight(
            self.sizeHintForRow(0) * self.model().rowCount() + 2 * self.frameWidth())
        self.model().itemChanged.emit(QtGui.QStandardItem())

class ProfilePlotter(NapariMPLWidget):
    def __init__(self, napari_viewer, selector, options=None):
        super().__init__(napari_viewer)
        self.selector = selector

