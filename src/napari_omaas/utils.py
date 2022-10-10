from qtpy import QtCore, QtGui

__all__ = ('get_valid_image_layers', 'SelectorListModel')


# functions
def get_valid_image_layers(layer_list):
    """
    Extract napari images layers of 3 or more dimensions from the input list.
    """
    out = [layer for layer in layer_list if ((layer._type_string == 'image' and layer.data.ndim >= 3) | (layer._type_string == 'shapes'))] # include the shape may make no sense now
    out.reverse()
    return out


class SelectorListModel(QtGui.QStandardItemModel):
    """Subclass of QtGui.QStandardItemModel.

    Automatically builds from a list of QtGui.QStandardItems or derivatives.
    """
    def __init__(self, items=None):
        super().__init__()
        if items:
            for item in items:
                self.appendRow(item)

    def get_checked(self):
        """
        Return all items with state QtCore.Qt.Checked.
        """
        checked = []
        for index in range(self.rowCount()):
            item = self.item(index)
            if item.checkState() == QtCore.Qt.Checked:
                checked.append(item.layer)
        return checked