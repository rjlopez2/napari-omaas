# from skimage import data
import napari
import napari_omaas
from napari_omaas import utils
import numpy as np

# import myModule
# import importlib
# importlib.reload(myModule)
# from myModule import *

# t = 70
# print(t)
# viewer = napari.view_image(data.cells3d(), channel_axis=1, ndisplay=3)

# viewer.open("12h-49m-26s.sif", plugin="napari-"  )
viewer = napari.Viewer()
o = napari_omaas.OMAAS(viewer)
viewer.window.add_dock_widget(o, area='right')
# o.tabs.setCurrentIndex(1)
# my_file  = "/Users/rubencito/Library/CloudStorage/OneDrive-UniversitaetBern/Bern/Odening_lab/OMAAS/test_data/for_APD/20230504_17h-00m-43.sif"
my_file  = "/Users/rubencito/Library/CloudStorage/OneDrive-UniversitaetBern/Bern/Odening_lab/OMAAS/test_data/for_APD/2_5Hz/20233011_15h-13m-56"
viewer.open(path=my_file, plugin= "napari-omaas")

my_shape = [np.array([[-25.72179151, 368.84419005],
            [-25.72179151,  71.75752114],
            [274.72178891,  71.75752114],
            [274.72178891, 368.84419005]])]

viewer.add_shapes(my_shape)
o.rotate_l_crop.setChecked(True)
o.crop_from_shape_btn.click()

del viewer.layers[0]
del viewer.layers[0]



# testing new FIR temporal filter

o.temp_filter_types.setCurrentText("FIR")
current_layer = viewer.layers[-1]
# for i in [5, 10, 15, 20, 30, 50, 100]: # this does not change anything
for i in np.arange(0, 1, 0.1).tolist(): # this does not change anything
    viewer.layers.selection.active = current_layer
    # o.butter_order_val.setValue(i)
    o.butter_cutoff_freq_val.setText(str(i))
    o.apply_temp_filt_btn.click()

# viewer.window.remove_dock_widget(o)

# my_file = "/Users/rubencito/Library/CloudStorage/OneDrive-UniversitaetBern/Bern/Odening_lab/OMAAS/test_data/for_APD/average_stack.tif"
# my_file = "/Users/rubencito/Library/CloudStorage/OneDrive-UniversitaetBern/Bern/Odening_lab/OMAAS/test_data/for_APD/inv_normalize_stack.tif"
# my_file = "/Users/rubencito/Library/CloudStorage/OneDrive-UniversitaetBern/Bern/Odening_lab/OMAAS/test_data/for_APD/inv_normalize_stack_filtered.tif"
# my_file = r"C:\Users\lopez\Desktop\test_20240327\20240327_13h-40m-03"
# my_file = r"D:\OM_data\raw_data\20231019\Videosdata\20231019_14h-08m-54s.sif"
# viewer.open(path=my_file, plugin= "napari-omaas")
napari.run()  # start the "event loop" and show the viewer
