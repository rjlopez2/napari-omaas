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
# my_file  = "/Users/rubencito/Library/CloudStorage/OneDrive-UniversitaetBern/Bern/Odening_lab/OMAAS/test_data/for_APD/20230504_17h-00m-43.sif"
# viewer.open(my_file, plugin= "napari-sif-reader")

# my_file = "/Users/rubencito/Library/CloudStorage/OneDrive-UniversitaetBern/Bern/Odening_lab/OMAAS/test_data/for_APD/average_stack.tif"
# my_file = "/Users/rubencito/Library/CloudStorage/OneDrive-UniversitaetBern/Bern/Odening_lab/OMAAS/test_data/for_APD/inv_normalize_stack.tif"
# my_file = "/Users/rubencito/Library/CloudStorage/OneDrive-UniversitaetBern/Bern/Odening_lab/OMAAS/test_data/for_APD/inv_normalize_stack_filtered.tif"
my_file = "/Users/rubencito/Library/CloudStorage/OneDrive-UniversitaetBern/Bern/Odening_lab/OMAAS/test_data/for_APD/4_Hz/20233011_15h-12m-06"
viewer.open(my_file, colormap = "turbo")

napari.run()  # start the "event loop" and show the viewer
