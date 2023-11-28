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
# widg = napari_omaas.OMAAS.(viewer=viewer)  # or any other QWidget
viewer.window.add_dock_widget(o, area='right')
my_file  = "/Users/rubencito/Library/CloudStorage/OneDrive-UniversitaetBern/Bern/Odening_lab/OMAAS/test_data/for_APD/20230504_17h-00m-43.sif"
viewer.open(my_file, plugin= "napari-sif-reader")

# my_file = "/Users/rubencito/Library/CloudStorage/OneDrive-UniversitaetBern/Bern/Odening_lab/OMAAS/test_data/for_APD/average_stack.tif"
# my_file = "/Users/rubencito/Library/CloudStorage/OneDrive-UniversitaetBern/Bern/Odening_lab/OMAAS/test_data/for_APD/inv_normalize_stack.tif"
# my_file = "/Users/rubencito/Library/CloudStorage/OneDrive-UniversitaetBern/Bern/Odening_lab/OMAAS/test_data/for_APD/inv_normalize_stack_filtered.tif"
# viewer.open(my_file, colormap = "turbo")


# ##### invert and normalize data #####

o.inv_and_norm_data_btn.click()

# add shape

viewer.add_shapes(data = [np.array([[120.6049183 , 512.25410934],
        [120.6049183 , 540.5934153 ],
        [145.54350755, 540.5934153 ],
        [145.54350755, 512.25410934]])], name="shape_20230504_17h-00m-43")

# make selections in the selectors
o.listImagewidget.item(2).setSelected(True)
o.listShapeswidget.item(0).setSelected(True)
# plot
o.plot_profile_btn.click()
# average trace
# o.create_average_AP_btn.click()


# ###############################################################
# # automatic load, invert and normalize the signal and
# # then make the plotting
# ###############################################################

# # load a spool file
# my_file = "/Users/rubencito/Library/CloudStorage/OneDrive-UniversitaetBern/Bern/Odening_lab/OMAAS/test_data/for_APD/2_5Hz/20230710_12h-09m-00"
# o.dir_box_text.setText(my_file)
# o.load_spool_dir_btn.click()
# # invert and normalize signal
# o.inv_and_norm_data_btn.click()


# # add shape

# viewer.add_shapes(
#     data = [np.array([[174.03193643, 460.55054341],
#         [174.03193643, 488.88984937],
#         [198.97052568, 488.88984937],
#         [198.97052568, 460.55054341]])],
#         name = "shape_20230710_12h-09m-0"
# )
# # make selections in the selectors
# o.listImagewidget.item(2).setSelected(True)
# o.listShapeswidget.item(0).setSelected(True)
# # plot
# o.plot_profile_btn.click()


# viewer.open_sample("napari-omaas", "heartsample")

napari.run()  # start the "event loop" and show the viewer

print("end")