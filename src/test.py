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
# my_file  = "/Users/rubencito/Library/CloudStorage/OneDrive-UniversitaetBern/Bern/Odening_lab/OMAAS/test_data/for_APD/20230504_17h-00m-43.sif"
# viewer.open(my_file, plugin= "napari-sif-reader")

# my_file = "/Users/rubencito/Library/CloudStorage/OneDrive-UniversitaetBern/Bern/Odening_lab/OMAAS/test_data/for_APD/average_stack.tif"
# my_file = "/Users/rubencito/Library/CloudStorage/OneDrive-UniversitaetBern/Bern/Odening_lab/OMAAS/test_data/for_APD/inv_normalize_stack.tif"
my_file = "/Users/rubencito/Library/CloudStorage/OneDrive-UniversitaetBern/Bern/Odening_lab/OMAAS/test_data/for_APD/inv_normalize_stack_filtered.tif"
viewer.open(my_file, colormap = "turbo")

# viewer.layers.selection.active.visible = False

# ##### invert and normalize data #####

# o.inv_and_norm_data_btn.click()
# viewer.layers.selection.active.visible = False
# viewer.layers.select_previous()
# viewer.layers.selection.active.visible = False
# viewer.layers.select_next()

# add shape

# viewer.add_shapes(data = [np.array([[120.6049183 , 512.25410934],
#         [120.6049183 , 540.5934153 ],
#         [145.54350755, 540.5934153 ],
#         [145.54350755, 512.25410934]])], name="shape_20230504_17h-00m-43")

# viewer.layers[0].data = viewer.layers[0].data[:60]

# make selections in the selectors

# o.listImagewidget.item(0).setSelected(True)
# o.listShapeswidget.item(0).setSelected(True)
# # plot
# o.plot_profile_btn.click()

# # average trace

# # viewer.layers.select_previous()
# o.preview_AP_splitted_btn.click()
# # viewer.layers.selection.active.visible = False
# o.create_average_AP_btn.click()
# # viewer.layers.selection.active.visible = False

# # # # filter trace + segment

# # o.apply_spat_filt_btn.click()
# # viewer.layers.selection.active.visible = False
# # o.apply_temp_filt_btn.click()
# # viewer.layers.selection.active.visible = False
# o.apply_segmentation_btn.click()
# # # # o.spat_filter_types.setCurrentText("Bilateral")


# # # # make activation map

# o.listImagewidget.item(0).setSelected(False)
# o.listImagewidget.item(2).setSelected(True)

# # maps_values =  [10, 25, 50, 75, 90]
# # for value in maps_values:
# #     o.slider_APD_map_percentage.setValue(value)
# #     o.make_maps_btn.click()

# o.toggle_map_type.setChecked(True)  
# o.make_interpolation_check.setChecked(True)
# o.make_maps_btn.click()

# o.make_maps_btn.click()

# make APD map

# o.listImagewidget.item(0).setSelected(False)
# o.listImagewidget.item(1).setSelected(True)

# o.make_maps_btn.click()




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