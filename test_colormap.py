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

########################################
########## plot trace and crop ########
########################################

# add new shape for potting a profile trace
my_shape= [np.array([[84.56441374,  70.06110599],
                    [84.56441374,  94.16163661],
                    [110.960233  ,  94.16163661],
                    [110.960233  ,  70.06110599]])]

viewer.add_shapes(my_shape)
o.plot_last_generated_img()

# clip the image off the adges (temporal)
o.clip_label_range.setChecked(True)
o.double_slider_clip_trace.setValue((209.95, 1951.97))
o.clip_trace_btn.click()

########################################
########## invert and normalize ##########
########################################

o.data_normalization_options.setCurrentText("Normalize (global)")
# o.apply_normalization_btn.click()
o.inv_and_norm_data_btn.click()
o.plot_last_generated_img()

########################################
########## filter the traces ##########
########################################

# apply spatial filter
o.spat_filter_types.setCurrentText('Median')
o.apply_spat_filt_btn.click()
# apply temporal filter
o.butter_order_val.setValue(5)
o.butter_cutoff_freq_val.setText(str(25))
o.apply_temp_filt_btn.click()
o.plot_last_generated_img()



# o.tabs.setCurrentIndex(3)
# o.mapping_tabs.setCurrentIndex(1)
# viewer.window.remove_dock_widget(o)
napari.run()  # start the "event loop" and show the viewer
