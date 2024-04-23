# from skimage import data
import napari
import napari_omaas
from napari_omaas import utils
import numpy as np
from napari.settings import get_settings
import os
from glob import glob
import pickle


# set pülayback setting to render at 100 fps
settings = get_settings()
settings.application.playback_fps = 100

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
# my_file = "/Users/rubencito/Library/CloudStorage/OneDrive-UniversitaetBern/Bern/Odening_lab/OMAAS/test_data/for_APD/inv_normalize_stack_filtered.tif"


########## load a spool file ##########
# my_file = "/Users/rubencito/Library/CloudStorage/OneDrive-UniversitaetBern/Bern/Odening_lab/OMAAS/test_data/for_APD/2_5Hz/20233011_15h-13m-56"
# my_file = "/Users/rubencito/Library/CloudStorage/OneDrive-UniversitaetBern/Bern/Odening_lab/OMAAS/test_data/for_motion_test/20233011/red/20233011_14h-26m-21"
# my_file = "/Users/rubencito/Library/CloudStorage/OneDrive-UniversitaetBern/Bern/Odening_lab/OMAAS/test_data/for_APD/2_5Hz/20230710_12h-09m-00"
# img, info = utils.return_spool_img_fun(my_file)
# view1 = np.rot90(img[:,:238, 75:350], axes = (1, 2))
# viewer.add_image(view1, metadata = info)
# o.dir_box_text.setText(my_file)
# o.load_spool_dir_btn.click()


# viewer.open(my_file, colormap = "turbo")

# viewer.layers.selection.active.visible = False

##### invert and normalize data #####

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

# viewer.add_shapes(data = [np.array([[73.12689301, 58.31786751],
#         [73.12689301, 86.65717347],
#         [98.06548226, 86.65717347],
#         [98.06548226, 58.31786751]])], name="shape_20230504_17h-00m-43")

# viewer.layers[0].data = viewer.layers[0].data[:60]

# make selections in the selectors

# o.listImagewidget.item(2).setSelected(True)
# o.listShapeswidget.item(0).setSelected(True)
# # plot
# o.plot_profile_btn.click()

# viewer.layers.select_previous()
# o.export_image_btn.click()

# # average trace

# # viewer.layers.select_previous()
# o.preview_AP_splitted_btn.click()
# # viewer.layers.selection.active.visible = False
# o.create_average_AP_btn.click()
# # viewer.layers.selection.active.visible = False

# # # filter trace + segment
# viewer.layers.select_previous()
# o.apply_spat_filt_btn.click()
# viewer.layers.selection.active.visible = False
# o.apply_temp_filt_btn.click()
# viewer.layers.selection.active.visible = False
# o.apply_segmentation_btn.click()
# o.spat_filter_types.setCurrentText("Bilateral")


# # make selections in the selectors

# o.listImagewidget.item(4).setSelected(True)
# o.listShapeswidget.item(0).setSelected(True)
# # plot
# o.plot_profile_btn.click()


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
# # add shape
# my_shape = [np.array([[120.65846541,  99.16317271],
#                               [120.65846541, 167.82143873],
#                               [192.55532888, 167.82143873],
#                               [192.55532888,  99.16317271]])]
my_shape = [np.array([[ 92.22038205, 131.41779359],
            [ 92.22038205, 155.98907334],
            [113.41678112, 155.98907334],
            [113.41678112, 131.41779359]])]

# my_shape = [np.array([[ 12.64035761, 217.60220196],
#                     [ 12.64035761, 218.33551779],
#                     [ 13.39946362, 218.33551779],
#                     [ 13.39946362, 217.60220196]])]
viewer.add_shapes(
    data = my_shape,
#     data = [np.array([[149.75387022, 133.80377879],
#         [149.75387022, 134.23215058],
#         [150.16793055, 134.23215058],
#         [150.16793055, 133.80377879]])],
        name = "my_shape"
)

dir_70mmHg_protocol = "/Users/rubencito/Library/CloudStorage/OneDrive-UniversitaetBern/Bern/Odening_lab/OMAAS/test_data/for_maps/results"
results_folders = [ my_dir for my_dir in glob(os.path.join(dir_70mmHg_protocol, "**", "*"), recursive=True) if "averaged" in my_dir and my_dir.endswith("npy")]
results_folders_metadata = [ my_dir for my_dir in glob(os.path.join(dir_70mmHg_protocol, "**", "*"), recursive=True) if my_dir.endswith("pkl")]
# o.make_maps_btn.click()
metadata = []
for path in results_folders_metadata:
    with open(path, "rb") as f:
        metadata.append(pickle.load(f))

# [viewer.open(path=path, metadata= meta, colormap="turbo") for path, meta in zip(results_folders, metadata)]
viewer.open(path=results_folders[0], metadata= metadata[0], colormap="turbo")



# ###############################################################
# # automatic load, invert and normalize the signal and
# # then make the plotting
# ###############################################################

# # load a spool file
# my_file = "/Users/rubencito/Library/CloudStorage/OneDrive-UniversitaetBern/Bern/Odening_lab/OMAAS/test_data/for_maps/20240327_13h-40m-03"
# my_file = "/Users/rubencito/Library/CloudStorage/OneDrive-UniversitaetBern/Bern/Odening_lab/OMAAS/test_data/for_motion_test/mix/20240229_12h-00m-09"
# o.dir_box_text.setText(my_file)
# o.load_spool_dir_btn.click()
# # invert and normalize signal
# o.inv_and_norm_data_btn.click()
# img, info = utils.return_spool_img_fun(my_file)
# # crop/rotate if needed
# img_crop = np.rot90(img[:, :, 22:345], axes = (1, 2))
# viewer.add_image(img_crop, metadata= info, name = os.path.basename(my_file), colormap="turbo")

# #################
# # debugging map #
# #################
# my_file = "/Users/rubencito/Library/CloudStorage/OneDrive-UniversitaetBern/Bern/Odening_lab/OMAAS/sandbox/analysis/output_results/20240207_13h-50m-42_alternaratio_MotStab_LocNor_clip_Ave_FiltGaussian_KrnlSiz5_Sgma1.0_FiltButterworth_cffreq35_ord5_fps157_NullBckgrnd.npy"
# viewer.open(my_file, colormap = "turbo")


# make selections in the selectors
o.listImagewidget.item(0).setSelected(True)
o.listShapeswidget.item(0).setSelected(True)
# # plot
o.plot_profile_btn.click()

# # Preview traces for averaging
o.preview_AP_splitted_btn.click()

# # change current tab to "Mapping" tab
o.tabs.setCurrentIndex(4)
o.toggle_map_type.setChecked(True)

# #################

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



################################################
######## workflow for generation of maps #######
################################################


##### load data #####
# my_file = "/Users/rubencito/Library/CloudStorage/OneDrive-UniversitaetBern/Bern/Odening_lab/OMAAS/test_data/for_APD/2_5Hz/20233011_15h-13m-56"
# img, info = utils.return_spool_img_fun(my_file)
# view1 = np.rot90(img[:,:238, 75:350], axes = (1, 2))
# viewer.add_image(view1, metadata = info)
# viewer.add_image(view1[0,...], name = "view1_1frame", metadata = info)

# o.crop_from_shape_btn.click()

# ##### invert and normalize data #####
# o.inv_and_norm_data_btn.click()

# ##### add shape #####
# viewer.add_shapes(data = 
#                   [np.array([[133.75316094, 132.63923711],
#                              [133.75316094, 140.84176856],
#                              [141.04430001, 140.84176856],
#                              [141.04430001, 132.63923711]])], 
#                   name="20233011_15h-13m-56_shape")

# ##### select the current image #####
# viewer.layers.select_previous()

# ##### make selections in the selectors #####

# o.listImagewidget.item(2).setSelected(True)
# o.listShapeswidget.item(0).setSelected(True)

# ##### plot profile on inverted image #####
# o.plot_profile_btn.click()

# ##### clip image #####
# o.double_slider_clip_trace.setValue((287.12, 2247))
# o.clip_label_range.setChecked(True)

# o.clip_trace_btn.click()

# # make new selections in the selectors
# o.listImagewidget.item(2).setSelected(False)
# o.listImagewidget.item(3).setSelected(True)

# # re-plot
# o.plot_profile_btn.click()
# o.plot_profile_btn.click()

# # change kernel filter to 10 for instance
# o.filt_kernel_value.setValue(10)
# # apply spatial filter
# o.apply_spat_filt_btn.click()

# # apply temp filter to current image
# o.butter_cutoff_freq_val.setValue(25)
# o.apply_temp_filt_btn.click()


# # make new selections in the selectors
# # o.listImagewidget.item(3).setSelected(False)
# o.listImagewidget.item(5).setSelected(True)

# # re-plot
# o.plot_profile_btn.click()
# o.plot_profile_btn.click()

# # apply segmentation
# o.apply_segmentation_btn.click() # using default method

# # make new selections in the selectors
# o.listImagewidget.item(3).setSelected(False)
# o.listImagewidget.item(5).setSelected(False)
# o.listImagewidget.item(6).setSelected(True)

# # re-plot
# o.plot_profile_btn.click()
# o.plot_profile_btn.click()


# # Preview traces for averaging
# o.preview_AP_splitted_btn.click()


# #  average traces
# o.create_average_AP_btn.click()

# # make new selections in the selectors
# o.listImagewidget.item(6).setSelected(False)
# o.listImagewidget.item(7).setSelected(True)
# # re-plot
# o.plot_profile_btn.click()
# o.plot_profile_btn.click()

# # # clip again the averaged trace
# # # o.double_slider_clip_trace.value()
# # o.double_slider_clip_trace.setValue((77.7, 323.15))
# # o.clip_label_range.setChecked(True)
# # o.clip_trace_btn.click()

# # # make new selections in the selectors
# # o.listImagewidget.item(7).setSelected(False)
# # o.listImagewidget.item(8).setSelected(True)
# # # re-plot
# # o.plot_profile_btn.click()
# # o.plot_profile_btn.click()


# # set all but the last image  hidden
# for layer in viewer.layers:
#     layer.visible = False

# viewer.layers[-1].visible = True

# # select current images in the main layer selector 
# viewer.layers.selection.active = viewer.layers[-1]


# # # create map
# # value = 10

# o.toggle_map_type.setChecked(True) # set to APD maps
# # o.slider_APD_map_percentage.setValue(value)
# o.make_maps_btn.click()







napari.run()  # start the "event loop" and show the viewer

# # print("end")