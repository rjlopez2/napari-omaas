# from skimage import data
import napari
import napari_omaas
from napari_omaas import utils
import numpy as np
import os
from collections import OrderedDict
from warnings import warn
from napari.layers import Shapes, Image, Labels
from napari.settings import get_settings
from qtpy.QtWidgets import QApplication

settings = get_settings()
settings.application.playback_fps = 30

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
# viewer.open(path=my_file, plugin= "napari-omaas")

#########################################
############### open file ###############
#########################################

# my_file = "/Users/rubencito/Library/CloudStorage/OneDrive-UniversitaetBern/Bern/Odening_lab/OMAAS/test_data/for_APD/average_stack.tif"
# my_file = "/Users/rubencito/Library/CloudStorage/OneDrive-UniversitaetBern/Bern/Odening_lab/OMAAS/test_data/for_APD/inv_normalize_stack.tif"
# my_file = "/Users/rubencito/Library/CloudStorage/OneDrive-UniversitaetBern/Bern/Odening_lab/OMAAS/test_data/for_APD/inv_normalize_stack_filtered.tif"
# my_file = r"C:\Users\lopez\Desktop\test_20240327\20240327_13h-40m-03"
# my_file = "\\\\PC160\\GroupOdening\\OM_data_Bern\\raw_data\\2024\\20240619\\Videosdata\\experiment\\data\\pacing\\drugs\\10nM\\20240619_17h-14m-52"
# my_file = r"\\PC160\GroupOdening\OM_data_Bern\raw_data\2024\20240619\Videosdata\experiment\data\pacing\drugs\1nM\20240619_17h-07m-45"
my_file = r"\\PC160\GroupOdening\OM_data_Bern\raw_data\2024\20240619\Videosdata\experiment\data\pacing\drugs\1nM\20240619_17h-10m-30"
# my_file = r"D:\OM_data\raw_data\20231019\Videosdata\20231019_14h-08m-54s.sif"
viewer.open(path=my_file, plugin= "napari-omaas")


#########################################
########### settings & steps ############
#########################################
root_path = os.path.abspath(os.path.join(my_file, 
                                         "..",
                                         "..",
                                         ".."))

view1_crop_shape_path = os.path.abspath(os.path.join(root_path, '..', '..', 
                                                'analysis',
                                                'settings',
                                                'ROI_crop_view1.csv'
                                               ))

ROIS_APD_path = os.path.abspath(os.path.join(root_path, '..', '..', 
                                            'analysis',
                                            'settings',
                                            'view1',
                                            'ROIs', 
                                            'ROIS_APD_analysis_20240327.csv'))

manual_mask_for_segmentation_path_2 = os.path.abspath(os.path.join(root_path, '..', '..', 
                                                                'analysis',
                                                                'settings',
                                                                'view1',
                                                                'segmentation_mask', 
                                                                'mask_v1_2.tif'))

my_ref_shape = [np.array([[143.42264109, 115.95655838],
        [143.42264109, 144.24370889],
        [171.67219674, 144.24370889],
        [171.67219674, 115.95655838]])]

settings = {
    'slider_APD_detection_threshold' : 0.0172,
    'spa_filt_type': 'Median', # valid spatial filters are: ["Gaussian", "Box", "Laplace", "Median", "Bilateral"]
    'gauss_sigma_filt': 1,
    'spac_kernel_filt_size' : 9,
    'butt_order_filt': 5,
    'butt_cutoof_freq_filt': 25,
    'MobStab_ck': 7,
    'MobStab_RefFrame': 50, # NOTE: this is in frame (no ms)
    # 'MobStab_RefFrame': 71, # for new image on date 20240619
    'MobStab_PreSmoSpat': 1,
    'MobStab_PreSmoSTem': 1,
    'normalization_method': 'glob_max', # values allowed are 'loc_max', 'glob_max', 'slide_window'
    'clipping_values': (209, 6722), # NOTE: this is in ms.
    # 'clipping_values': (267, 1588), # for new image on date 20240619
    'ref_shape':my_ref_shape,
    'invert_ratio': True,
    'paths_for_compute_APDs_from_ROIS': ROIS_APD_path,
    # 'ROIS_features': {"features" : features, 
    #                   "text": text, 
    #                   "edge_color_cycle" : edge_color_cycle, 
    #                   "edge_color":'region'},
    'auto_segmentation' : False,
    # 'use_external_mask': manual_mask_for_segmentation_path,
    'use_external_mask': manual_mask_for_segmentation_path_2, # use another mask for the drugs treatment (heart moved)
    'map_values' : [25, 75, 90],
    # 'map_values' : [ 90],
    'save_settings': True,
    'çrop_view_paths': OrderedDict(
        {
            'view1' : view1_crop_shape_path,
            # 'view1' : r"C:\Users\lopez\Desktop\testCROP.csv",
            # 'view2' : view2_crop_shape_path,
            # 'view3' : view3_crop_shape_path,
            # 'view4' : view4_crop_shape_path
        }
        )
}

steps = [
            "crop_view",
            "spplit_channels",
            "motion_correction", 
            "normalize_img",
            "compute_ratio", 
            # "clipping_trace", 
            # "average_trace", 
            "apply_spa_filter", 
            "apply_tem_filter", 
            "segment_image",
            "compute_ROI_based_APD",
            # "export_APDFromROIs_results",
            #"compute_maps", 
            #"export_map_results",            
           ]


#########################################
############## crop views ###############
#########################################

print("############ Cropping View ############")
if 'çrop_view_paths' in settings:
    for view_name, ROI_path in settings['çrop_view_paths'].items():
        try:
            viewer.open(ROI_path, name=view_name)
            o.rotate_l_crop.setChecked(True)
            o.crop_from_shape_btn.click()
            del viewer.layers[-2]
            del viewer.layers[-2]
        except Exception as e:
            warn (f"You have the following error @ steps 'crop_view': --->> {e} <----")


#########################################
############ split channels #############
#########################################

if 'spplit_channels' in steps:
    print("############ splitting channels ############")

    o.splt_chann_btn.click()
    # collect the resulting images
    splitted_imgs = [layer for layer in viewer.layers if isinstance(layer, Image) and "Ch" in layer.name]


#########################################
########### Motion correction ###########
#########################################

# if 'motion_correction' in steps:
#     print("############ applying  motion correction ############")

#         # set parameters for motion correction
#     if 'MobStab_ck' in settings:
#         print(f"########## Changing contrast kernel for Motions stabilization to value: '{settings['MobStab_ck']}'. ##########")
#         o.c_kernels.setValue(settings['MobStab_ck']) # customize contrast kernel
#     if 'MobStab_PreSmoSTem' in settings:
#         print(f"########## Changing temp pre-smoothing for Motions stabilization to value: '{settings['MobStab_PreSmoSTem']}'. ##########")
#         o.pre_smooth_temp.setValue(settings['MobStab_PreSmoSTem'])
#     if 'MobStab_PreSmoSpat' in settings:
#         print(f"########## Changing Spat pre-smoothing for Motions stabilization to value: '{settings['MobStab_PreSmoSpat']}'. ##########")
#         o.pre_smooth_spat.setValue(settings['MobStab_PreSmoSpat'])        
#     if 'MobStab_RefFrame' in settings:
#         print(f"########## Changing Ref frame for Motions stabilization to value: '{settings['MobStab_RefFrame']}'. ##########")
#         o.ref_frame_val.setText(f"{settings['MobStab_RefFrame']}")

#     for channel in splitted_imgs:
#         viewer.layers.selection.active = channel
#         o.apply_optimap_mot_corr_btn.click()

#     splitted_imgs = [layer for layer in viewer.layers if isinstance(layer, Image) and  "MotStab" in layer.name ]


#########################################
########### Normalize images ############
#########################################

if 'normalize_img' in steps and 'motion_correction' in steps:
    print("############ normalizing splitted channels ############")

    if 'normalization_method' in settings:
        # values are:
        # respective orders are: 
        norm_methods = {'loc_max':"Normalize (loc max)",
                        'slide_window': "Normalize slide window",
                        'glob_max': "Normalize (global)"}
        
        if settings['normalization_method'] in norm_methods:
            o.data_normalization_options.setCurrentText(norm_methods[settings['normalization_method']])
            print(f"############ Setting normalization method to: { norm_methods[settings['normalization_method']] }. ############")
        else:
            warn(f"Your current normalization method: '{settings['normalization_method']}' is not valid. Valid methods are: {[f'{val},  ' for val in norm_methods.keys()]}. Using default normalization method.")
            o.data_normalization_options.setCurrentIndex(0)

    for channel in splitted_imgs:
        viewer.layers.selection.active = channel
        o.apply_normalization_btn.click()
        # QApplication.instance().processEvents()
    splitted_imgs = [layer for layer in viewer.layers if isinstance(layer, Image) and "Ch" in layer.name and "Nor" in layer.name]


#########################################
############# compute ratio #############
#########################################

# if 'compute_ratio' in steps:
#     print("############ computing ratio ############")
#     o.Ch0_ratio.setCurrentText(splitted_imgs[0].name)
#     o.Ch1_ratio.setCurrentText(splitted_imgs[1].name)
    
#     if 'invert_ratio' in settings:
#         o.is_ratio_inverted.setChecked(settings['invert_ratio'])
    
#     o.compute_ratio_btn.click()
#     if o.is_ratio_inverted.isChecked():
#         o.is_ratio_inverted.setChecked(False)







napari.run()  # start the "event loop" and show the viewer
