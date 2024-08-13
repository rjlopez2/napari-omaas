import napari
import napari_omaas
from napari_omaas import utils
import numpy as np
from napari_omaas.utils import CustomException
import sys




viewer = napari.Viewer()
o = napari_omaas.OMAAS(viewer)
viewer.window.add_dock_widget(o, area='right')
# my_file  = "/Users/rubencito/Library/CloudStorage/OneDrive-UniversitaetBern/Bern/Odening_lab/OMAAS/test_data/for_APD/20230504_17h-00m-43.sif"
# my_file  = "/Users/rubencito/Library/CloudStorage/OneDrive-UniversitaetBern/Bern/Odening_lab/OMAAS/test_data/for_APD/2_5Hz/20233011_15h-13m-56/analysis/20233011_15h-13m-56_Crop_clip_Inv_GloNor_FiltMedian_MednFilt5_FiltButterworth_cffreq25_ord5_fps356_Ave_NullBckgrnd_ActMap_InterpF.tif"
# viewer.open(path=my_file, colormap = "turbo")
my_file  = [
    "/Users/rubencito/Library/CloudStorage/OneDrive-UniversitaetBern/Bern/Odening_lab/OMAAS/test_data/for_APD/2_5Hz/20233011_15h-13m-56/analysis/20233011_15h-13m-56_Crop_clip_Inv_GloNor_FiltMedian_MednFilt5_FiltButterworth_cffreq25_ord5_fps356_Ave_NullBckgrnd_ActMap_InterpF.tif",
    "/Users/rubencito/Library/CloudStorage/OneDrive-UniversitaetBern/Bern/Odening_lab/OMAAS/test_data/for_APD/2_5Hz/20233011_15h-13m-56/analysis/20233011_15h-13m-56_Crop_clip_Inv_GloNor_FiltMedian_MednFilt5_FiltButterworth_cffreq25_ord5_fps356_Ave_NullBckgrnd_APDMap25_InterpF.tif",
    "/Users/rubencito/Library/CloudStorage/OneDrive-UniversitaetBern/Bern/Odening_lab/OMAAS/test_data/for_APD/2_5Hz/20233011_15h-13m-56/analysis/20233011_15h-13m-56_Crop_clip_Inv_GloNor_FiltMedian_MednFilt5_FiltButterworth_cffreq25_ord5_fps356_Ave_NullBckgrnd_APDMap75_InterpF.tif",
    "/Users/rubencito/Library/CloudStorage/OneDrive-UniversitaetBern/Bern/Odening_lab/OMAAS/test_data/for_APD/2_5Hz/20233011_15h-13m-56/analysis/20233011_15h-13m-56_Crop_clip_Inv_GloNor_FiltMedian_MednFilt5_FiltButterworth_cffreq25_ord5_fps356_Ave_NullBckgrnd_APDMap90_InterpF.tif",
            ]

[viewer.open(path=path, colormap = "turbo") for path in my_file]



# my_file  = "/Users/rubencito/Library/CloudStorage/OneDrive-UniversitaetBern/Bern/Odening_lab/OMAAS/test_data/for_APD/2_5Hz/20233011_15h-13m-56"
# viewer.open(path=my_file, plugin= "napari-omaas")

# my_shape = [np.array([[-25.72179151, 368.84419005],
#             [-25.72179151,  71.75752114],
#             [274.72178891,  71.75752114],
#             [274.72178891, 368.84419005]])]

# viewer.add_shapes(my_shape)
# o.rotate_l_crop.setChecked(True)
# o.crop_from_shape_btn.click()

# del viewer.layers[0]
# del viewer.layers[0]


o.tabs.setCurrentIndex(3)
o.mapping_tabs.setCurrentIndex(1)


o.map_imgs_selector.setCurrentIndex(0)
o.plot_curr_map_btn.click()

o.preview_erode_btn.click()

# o.plot_curr_map_btn.click()


napari.run()