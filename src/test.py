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
my_file  = "/Users/rubencito/Library/CloudStorage/OneDrive-UniversitaetBern/Bern/Odening_lab/OMAAS/test_data/for_APD/20230504_17h-01m-53.sif"
# my_file = "/Users/rubencito/programming_stuff/p_experimental/image_analysis/napari_play/OMAAS/test_data/single_scan/test_ruben/20230413_11h-57m-31s.sif"
viewer.open(my_file, plugin= "napari-sif-reader")
###### invert data #####
inv_data = utils.invert_signal(viewer.layers[-1].data)
##### normalize data #####
norm_data = utils.local_normal_fun(inv_data)
viewer.add_image(norm_data, name=viewer.layers[-1].name  + "_Inv_Norm", colormap= "turbo", metadata= viewer.layers[-1].metadata)

# add shape
viewer.layers["ROI selection"].data = [np.array([[120.6049183 , 512.25410934],
        [120.6049183 , 540.5934153 ],
        [145.54350755, 540.5934153 ],
        [145.54350755, 512.25410934]])]


# viewer.open_sample("napari-omaas", "heartsample")

napari.run()  # start the "event loop" and show the viewer

print("end")