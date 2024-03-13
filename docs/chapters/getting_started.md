
# Installation

You can install `napari-omaas` via [pip]:

    pip install napari-omaas



To install the latest development version (recommended) :

    pip install git+https://github.com/rjlopez2/napari-omaas.git

# Usage

This plugin can read images generated with Andor Technologies cameras. It has been currently tested on Zyla cameras. Just drag and drop an image to the napari GUI, and the image will display. Alternatively, you can programmatically load/read the image within a notebook.

```python
import napari
file = "path/to/my/file/my_image.sif"
viewer = napari.Viewer()
viewer.open(path=file, plugin="napari-omaas", name = "my_image")
```

to display the metadata use the standard call to the corresponding layer:

```python
viewer.layers['my_image'].metadata
```

In addition to opening this specific image format (.sif), it allows the users to perform some basic operations and visualization on images, such as normalization, temporal/spatial filters, motion tracking/compenstaion, plot profile, etc.
# Examples

The following example ilustrate how to perform normalization (pixelwise) on a time serie image and plot its 2d profile along the t dimension withing the average data from the ROI selected.

![](https://github.com/rjlopez2/napari-omaas/blob/documentation/example_imgs/Oct-31-2023%2016-45-55_plot_profile.gif?raw=true)


The next example shows how to compute action potential duration in the same image stack.

![](https://github.com/rjlopez2/napari-omaas/blob/documentation/example_imgs/Oct-31-2023%2016-49-02_APD_analysis.gif?raw=true)

# Roadmap

This plugin is composed of two major components: **analysis** and **acquisition**.

Bellow is a list of some features this pluggin aims to do.

## Analysis Features
    
- [x] Read sif files from Andor Technologies.
- [x] Display time profile of ROIs on image sequences.
- [x] Normalize images.
    - [x] Perform peak analysis of action potential / Calcium traces.
    - [x] Add motion correction.
    - [x] APD analysis.
    - [ ] Create activation maps.
    - [ ] Segment images and align heart ROIs.
- [x] Export results and analysis log.

## Acquisition Features

- [ ] Control Zyla camera for the acquisition of data
    - [ ] test using the PYME module
- [ ] Real-time analysis(?)



