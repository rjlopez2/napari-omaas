
# Installation

To use this plugin you will need first to install napari, a fast, interactive viewer for multi-dimensional images in Python. Find more info about napari usage and installation [here](https://napari.org/stable/).

```{admonition} Important
:class: warning
While not strictly required, it is highly recommended to install this plugin (together with napari and all additional dependencies) into a clean virtual environment using an environment manager like conda or venv.

This should be set up before you install napari and napari-omaas. For example, setting with up a Python {{ python_version }} environment with conda:

{{ conda_create_env }}
```
After you have napari installed in your system, you can install `napari-omaas` via [pip] using the comand line:

```sh
pip install napari-omaas
```

To install the latest development version (recommended) :

```sh
pip install git+https://github.com/rjlopez2/napari-omaas.git
```

## Custom installation:
```{admonition} Note
:class: Note
This is a recommended method we currently use to install napari-omaas and all additional dependencies in a custom environment:
```
1. Download manually or via comand line the `environment_OMAAS_CPU.yml` file from the [OMAAS](https://github.com/rjlopez2/napari-omaas) repository. This repo contain a number of configuration files for our experiments setup.

```sh
curl -O https://raw.githubusercontent.com/rjlopez2/OMAAS/master/setup_files/environments/environment_OMAAS_CPU.yml
```
2. Cretae the environment using the downlowded file:

```sh
conda create -f environment_OMAAS_CPU.yml
```
3. Activate the environment:

```sh
conda activate omaas_base
```

You should now be ready to use napari-omaas.

## Update to the latest version:

1. Uninstall the current version

```sh
pip uninstall napari-omaas
```

2. Install the latest developing version:

```sh
pip install git+https://github.com/rjlopez2/napari-omaas.git
```


# Usage

This plugin can read images generated with Andor Technologies cameras. It has been currently tested on Zyla cameras. Just drag and drop an image (.sif format or spooling folder) to the napari GUI, and the image will display. Alternatively, you can programmatically load/read the image within a notebook.

## Launch the application


### Via command line

<details>
<summary>Click to expand</summary>

First activate your environment and then launch the application with the following command:

```sh
conda activate omaas_base
napari -w napari-omaas
```
A new window should appera showing the Napari viewer with the `napari-omaas` pluging attached.
</details>

### Via Jupyter-Notebook or python script

<details>
<summary>Click to expand</summary>

```python
import napari
import napari_omaas

viewer = napari.Viewer()
o = napari_omaas.OMAAS(viewer)
viewer.window.add_dock_widget(o, area='right')

file = "path/to/my/file/my_image.sif"
viewer.open(path=file, plugin="napari-omaas", name = "my_image")
```
to display the metadata use the standard call to the corresponding layer:

```python
viewer.layers['my_image'].metadata
```
</details>
<br>

You can also perform some basic operations on images, such as normalization, temporal/spatial filters, plot profile, but also apply more advanced image processing methods such as motion tracking/compensation, etc.
<br><br>
# Examples

The following example ilustrate how to perform normalization (pixelwise) on a time serie image and plot its 2d profile along the t dimension withing the average data from the ROI selected.

![](https://github.com/rjlopez2/napari-omaas/blob/documentation/example_imgs/Oct-31-2023%2016-45-55_plot_profile.gif?raw=true)


The next example shows how to compute action potential duration in the same image stack.

![](https://github.com/rjlopez2/napari-omaas/blob/documentation/example_imgs/Oct-31-2023%2016-49-02_APD_analysis.gif?raw=true)

# Roadmap

This plugin is composed of two major components: **analysis** and **acquisition**.

Bellow is a list of some features this plugin aims to do.

## Analysis Features
    
- [x] Read sif files from Andor Technologies.
- [x] Display time profile of ROIs on image sequences.
- [x] Normalize images.
    - [x] Perform peak analysis of action potential / Calcium traces.
    - [x] Add motion correction.
    - [x] APD analysis.
    - [x] Create activation maps.
    - [x] Segment images.
    - [ ] Automatic crop and alignment of heart ROIs.
- [x] Export results, metadata and analysis log.

## Acquisition Features

- [ ] Control Zyla camera for the acquisition of data
    - [ ] test using the PYME module
- [ ] Real-time analysis(?)


[pip]: https://pypi.org/project/pip/
