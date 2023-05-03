# napari-omaas

[![License BSD-3](https://img.shields.io/pypi/l/napari-omaas.svg?color=green)](https://github.com/rjlopez2/napari-omaas/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-omaas.svg?color=green)](https://pypi.org/project/napari-omaas)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-omaas.svg?color=green)](https://python.org)
[![tests](https://github.com/rjlopez2/napari-omaas/workflows/tests/badge.svg)](https://github.com/rjlopez2/napari-omaas/actions)
[![codecov](https://codecov.io/gh/rjlopez2/napari-omaas/branch/main/graph/badge.svg)](https://codecov.io/gh/rjlopez2/napari-omaas)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-omaas)](https://napari-hub.org/plugins/napari-omaas)

**napari-OMAAS stands for Optical Mapping Acquisition and Analysis Software for panoramic heart imaging**

This plugin intends to be an analysis and acquisition tool for optical mapping systems from panoramic imaging of intact hearts' potentiometric or Ca^2+^ fluorescence signals.

This plugin is in a very early developmental/experimental stage and only supports reading images in .sif format from Andor Technologies powered by the [sif_parser] python module.

## Usage

This plugin can only read images generated with Andor Technologies cameras currently, having been tested on Zyla cameras. Just drag and drop an image to the napari GUI, and the image will display. Alternatively, you can programmatically load/read the image within a notebook.
    
    import napari
    
    file = "path/to/my/file/my_image.sif"

    viewer = napari.Viewer()
    viewer.open(path=file, plugin="napari-omaas", name = "my_image")

to display the metadata use the standard call to the corresponding layer:

    viewer.layers['my_image'].metadata


## Roadmap

This plugin is composed of two major components: **analysis** and **acquisition**.

### Analysis Features
    
- [x] Read sif files from Andor Technologies.
- [ ] Display time profile of ROIs on image sequences.
- [ ] Normalize images.
    - [ ] Segment images and align heart ROIs.
    - [ ] Perform peak analysis of action potential / Calcium traces 
    - [ ] Create activation maps
    - [ ] Add motion correction
- [ ] Export results and analysis log.

### Acquisition Features

- [ ] Control Zyla camera for the acquisition of data
    - [ ] test using the PYME module
- [ ] Real-time analysis(?)

    

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

Also review the napari docs for plugin developers:
https://napari.org/stable/plugins/index.html
-->

## Installation

You can install `napari-omaas` via [pip]:

    pip install napari-omaas



To install the latest development version :

    pip install git+https://github.com/rjlopez2/napari-omaas.git


## Contributing

Contributions are very welcome. Run tests with [tox], ensuring
the coverage remains the same before you submit a pull request.

## License

Distributed under the terms of the [BSD-3] license,
"napari-omaas" is free and open source software.

## Issues

If you encounter any problems, please [file an issue] and a  detailed description.

[napari]: https://github.com/napari/napari
[Cookiecutter]: https://github.com/audreyr/cookiecutter
[@napari]: https://github.com/napari
[MIT]: http://opensource.org/licenses/MIT
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[GNU GPL v3.0]: http://www.gnu.org/licenses/gpl-3.0.txt
[GNU LGPL v3.0]: http://www.gnu.org/licenses/lgpl-3.0.txt
[Apache Software License 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[Mozilla Public License 2.0]: https://www.mozilla.org/media/MPL/2.0/index.txt
[cookiecutter-napari-plugin]: https://github.com/napari/cookiecutter-napari-plugin

[file an issue]: https://github.com/rjlopez2/napari-omaas/issues

[napari]: https://github.com/napari/napari
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
[sif_parser]: https://pypi.org/project/sif-parser/
