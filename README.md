# napari-omaas

[![License BSD-3](https://img.shields.io/pypi/l/napari-omaas.svg?color=green)](https://github.com/rjlopez2/napari-omaas/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-omaas.svg?color=green)](https://pypi.org/project/napari-omaas)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-omaas.svg?color=green)](https://python.org)
[![tests](https://github.com/rjlopez2/napari-omaas/workflows/tests/badge.svg)](https://github.com/rjlopez2/napari-omaas/actions)
[![codecov](https://codecov.io/gh/rjlopez2/napari-omaas/branch/main/graph/badge.svg)](https://codecov.io/gh/rjlopez2/napari-omaas)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-omaas)](https://napari-hub.org/plugins/napari-omaas)

**napari-OMAAS stands for Optical Mapping Acquisition and Analysis Software for panoramic heart imaging**

This napari plugin intends to be an analysis and acquisition tool for optical mapping in potentiometric (V<sub>m</sub>) or calcium (Ca<sup>2+</sup>) fluorescence signals obtained from panoramic imaging of intact hearts.

 It supports reading images in .sif format and binary files generated from Andor Technologies powered by the [sif_parser] python module.



```{admonition} Experimental
:class: warning
This plugin is in early developmental/experimental stage so expect  braking changes at anytime.
```
# Examples

The following example ilustrate how to perform normalization (pixelwise) on a time serie image and plot its 2d profile along the t dimension withing the average data from the ROI selected.

![](https://github.com/rjlopez2/napari-omaas/blob/documentation/example_imgs/Oct-31-2023%2016-45-55_plot_profile.gif?raw=true)

----------------------------------
The next example shows how to compute action potential duration in the same image stack.

![](https://github.com/rjlopez2/napari-omaas/blob/documentation/example_imgs/Oct-31-2023%2016-49-02_APD_analysis.gif?raw=true)

----------------------------------

This [napari] plugin was generated with [Cookiecutter] using [@napari]'s [cookiecutter-napari-plugin] template.

<!--
Don't miss the full getting started guide to set up your new package:
https://github.com/napari/cookiecutter-napari-plugin#getting-started

Also review the napari docs for plugin developers:
https://napari.org/stable/plugins/index.html
-->


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

[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
[sif_parser]: https://pypi.org/project/sif-parser/
