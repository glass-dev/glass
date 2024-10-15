# **GLASS**: Generator for Large Scale Structure

<!-- Essentials -->

[![PyPI](https://img.shields.io/pypi/v/glass)](https://pypi.org/project/glass)
[![Documentation](https://readthedocs.org/projects/glass/badge/?version=latest)](https://glass.readthedocs.io/latest)
[![LICENSE](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

<!-- Code -->

[![Python Versions](https://img.shields.io/pypi/pyversions/glass)](https://pypi.org/project/glass)
[![Test](https://github.com/glass-dev/glass/actions/workflows/test.yml/badge.svg)](https://github.com/glass-dev/glass/actions/workflows/test.yml)
[![Coverage Status](https://coveralls.io/repos/github/glass-dev/glass/badge.svg?branch=main)](https://coveralls.io/github/glass-dev/glass?branch=main)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/glass-dev/glass/main.svg)](https://results.pre-commit.ci/latest/github/glass-dev/glass/main)

<!-- Science -->

[![arXiv](https://img.shields.io/badge/arXiv-2302.01942-red)](https://arxiv.org/abs/2302.01942)
[![adsabs](https://img.shields.io/badge/ads-2023OJAp....6E..11T-blueviolet)](https://ui.adsabs.harvard.edu/abs/2023OJAp....6E..11T)
[![doi](https://img.shields.io/badge/doi-10.21105/astro.2302.01942-blue)](https://dx.doi.org/10.21105/astro.2302.01942)

<!-- Community -->

[![GitHub Discussions](https://img.shields.io/static/v1?label=Discussions&message=Ask&color=blue&logo=github)](https://github.com/orgs/glass-dev/discussions)
[![Slack](https://img.shields.io/badge/join-Slack-4A154B)](https://glass-dev.github.io/slack)

This is the core library for GLASS, the Generator for Large Scale Structure. For
more information, see the full [documentation]. There are a number of [examples]
to get you started.

## Installation

Releases of the code can be installed with pip as usual:

```sh
pip install glass
```

If you are interested in the latest version of the code, you can pip-install
this repository:

```sh
pip install git+https://github.com/glass-dev/glass.git
```

## Citation

If you use GLASS simulations or the GLASS library in your research, please
[cite the original GLASS paper](https://glass.readthedocs.io/stable/user/publications.html)
in your publications.

<!-- markdownlint-disable MD013 -->

```bibtex
@ARTICLE{2023OJAp....6E..11T,
       author = {{Tessore}, Nicolas and {Loureiro}, Arthur and {Joachimi}, Benjamin and {von Wietersheim-Kramsta}, Maximilian and {Jeffrey}, Niall},
        title = "{GLASS: Generator for Large Scale Structure}",
      journal = {The Open Journal of Astrophysics},
     keywords = {Astrophysics - Cosmology and Nongalactic Astrophysics},
         year = 2023,
        month = mar,
       volume = {6},
          eid = {11},
        pages = {11},
          doi = {10.21105/astro.2302.01942},
archivePrefix = {arXiv},
       eprint = {2302.01942},
 primaryClass = {astro-ph.CO},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2023OJAp....6E..11T},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

## Getting in touch

The best way to get help about the code is currently to get in touch.

If you would like to start a discussion with the wider GLASS community about
e.g. a design decision or API change, you can use our [Discussions] page.

We also have a public [Slack workspace] for discussions about the project.

[documentation]: https://glass.readthedocs.io/
[examples]: https://glass.readthedocs.io/projects/examples/
[Discussions]: https://github.com/orgs/glass-dev/discussions
[Slack workspace]: https://glass-dev.github.io/slack
