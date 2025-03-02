# **GLASS**: Generator for Large Scale Structure

<!-- Essentials -->

[![PyPI](https://img.shields.io/pypi/v/glass)](https://pypi.org/project/glass)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/glass.svg)](https://anaconda.org/conda-forge/glass)
[![Documentation](https://readthedocs.org/projects/glass/badge/?version=stable)](https://glass.readthedocs.io/stable)
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
[![All Contributors](https://img.shields.io/github/all-contributors/glass-dev/glass?color=ee8449&style=flat-square)](#contributors)

This is the core library for GLASS, the Generator for Large Scale Structure. For
more information, see the full [documentation]. There are a number of [examples]
to get you started.

## Installation

Releases of the code can be installed with pip:

```sh
pip install glass
```

or conda:

```sh
conda install -c conda-forge glass
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

[documentation]: https://glass.readthedocs.io/stable
[examples]: https://glass.readthedocs.io/stable/examples.html
[Discussions]: https://github.com/orgs/glass-dev/discussions
[Slack workspace]: https://glass-dev.github.io/slack

## Contributors

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="http://ntessore.page"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/3993688?v=4&h=100&w=100&fit=cover&mask=circle&maxage=7d" alt="Nicolas Tessore"/></td>
      <td align="center" valign="top" width="14.28%"><a href="https://paddyroddy.github.io"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/15052188?v=4&h=100&w=100&fit=cover&mask=circle&maxage=7d" alt="Patrick J. Roddy"/></td>
      <td align="center" valign="top" width="14.28%"><a href="https://saransh-cpp.github.io/"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/74055102?v=4&h=100&w=100&fit=cover&mask=circle&maxage=7d" alt="Saransh Chopra"/></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ucapbba"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/87702063?v=4&h=100&w=100&fit=cover&mask=circle&maxage=7d" alt="baugstein"/></td>
      <td align="center" valign="top" width="14.28%"><a href="http://arthurmloureiro.github.io"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/6471279?v=4&h=100&w=100&fit=cover&mask=circle&maxage=7d" alt="Arthur Loureiro"/></td>
      <td align="center" valign="top" width="14.28%"><a href="https://mwiet.github.io"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/49800039?v=4&h=100&w=100&fit=cover&mask=circle&maxage=7d" alt="Maximilian von Wietersheim-Kramsta"/></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/joachimi"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/4989590?v=4&h=100&w=100&fit=cover&mask=circle&maxage=7d" alt="joachimi"/></td>
      <td align="center" valign="top" width="14.28%"><a href="https://nialljeffrey.github.io/"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/15345794?v=4&h=100&w=100&fit=cover&mask=circle&maxage=7d" alt="Niall Jeffrey"/></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ARCHER2-HPC"><img src="https://images.weserv.nl/?url=https://avatars.githubusercontent.com/u/60643641?v=4&h=100&w=100&fit=cover&mask=circle&maxage=7d" alt="ARCHER2, UK National Supercomputing Service"/></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->
