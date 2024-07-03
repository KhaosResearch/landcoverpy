# LandCoverPy, a scalable land cover/land use classification workflow

![lebanon_second_level_classification](https://github.com/KhaosResearch/landcoverpy/blob/v1.1/static/lebanon_example.png)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7462308.svg)](https://doi.org/10.5281/zenodo.7462308)

A scalable land cover classification workflow aimed to be able to scale to cover the Mediterranean bassin.

A research article describing the methodology followed on this workflow can be found at:

> **Scalable approach for high-resolution land cover: a case study in the Mediterranean Basin.**
> 
> Antonio Manuel Burgueño, José F. Aldana-Martín, María Vázquez-Pendón, Cristóbal Barba-González, Yaiza Jiménez Gómez, Virginia García Millán & Ismael Navas-Delgado
> 
> Journal of Big Data 10, 91 (2023). doi: [10.1186/s40537-023-00770-z](https://doi.org/10.1186/s40537-023-00770-z)

## Installation

Currently, the package is not available on PyPI, so you need to install it from the source code. To do so, you can clone the repository and install it using pip:

```bash
git clone https://github.com/KhaosResearch/landcoverpy.git
cd landcoverpy
pip install .
```

For development purposes, you can install the package in editable mode:

```bash
pip install -e .
```

In the future, the package will be available on PyPI, so you will be able to install it using pip:

```bash
pip install landcoverpy
```

## Usage

An usage example can be found at the [main usage notebook](notebooks/main_usage.ipynb).

## License
This project is licensed under the MIT license. See the [LICENSE](LICENSE) file for more info.
