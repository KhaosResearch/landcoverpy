# LandCoverPy: A Scalable Land Cover/Land Use Classification Workflow

![Iberian Peninsula Forests Example](https://github.com/KhaosResearch/landcoverpy/blob/master/static/forests_example.JPG)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7462308.svg)](https://doi.org/10.5281/zenodo.7462308)

LandCoverPy is a scalable land cover classification workflow designed to process large areas, including the Mediterranean Basin.

A research article describing the methodology behind this workflow is available:

> **Scalable approach for high-resolution land cover: a case study in the Mediterranean Basin.**
> 
> Antonio Manuel Burgueño, José F. Aldana-Martín, María Vázquez-Pendón, Cristóbal Barba-González, Yaiza Jiménez Gómez, Virginia García Millán & Ismael Navas-Delgado
> 
> Journal of Big Data 10, 91 (2023). doi: [10.1186/s40537-023-00770-z](https://doi.org/10.1186/s40537-023-00770-z)

## Introduction

LandCoverPy is a Python package that provides a workflow for land cover/land use classification. It is designed to be scalable and capable of processing large areas. Currently, it works with Sentinel-2 data and ASTER GDEM data, but it can be easily extended to support other data sources.

Since the library is intended for large areas, it uses **S3** as the main storage system and **MongoDB** to store metadata of different assets, optimizing the overall workflow.

## Usage with Docker Compose

The easiest way to use the library is through Docker Compose. A demo is provided in the [demo_deployment](demo_deployment) folder. This demo deploys a complete environment with all the services required to run the library and performs the full workflow on a small area. The workflow includes the following steps:

1. **Downloading Sentinel-2 images** from Google Cloud.
2. **Generating a training dataset** from the Sentinel-2 images.
3. **Training a model** using the training dataset.
4. **Classifying the area** using the trained model.

### Requirements

To run the demo and generate your first land cover classification, you need:

- **Docker**
- **Docker Compose**
- A **Google Cloud service account key** in JSON format, which must be placed in the folder [demo_deployment/app_data](demo_deployment/app_data) under the name `gcloud_user.json`. This key is necessary to download the Sentinel-2 images.

### Running the Demo

Execute the following commands:

```bash
cd demo_deployment
docker compose up
```

After a while, you can access the MinIO interface at [http://localhost:31113](http://localhost:31113) using the credentials:

- **Username:** `adminadmin`
- **Password:** `adminadmin`

*(You can modify these credentials in the Docker Compose file.)*

The classification results will be available in the bucket named `classification-maps`. Note that the first execution for each dataset may take some time, as the images need to be downloaded and the composite images computed.

The Docker Compose setup operates on data located in the folder [demo_deployment/app_data](demo_deployment/app_data). You can easily adapt it to work with your own data.

### Required Data Files

In addition to `gcloud_user.json`, you must provide four files:

- **[dataset.csv](demo_deployment/app_data/dataset.csv):**  
  A CSV file containing the training dataset. It should include the following columns:
  - `latitude`
  - `longitude`
  - `category` (main category, **mandatory**)
  - `subcategory` (sub-category, **optional**)

  *Note:* LandCoverPy employs hierarchical classification. First, a model is trained using the `category` classes for the whole area. Then, for each class in `category`, a model is trained using the `subcategory` classes to classify only the corresponding pixels.

- **[seasons.json](demo_deployment/app_data/seasons.json):**  
  A JSON file specifying the seasons for downloading images. The keys should be the season names, and the values should be lists of dates in the `YYYY-MM-DD` format. Choose dates that help differentiate the classes you want to classify.

- **[lc_labels.json](demo_deployment/app_data/lc_labels.json):**  
  A JSON file with the labels for the `category` classes. The keys should be the class names, and the values should be the class numbers (integers starting from 1; 0 is reserved).

- **[sl_labels.json](demo_deployment/app_data/sl_labels.json):**  
  A JSON file with the labels for the `subcategory` classes. The keys should be the class names, and the values should be the class numbers (integers starting from 2; 0 and 1 are reserved).

Finally, review the Docker Compose file [demo_deployment/docker-compose.yml](demo_deployment/docker-compose.yaml) to see the environment variables that can be configured.

## Installation

For production projects, Docker Compose might not be the best option. Instead, you can install the package directly into your environment. Since the package is not available on PyPI, you must install it from source.

Clone the repository and install using pip:

```bash
git clone https://github.com/KhaosResearch/landcoverpy.git
cd landcoverpy
pip install .
```

For development purposes, you can install the package in editable mode:

```bash
pip install -e .
```

## Usage

Full documentation is not available yet, but you can refer to the [demo_deployment](demo_deployment) folder for an example of how to use the package. For proper usage in Big Data scenarios, it is recommended to deploy a distributed Dask cluster. You can specify its IP using the `DASK_CLUSTER_IP` configuration variable.

## TODO

- Update composite creation to use windowing to reduce RAM usage and improve execution time.
- Enhance the README with more examples and explanations.
- Provide a better guide on how to use the demo with your own data.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.