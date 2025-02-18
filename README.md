# LandCoverPy, a scalable land cover/land use classification workflow

![iberian_peninsula_forests_example](https://github.com/KhaosResearch/landcoverpy/blob/v1.1/static/forests_example.JPG)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7462308.svg)](https://doi.org/10.5281/zenodo.7462308)

A scalable land cover classification workflow aimed to be able to scale to cover the Mediterranean bassin.

A research article describing the methodology followed on this workflow can be found at:

> **Scalable approach for high-resolution land cover: a case study in the Mediterranean Basin.**
> 
> Antonio Manuel Burgueño, José F. Aldana-Martín, María Vázquez-Pendón, Cristóbal Barba-González, Yaiza Jiménez Gómez, Virginia García Millán & Ismael Navas-Delgado
> 
> Journal of Big Data 10, 91 (2023). doi: [10.1186/s40537-023-00770-z](https://doi.org/10.1186/s40537-023-00770-z)

## Introduction

LandCoverPy is a Python package that provides a workflow for land cover/land use classification. It is designed to be scalable and to be able to process large areas. Currently, it works using Sentinel-2 data and ASTER GDEM data, but it could be easily extended to work with other data sources.

Since the library is prepared for large areas, S3 is used as the main storage system. Also MongoDB is used to store metadata of different assets which optimizes the workflow.

## Usage with docker compose.

The easiest way to use the library is through docker compose. A demo is provided in the folder [demo_deployment](demo_deployment). This demo, deploys a full environment with all the services needed to run the library and performs the full workflow on a small area. This workflow includes the following steps:

- 1. Downloading S2 images from Google Cloud.
- 2. Generates a training dataset from the S2 images.
- 3. Train a model using the training dataset.
- 4. Classify the area using the trained model.

To the requirements to run the demo and generating your first land cover classification is:
- Docker
- docker compose
- A Google Cloud service account key in JSON format, which should be placed in the folder [demo_deployment/app_data](demo_deployment/app_data). The file should be named `gcloud_user.json`. This is needed to download the S2 images.

Then, you can run the following commands:

```bash
cd demo_deployment
docker compose up
```

Once some time has passed, you can access the MinIO interface at [http://localhost:31113](http://localhost:31113) using the credentials `minioadmin` and `minioadmin` (all can be modified in the docker compose file). You can download the classification results from the bucket `classification-maps`.

The docker compose will work on data in folder [demo_deployment/app_data](demo_deployment/app_data). This could be easily adapted to work with your own data.

For this, you should provide 4 files (besides the `gcloud_user.json`):

- [demo_deployment/app_data/dataset.csv](demo_deployment/app_data/dataset.csv): A CSV file with the training dataset. The columns should be `latitude` `longitude` `category` and `subcategory`. The `category` column should be the main category and the `subcategory` column should be the subcategory. The `latitude` and `longitude` columns should be the coordinates of the training points. Since landcoverpy uses hierarchical classification, a model using the classes in the column `category` will be trained first for the whole area. Then, a model using the classes in the column `subcategory` will be trained for each class in the column `category`. These models will only classify the pixels classified as the corresponding class in the previous model. The `category` column is mandatory, but the `subcategory` column is optional, and can exists only for certain classes.
- [demo_deployment/app_data/seasons.json](demo_deployment/app_data/seasons.json): A JSON file with the seasons to download. The keys should be the season name and the values should be a list of dates in the format `YYYY-MM-DD`. This is the dates that will be downloaded for each season. Try to use dates that help to differentiate the classes you want to classify. It can be one or several seasons. 
- [demo_deployment/app_data/lc_labels.json](demo_deployment/app_data/lc_labels.json): A JSON file with the labels of the classes for the column `category`. The keys should be the class name and the values should be the class number. The class number should be an integer starting from 1 (0 is a reserved class). This is the class number that will be used in the classification.
- [demo_deployment/app_data/sl_labels.json](demo_deployment/app_data/sl_labels.json): A JSON file with the labels of the classes for the column `subcategory`. The keys should be the class name and the values should be the class number. The class number should be an integer starting from 2 (0 and 1 are reserved classes). This is the class number that will be used in the classification.

Finally, you should take a look at docker compose file [demo_deployment/docker-compose.yml](demo_deployment/docker-compose.yaml) to see the environment variables that can be set.

## Installation

If you are going to work in a serious project, maybe the docker compose is not the best option. In that case, you can install the package directly in your environment.
The package is not available on PyPI, so you need to install it from the source code. To do so, you can clone the repository and install it using pip:

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

The full documentation is not available yet, but you can check the [demo_deployment](demo_deployment) folder to see an example of how should be used. For a proper usage in Big Data scenarios, you should deploy a distributed dask cluster. Using the config variable `DASK_CLUSTER_IP` you can specify its IP.

## TODO
- Update composite creation to use windowing in order to reduce RAM usage and improve execution time.
- Better README with more examples and explanations.
- Better explanation on how to use the demo with your own data.

## License
This project is licensed under the MIT license. See the [LICENSE](LICENSE) file for more info.
