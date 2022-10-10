# Scripts for the download and denoise of aster-dem products

## Download
This script downloads all products between two coordinates.

Products of areas containing only water are not distributed. This is the cause for most errors of product not found.

## Denoise
Has two dependencies, [mc](https://github.com/minio/mc) and [saga](https://sourceforge.net/projects/saga-gis/).

This scripts takes aster products from a minio bucket, applies 2 denoise filter and uploads them to minio again.

Filter 1: Sigma 85 Iterations 60 for Topographic correction.
Filter 2: Sigma 90 Iterations 5  for Slope, illumination, aspect, etc.

## Aspect and Slope

Has two dependencies, [mc](https://github.com/minio/mc) and [saga](https://sourceforge.net/projects/saga-gis/).

This scripts takes the aster products from a minio bucket, calculates their aspect and slope bands, then uploads the results to a new bucket in minio.