#!/bin/sh

# This scripts takes the aster products from a minio bucket, calculates their aspect and slope bands, then uploads the results to a new bucket in minio.


MINIO_ALIAS=minio25
MINIO_BUCKET='aster-gdem-denoised-s90-i5'
MINIO_BUCKET_NAME=etc-products

PRODUCTS=$(mc ls $MINIO_ALIAS/$MINIO_BUCKET | rev | cut -d ' ' -f 1 | rev)


for PRODUCT in $PRODUCTS; do
  # Get coordinates from name
  COORDINATES=$(echo $PRODUCT | cut -d '_' -f 2 | cut -d '.' -f 1 | cut -c1-7)

  # Create temp folders
  mkdir -p aster-temp/aspect-slope/aspect
  mkdir -p aster-temp/aspect-slope/slope

  # Copy product from minio
  mc cp --recursive $MINIO_ALIAS/$MINIO_BUCKET/$PRODUCT aster-temp/$PRODUCT

  
  # Get aspect and slope
  saga_cmd ta_morphometry 0 -ELEVATION "aster-temp/${PRODUCT}ASTGTMV003_${COORDINATES}.sdat" -SLOPE "aster-temp/aspect-slope/slope" -ASPECT "aster-temp/aspect-slope/aspect"

  
  # Copy the results to minio
  mc cp  aster-temp/aspect-slope/* $MINIO_ALIAS/aster-gdem-aspect-slope/$COORDINATES/
  
  # Clear temp folders
  rm -r aster-temp/*
  echo "Complete Processing for $PRODUCT"
done