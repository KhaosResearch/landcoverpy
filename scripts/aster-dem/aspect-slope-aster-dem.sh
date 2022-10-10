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
  mkdir -p aster-temp/aspect-slope

  # Copy product from minio
  mc cp --recursive $MINIO_ALIAS/$MINIO_BUCKET/$PRODUCT aster-temp/

  # Get aspect and slope

  # Create mapset based on file's location
  grass78 -c aster-temp/ASTGTMV003_${COORDINATES}.sdat -e ~/grassdata/${COORDINATES}/

  # Add file to mapset
  grass78 -c ~/grassdata/${COORDINATES}/PERMANENT/ --exec r.in.gdal input="aster-temp/ASTGTMV003_${COORDINATES}.sdat" band=1 output="elevation" --overwrite -o

  # Calculate aspect and slope
  grass78  -c ~/grassdata/${COORDINATES}/PERMANENT/ --exec r.slope.aspect elevation=elevation format="degrees" precision="FCELL" -a zscale=1 min_slope=0 slope=slope aspect=aspect --overwrite

  # Export slope to tif
  grass78  -c ~/grassdata/${COORDINATES}/PERMANENT/ --exec g.region raster=slope
  grass78  -c ~/grassdata/${COORDINATES}/PERMANENT/ --exec r.out.gdal -t -m input="slope" output="aster-temp/aspect-slope/slope.tif" format="GTiff" createopt="TFW=YES,COMPRESS=LZW" --overwrite

  # Export aspect to tif
  grass78  -c ~/grassdata/${COORDINATES}/PERMANENT/ --exec g.region raster=aspect
  grass78  -c ~/grassdata/${COORDINATES}/PERMANENT/ --exec r.out.gdal -t -m input="aspect" output="aster-temp/aspect-slope/aspect.tif" format="GTiff" createopt="TFW=YES,COMPRESS=LZW" --overwrite

  # Copy the results to minio
  mc cp  aster-temp/aspect-slope/* $MINIO_ALIAS/aster-gdem-aspect-slope/$COORDINATES/

  # Clear temp folders
  rm -r aster-temp/*

  echo "Complete Processing for $PRODUCT"
done