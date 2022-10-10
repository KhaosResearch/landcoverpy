#!!/bin/sh

# This scripts takes aster products from a minio bucket, applies 2 denoise filter and uploads them to minio again
# Filter 1: Sigma 85 Iterations 60 -> Topographic correction
# Filter 2: Sigma 90 Iterations 5  -> Slope, illumination, aspect, etc.
MINIO_ALIAS=minio25
MINIO_BUCKET='aster-gdem'

PRODUCTS=$(mc ls $MINIO_ALIAS/$MINIO_BUCKET | rev | cut -d ' ' -f 1 | rev)

for PRODUCT in $PRODUCTS; do
  # Get coordinates from name
  COORDINATES=$(echo $PRODUCT | cut -d '_' -f 2 | cut -d '.' -f 1)
  # Create temp folders
  mkdir -p aster-temp/sigma85iter60
  mkdir -p aster-temp/sigma90iter5
  echo "$COORDINATES"
  # Copy product from minio
  mc cp $MINIO_ALIAS/$MINIO_BUCKET/$PRODUCT aster-temp/$PRODUCT
  # Unzip DEM
  unzip aster-temp/$PRODUCT ASTGTMV003_*.zip -d aster-temp
  unzip aster-temp/ASTGTMV003_*.zip *_dem.tif -d aster-temp
  # Aplicate denoise
  saga_cmd grid_filter 10 -INPUT "aster-temp/ASTGTMV003_${COORDINATES}_dem.tif" -OUTPUT "aster-temp/sigma85iter60/ASTGTMV003_$COORDINATES" -SIGMA 0.85 -ITER 60
  saga_cmd grid_filter 10 -INPUT "aster-temp/ASTGTMV003_${COORDINATES}_dem.tif" -OUTPUT "aster-temp/sigma90iter5/ASTGTMV003_$COORDINATES" -SIGMA 0.90 -ITER 5
  # Copy the results to minio
  mc cp aster-temp/sigma85iter60/* $MINIO_ALIAS/${MINIO_BUCKET}-denoised-s85-i60/$COORDINATES/
  mc cp aster-temp/sigma90iter5/* $MINIO_ALIAS/${MINIO_BUCKET}-denoised-s90-i5/$COORDINATES/
  # Clear temp folders
  rm -r aster-temp/*
  echo "Complete Processing for $PRODUCT"
done