#!!/bin/sh

# This scripts takes aster products from a minio bucket, applies 2 denoise filter and uploads them to minio again
# Filter 1: Sigma 85 Iterations 60 -> Topographic correction
# Filter 2: Sigma 90 Iterations 5  -> Slope, illumination, aspect, etc.
MINIO_ALIAS=minio2
MINIO_BUCKET='aster-gdem'

PRODUCTS=$(mc ls $MINIO_ALIAS/$MINIO_BUCKET | rev | cut -d ' ' -f 1 | rev)

for PRODUCT in $PRODUCTS; do
  # Get coordinates from name
  COORDINATES=$(echo $PRODUCT | cut -d '_' -f 2 | cut -d '.' -f 1)
  echo "$COORDINATES"
  if [[ $(mc ls $MINIO_ALIAS/${MINIO_BUCKET}-denoised-s85-i60/$COORDINATES/) ]]; then
    echo ""
  else
    echo "Error denoised s85"
  fi
  if [[ $(mc ls $MINIO_ALIAS/${MINIO_BUCKET}-denoised-s90-i5/$COORDINATES/) ]]; then
    echo ""
  else
    echo "Error denoised s90"
  fi
done