import pdal
import json

# Definir el pipeline de PDAL en formato JSON
pipeline_json = json.dumps({
  "pipeline": [
    {
      "type": "readers.las",
      "filename": "/mnt/home/am/landcoverpy/scripts/PNOA_2020_AND_372-4066_ORT-CLA-RGB.laz"
    },
    {
      "type": "filters.decimation",
      "step": 1
    },
    {
      "type": "filters.outlier",
      "method": "statistical",
      "mean_k": 8,
      "multiplier": 2.5
    },
    {
      "type": "filters.range",
      "limits": "Classification![7:7]"
    },
    {
      "type": "filters.hag_nn"
    },
    {
      "type": "filters.ferry",
      "dimensions": "HeightAboveGround=Z"
    },
    {
      "type": "writers.gdal",
      "resolution": 10.0,
      "output_type": "max",
      "dimension": "Z",
      "filename": "/mnt/home/am/landcoverpy/scripts/output_max_height.tif",
      "data_type": "float"
    }
  ]
})

# Cargar y ejecutar el pipeline de PDAL
pipeline = pdal.Pipeline(pipeline_json)
pipeline.execute()
print("Pipeline executed successfully!")
