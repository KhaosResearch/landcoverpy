# Script for classifying satellite images.

This script classifies a precalculated index image using its corresponding classifier.
Due to the classification process consuming a large amount of resources, it can be adjusted to suit user available resources.
If resources are not an issue, it can be left by default. Otherwise, increase the number of chunks used for the classification process.
        
For example, to classify a NDVI image add the following line of code to the main function
```Python
    image_classified = classifier('path/ndvi.tif', 'path/classifier_ndvi.joblib')
```

    Note : The input classifier must be a joblib file.