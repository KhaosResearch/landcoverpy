import numpy as np
import rasterio

def __read_band(band_path: str):
    with rasterio.open(band_path) as band_file:
        band = band_file.read(1).astype(np.float32)
    return band

def composite(*band_paths: str, method: str = "median"):
    """
    Calculates de composite between a series of bands

    :param band_paths: List of paths to calculate the composite from.
    :param method: To calculate the composite. Values: "median", "mean".
    """
    composite_bands = [__read_band(band_path) for band_path in band_paths]

    shapes = [np.shape(band) for band in composite_bands] 

    # Check if all arrays are of the same shape 
    if not np.all(np.array(list(map(lambda x: x == shapes[0], shapes)))):  
        raise ValueError(f"Not all bands have the same shape\n{shapes}")
    
    if method == "mean":
        composite_out = np.mean(composite_bands, axis=0)
    elif method == "median":
        composite_out = np.median(composite_bands, axis=0)
    else:
        raise ValueError(f"Method '{method}' is not recognized.")

    return composite_out

