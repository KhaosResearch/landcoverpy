import numpy as np
import rasterio

def read_band(band_path: str):
    with rasterio.open(band_path) as band_file:
        band = band_file.read(1).astype(np.float32)
    return band

def composite(*band_paths, c):
    composite_bands = [read_band(band_path) for band_path in band_paths]
    composite_out = np.mean(composite_bands, axis=0)
    return composite_out

