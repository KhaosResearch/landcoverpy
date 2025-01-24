from pathlib import Path

import numpy as np
import rasterio

from landcoverpy.utilities.raster import _rescale_band

# Allow division by zero.
np.seterr(divide="ignore", invalid="ignore")

def read(filename):
    """
    Read raster data from file.
    :param filename: Path to input file.
    :return: Raster d-array and metadata.
    """
    with rasterio.open(filename) as f:
        B = f.read().astype(np.float32)
        B[B == 0] = np.nan
        kwargs = f.meta
    return B, kwargs


def cloud_mask(scl: Path, *, output: Path= None) -> np.ndarray:
    """
    Computes cloud mask of an image based on the SCL raster provided by Sentinel.

    :param scl: SCL band for Sentinel-2 (20m).
    :param output: Path to output file.
    :return: Cloud cover mask (0 - no cloud, 1 - cloud).
    """
    scl_cloud_values = [3, 8, 9, 10, 11]  # Classification band's cloud-related values.

    with rasterio.open(scl, "r") as f:
        kwargs = f.meta
        mask = f.read()

    # Calculate cloud mask from Sentinel's cloud related values.
    mask = np.isin(mask, scl_cloud_values).astype(np.int8)

    cloud_mask_10m, output_kwargs = _rescale_band(mask, kwargs)

    if output:
        output_kwargs.update(driver="GTiff", dtype=rasterio.int8, count=1)
        with rasterio.open(output, "w", **output_kwargs) as f:
            f.write(cloud_mask_10m)

    return cloud_mask_10m


def true_color(r: Path, g: Path, b: Path, *, output: Path= None) -> np.ndarray:
    """
    Computes true color image composite (RGB).

    :param r: RED - B04 band for Sentinel-2 (10m).
    :param g: GREEN - B03 band for Sentinel-2 (10m).
    :param b: BLUE - B02 band for Sentinel-2 (10m).
    :param output: Path to output file.
    :return: True color image.
    """
    red, kwargs = read(r)
    green, _ = read(g)
    blue, _ = read(b)

    # Compose true color image.
    # Adjust each band by the min-max, so it will plot as RGB.
    rgb_image_raw = np.concatenate((red, green, blue), axis=0)

    max_pixel_value = rgb_image_raw.max(initial=0)
    rgb_image = np.multiply(rgb_image_raw, 255.0)
    rgb_image = np.divide(rgb_image, max_pixel_value)
    rgb_image = rgb_image.astype(np.uint8)

    if output:
        kwargs.update(driver="GTiff", dtype=rasterio.float32, nodata=0, count=3)
        with rasterio.open(output, "w", **kwargs) as rgb:
            rgb.write(rgb_image_raw)

    return rgb_image


def moisture(b8a: Path, b11: Path, *, output: Path= None) -> np.ndarray:
    """
    Compute moisture index.

    ..note:: In Sentinel-2 Level-2A products, zero values are reserved for 'No Data'.
     This value is used to define which pixels should be masked. See also:

    * https://docs.digitalearthafrica.org/en/latest/data_specs/Sentinel-2_Level-2A_specs.html

    :param b8a: B8A band for Sentinel-2 (60m).
    :param b11: B11 band for Sentinel-2 (60m).
    :param output: Path to output file.
    :return: Moisture index.
    """
    band_8a, kwargs = read(b8a)
    band_11, _ = read(b11)

    moisture = (band_8a - band_11) / (band_8a + band_11)

    moisture[moisture == np.inf] = np.nan
    moisture[moisture == -np.inf] = np.nan

    if output:
        kwargs.update(driver="GTiff", dtype=rasterio.float32, nodata=np.nan, count=1)
        with rasterio.open(output, "w", **kwargs) as f:
            f.write(moisture.astype(rasterio.float32))

    return moisture


def ndvi(b4: Path, b8: Path, *, output: Path= None) -> np.ndarray:
    """
    Compute Normalized Difference Vegetation Index (NDVI).

    Value ranges from -1 to 1. Negative values correspond to water.
    Values close to zero (-0.1 to 0.1) generally correspond to barren areas of rock, sand, or snow. Low,
    positive values represent shrub and grassland (approximately 0.2 to 0.4), while high values indicate
    temperate and tropical rainforests (values approaching 1).

    ..note:: https://medium.com/analytics-vidhya/satellite-imagery-analysis-with-python-3f8ccf8a7c32

    :param b4: RED - B04 band for Sentinel-2 (10m).
    :param b8: NIR - B08 band for Sentinel-2 (10m).
    :param output: Path to output file.
    :return: NDVI index.
    """
    red, kwargs = read(b4)
    nir, _ = read(b8)

    ndvi = (nir - red) / (nir + red)

    ndvi[ndvi == np.inf] = np.nan
    ndvi[ndvi == -np.inf] = np.nan

    if output:
        kwargs.update(driver="GTiff", dtype=rasterio.float32, nodata=np.nan, count=1)
        with rasterio.open(output, "w", **kwargs) as f:
            f.write(ndvi.astype(rasterio.float32))

    return ndvi


def ndsi(b3: Path, b11: Path, *, output: Path= None) -> np.ndarray:
    """
    Compute Normalized Difference Snow Index (NDSI) index.
    Values above 0.42 are usually snow.

    ..note:: https://eos.com/index-stack/

    :param b3: GREEN - B03 band for Sentinel-2 (20m).
    :param b11: SWIR - B11 band for Sentinel-2 (20m).
    :param output: Path to output file.
    :return: NDSI index.
    """
    band_3, kwargs = read(b3)
    band_11, _ = read(b11)

    ndsi = (band_3 - band_11) / (band_3 + band_11)
    ndsi = (ndsi > 0.42) * 1.0  # TODO - apply threshold (values above 0.42 are regarded as snowy)

    ndsi[ndsi == np.inf] = np.nan
    ndsi[ndsi == -np.inf] = np.nan

    if output:
        kwargs.update(driver="GTiff", dtype=rasterio.float32, nodata=np.nan, count=1)
        with rasterio.open(output, "w", **kwargs) as f:
            f.write(ndsi.astype(rasterio.float32))

    return ndsi


def ndwi(b3: Path, b8: Path, *, output: Path= None) -> np.ndarray:
    """
    Compute Normalized Difference Water Index (NDWI) index.

    ..note:: In Sentinel-2 Level-2A products, zero values are reserved for 'No Data'.
     This value is used to define which pixels should be masked. See also:

    * https://docs.digitalearthafrica.org/en/latest/data_specs/Sentinel-2_Level-2A_specs.html

    ..note:: https://eos.com/index-stack/

    :param b3: GREEN - B03 band for Sentinel-2 (10m).
    :param b8: NIR - B08 band for Sentinel-2 (10m).
    :param output: Path to output file.
    :return: NDWI index.
    """
    band_3, kwargs = read(b3)
    band_8, _ = read(b8)

    ndwi = (band_3 - band_8) / (band_3 + band_8)

    ndwi[ndwi == np.inf] = np.nan
    ndwi[ndwi == -np.inf] = np.nan

    if output:
        kwargs.update(driver="GTiff", dtype=rasterio.float32, nodata=np.nan, count=1)
        with rasterio.open(output, "w", **kwargs) as gtif:
            gtif.write(ndwi.astype(rasterio.float32))

    return ndwi


def evi2(b4: Path, b8: Path, *, output: Path= None) -> np.ndarray:
    """
    Compute Enhanced Vegetation Index 2 (EVI2) index.

    :param b4: B04 band for Sentinel-2 (10m).
    :param b8: B08 band for Sentinel-2 (10m).
    :param output: Path to output file.
    :return: EVI2 index.
    """
    band_4, kwargs = read(b4)
    band_8, _ = read(b8)

    evi2 = 2.4 * ((band_8 - band_4) / (band_8 + band_4 + 1.0))

    evi2[evi2 == np.inf] = np.nan
    evi2[evi2 == -np.inf] = np.nan

    if output:
        kwargs.update(driver="GTiff", dtype=rasterio.float32, nodata=np.nan, count=1)
        with rasterio.open(output, "w", **kwargs) as f:
            f.write(evi2.astype(rasterio.float32))

    return evi2


def osavi(b4: Path, b8: Path, Y: float = 0.16, *, output: Path= None) -> np.ndarray:
    """
    Optimized Soil Adjusted Vegetation Index (OSAVI) index.

    :param b4: B04 band for Sentinel-2 (10m).
    :param b8: B08 band for Sentinel-2 (10m).
    :param Y: Y coefficient.
    :param output: Path to output file.
    :return: OSAVI index.
    """
    band_4, kwargs = read(b4)
    band_8, _ = read(b8)

    osavi = (1 + Y) * (band_8 - band_4) / (band_8 + band_4 + Y)

    osavi[osavi == np.inf] = np.nan
    osavi[osavi == -np.inf] = np.nan

    if output:
        kwargs.update(driver="GTiff", dtype=rasterio.float32, nodata=np.nan, count=1)
        with rasterio.open(output, "w", **kwargs) as f:
            f.write(osavi.astype(rasterio.float32))

    return osavi


def ndre(b5: Path, b9: Path, *, output: Path= None) -> np.ndarray:
    """
    Normalized Difference NIR/Rededge Normalized Difference Red-Edge (NDRE) index.

    :param b5: B05 band for Sentinel-2 (60m).
    :param b9: B09 band for Sentinel-2 (60m).
    :param output: Path to output file.
    :return: NDRE index.
    """
    band_5, kwargs = read(b5)
    band_9, _ = read(b9)

    ndre = (band_9 - band_5) / (band_9 + band_5)

    ndre[ndre == np.inf] = np.nan
    ndre[ndre == -np.inf] = np.nan

    if output:
        kwargs.update(driver="GTiff", dtype=rasterio.float32, nodata=np.nan, count=1)
        with rasterio.open(output, "w", **kwargs) as f:
            f.write(ndre.astype(rasterio.float32))

    return ndre


def mndwi(b3: Path, b11: Path, *, output: Path= None) -> np.ndarray:
    """
    Modified NDWI (MNDWI) index.

    :param b3: B03 band for Sentinel-2 (20m).
    :param b11: B11 band for Sentinel-2 (20m).
    :param output: Path to output file.
    :return: MNDWI index.
    """
    band_3, kwargs = read(b3)
    band_11, _ = read(b11)

    mndwi = (band_3 - band_11) / (band_3 + band_11)

    mndwi[mndwi == np.inf] = np.nan
    mndwi[mndwi == -np.inf] = np.nan

    if output:
        kwargs.update(driver="GTiff", dtype=rasterio.float32, nodata=np.nan, count=1)
        with rasterio.open(output, "w", **kwargs) as f:
            f.write(mndwi.astype(rasterio.float32))

    return mndwi


def bri(b3: Path, b5: Path, b8: Path, *, output: Path= None) -> np.ndarray:
    """
    Browning Reflectance Index (BRI) index.

    :param b3: B03 band for Sentinel-2 (10m).
    :param b5: B05 band for Sentinel-2 (20m).
    :param b8: B08 band for Sentinel-2 (10m).
    :param output: Path to output file.
    :return: BRI index.
    """
    band_3, kwargs = read(b3)
    band_5, kwargs_5 = read(b5)
    band_5, _ = _rescale_band(band_5, kwargs_5)
    band_8, _ = read(b8)

    bri = (1 / band_3 - 1 / band_5) / band_8

    bri[bri == np.inf] = np.nan
    bri[bri == -np.inf] = np.nan

    if output:
        kwargs.update(driver="GTiff", dtype=rasterio.float32, nodata=np.nan, count=1)
        with rasterio.open(output, "w", **kwargs) as f:
            f.write(bri.astype(rasterio.float32))

    return bri


def evi(b2: Path, b4: Path, b8: Path, *, output: Path= None) -> np.ndarray:
    """
    Compute Enhanced Vegetation Index (EVI) index.
    Its value ranges from -1 to 1, with healthy vegetation generally around 0.20 to 0.80.

    ..note:: In Sentinel-2 Level-2A products, zero values are reserved for 'No Data'.
     This value is used to define which pixels should be masked. See also:

    * https://docs.digitalearthafrica.org/en/latest/data_specs/Sentinel-2_Level-2A_specs.html

    :param b2: B02 band for Sentinel-2 (10m).
    :param b4: B04 band for Sentinel-2 (10m).
    :param b8: B08 band for Sentinel-2 (10m).
    :param output: Path to output file.
    :return: EVI index.
    """
    band_2, kwargs = read(b2)
    band_4, _ = read(b4)
    band_8, _ = read(b8)

    evi = (2.5 * (band_8 - band_4)) / ((band_8 + 6 * band_4 - 7.5 * band_2) + 1)

    evi[evi == np.inf] = np.nan
    evi[evi == -np.inf] = np.nan

    if output:
        kwargs.update(driver="GTiff", dtype=rasterio.float32, nodata=np.nan, count=1)
        with rasterio.open(output, "w", **kwargs) as f:
            f.write(evi.astype(rasterio.float32))

    return evi


def ndyi(b2: Path, b3: Path, *, output: Path= None) -> np.ndarray:
    """
    Compute Normalized Difference Yellow Index (NDYI) index.

    * https://doi.org/10.1016/j.rse.2020.111660

    :param b2: B02 band for Sentinel-2 (10m).
    :param b3: B03 band for Sentinel-2 (10m).
    :param output: Path to output file.
    :return: NDYI index.
    """
    band_2, kwargs = read(b2)
    band_3, _ = read(b3)

    ndyi = (band_3 - band_2) / (band_3 + band_2)

    ndyi[ndyi == np.inf] = np.nan
    ndyi[ndyi == -np.inf] = np.nan

    if output:
        kwargs.update(driver="GTiff", dtype=rasterio.float32, nodata=np.nan, count=1)
        with rasterio.open(output, "w", **kwargs) as f:
            f.write(ndyi.astype(rasterio.float32))

    return ndyi


def ri(b3: Path, b4: Path, *, output: Path= None) -> np.ndarray:
    """
    Compute Normalized Difference Red/Green Redness (RI) index.

    :param b3: B03 band for Sentinel-2 (10m).
    :param b4: B04 band for Sentinel-2 (10m).
    :param output: Path to output file.
    :return: RI index.
    """
    band_3, kwargs = read(b3)
    band_4, _ = read(b4)

    ri = (band_4 - band_3) / (band_4 + band_3)

    ri[ri == np.inf] = np.nan
    ri[ri == -np.inf] = np.nan

    if output:
        kwargs.update(driver="GTiff", dtype=rasterio.float32, nodata=np.nan, count=1)
        with rasterio.open(output, "w", **kwargs) as f:
            f.write(ri.astype(rasterio.float32))

    return ri


def cri1(b2: Path, b3: Path, *, output: Path= None) -> np.ndarray:
    """
    Compute Carotenoid Reflectance (CRI1) index.

    :param b2: B02 band for Sentinel-2 (10m).
    :param b3: B03 band for Sentinel-2 (10m).
    :param output: Path to output file.
    :return: CRI1 index.
    """
    band_2, kwargs = read(b2)
    band_3, _ = read(b3)

    cri1 = (1 / band_2) / (1 / band_3)

    cri1[cri1 == np.inf] = np.nan
    cri1[cri1 == -np.inf] = np.nan

    if output:
        kwargs.update(driver="GTiff", dtype=rasterio.float32, nodata=np.nan, count=1)
        with rasterio.open(output, "w", **kwargs) as f:
            f.write(cri1.astype(rasterio.float32))

    return cri1


def bsi(b2: Path, b4: Path, b8: Path, b11: Path, *, output: Path= None) -> np.ndarray:
    """
    Bare Soil Index (BSI) is a numerical indicator to capture soil variations.

    :param b2: BLUE band (B02 for Sentinel-2 (10m)).
    :param b4: RED band (B04 for Sentinel-2 (10m)).
    :param b8: NIR band (B08 for Sentinel-2 (10m)).
    :param b11: SWIR band (B11 for Sentinel-2 (20m)).
    :param output: Path to output file.
    :return: BSI index.
    """
    band_2, kwargs = read(b2)
    band_4, _ = read(b4)
    band_8, _ = read(b8)
    band_11, kwargs_11 = read(b11)
    band_11, _ = _rescale_band(band_11, kwargs_11)

    bsi = ((band_11 + band_4) - (band_8 + band_2)) / ((band_11 + band_4) + (band_8 + band_2))

    bsi[bsi == np.inf] = np.nan
    bsi[bsi == -np.inf] = np.nan

    if output:
        kwargs.update(driver="GTiff", dtype=rasterio.float32, nodata=np.nan, count=1)
        with rasterio.open(output, "w", **kwargs) as gtif:
            gtif.write(bsi.astype(rasterio.float32))

    return bsi