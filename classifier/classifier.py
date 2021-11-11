import pandas as pd
import numpy as np
from shapely.ops import transform
from functools import partial
import pyproj
import rasterio
from shapely.geometry import Point
import joblib


def read_band(band_filename, dst_crs = 'EPSG:4326'):
    """
    Reads the band specified in 'band_filename' and extracts information by pixel, 
    where each pixel value is a matrix position in the same order. 
    A 100x100px image resolution would result in a 100x100 matrix (before reescaling).
    Gets corners coordinates.
    Performs an Image rescaling to normalize pixel resolution to 10 meters per pixel.

    Params:
        band_filename (str) : Input band filename (jp2, tif).
        dst_crs (str) : Destination coordinate system (default is EPGS:4326).

    Returns:
        A tuple that contains:

            band (matrix) : matrix with image pixel information by position.
            top_left (tuple) : tuple of top left pixel latitude and longitude.
            bottom_right (tuple) : tuple of bottom right pixel latitude and longitude.
    
    """
    
    with rasterio.open(band_filename) as band_file:
        band = band_file.read(1).astype(np.float32)
        kwargs = band_file.meta        
        init_crs = band_file.crs    # Get source coordinate system

               
        # Get latitude and longitude      
        project = pyproj.Transformer.from_crs(init_crs, dst_crs, always_xy=True).transform        
        tl_lon, tl_lat = transform(project, Point(kwargs["transform"] * (0, 0))).bounds[0:2]
        br_lon, br_lat = transform(
            project, Point(kwargs["transform"] * (band.shape[1] - 1, band.shape[0] - 1))
        ).bounds[0:2]

        top_left = (tl_lat, tl_lon)
        bottom_right = (br_lat, br_lon)

        # Reescale image
        img_resolution = kwargs["transform"][0]
        scale_factor = img_resolution/10
        band = np.repeat(np.repeat(band, scale_factor, axis=0), scale_factor, axis=1)   

        band[band == 0] = np.nan    # Change no data value to nan

        
    return (band, top_left, bottom_right)

def get_latitude(rows, columns, initial_lat, final_lat):
    """
        Calculate latitude of a band pixel by pixel.

        Params:
            rows (int) : matrix number of rows.
            columns (int) : matrix number of columns.
            initial_lat (int) : value of the initial latitude (corresponding to the latitude of top left pixel of the band).
            final_lat (int) : value of the final latitude (corresponding to the latitude of bottom right pixel of the band).

        Returns:
            A matrix with latitude values for each pixel of a band.

    """
    
    pixel_lat = (final_lat-initial_lat)/rows
    aux = np.arange(initial_lat, final_lat, pixel_lat)
    result = ( np.tile(aux, columns)
                .transpose()
                .flatten() )

    return result
    

def get_longitude(columns, rows, initial_long, final_long):

    """
        Calculate longitude of a band pixel by pixel.

        Params:
            rows (int) : matrix number of rows.
            columns (int) : matrix number of columns.
            initial_long (int) : value of the initial longitude (corresponding to the longitude of top left pixel of the band).
            final_long (int) : value of the final longitude (corresponding to the longitude of bottom right pixel of the band).

        Returns:
            A matrix with longitude values for each pixel of a band.

    """

    pixel_long = (final_long-initial_long)/columns
    aux = np.arange(initial_long, final_long, pixel_long)
    result = np.tile(aux, rows).flatten() 

    return result


def classifier(band_filename, classifier_filename, n_chunks=10):
    """
        Classifies the image specified in 'band_file' pixel by pixel using the input classifier.
        Due to the classification process consuming a large amount of resources, it can be adjusted to suit user available resources.
        If resources are not an issue, it can be left by default. Otherwise, increase the number of chunks used for the classification process.

        
        Params:
            band_filename (str) : Input band filename (jp2, tif).
            classifier_filename (str) : Input classifier filename (joblib).
            n_chunks (int) : Number of chunks used for the classification process (default is 10).

        Returns:
            A matrix with the classification data, longitude and latitude for each pixel of the initial image.
    
    """
        
    band, top_left, bottom_right = read_band(band_filename)
    rows, columns = band.shape

    initial_long, initial_lat = top_left
    final_long, final_lat = bottom_right

    longitude = get_longitude(columns, rows, initial_long, final_long)
    latitude = get_latitude(rows, columns, initial_lat, final_lat)


    # Preprocessing data

    band = band.flatten()
    aux_df = pd.DataFrame(band)
    indexes = aux_df[np.isnan(band)].index
    dt = pd.DataFrame(band[~np.isnan(band)].flatten())          # Removes nan        

    
    # Classification
    prediction = np.array([])
    classifier = joblib.load(classifier_filename)
    array_split = np.array_split(dt, n_chunks)              

    for array in array_split:
        prediction_chunk = classifier.predict(pd.DataFrame(array))    
        prediction = np.append(prediction, prediction_chunk)
        del prediction_chunk


    # Reconstruction of the classification data, reshape final output

    array_pos = indexes - 1 - np.indices(indexes.shape).flatten()   # Shift position to original array
    dt = np.insert(prediction, array_pos, np.nan)

    dt = np.reshape(dt, [-1,1])
    latitude = np.reshape(latitude, [-1,1])
    longitude = np.reshape(longitude, [-1,1])

    result = np.concatenate((dt, latitude, longitude), axis=1)
    
    return result

    