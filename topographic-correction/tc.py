from helpers import (
    gather_aster,
    get_bound,
    get_corners,
    get_crs,
    get_sun,
    merge_dem,
    reproject_dem,
)

import glob
import os
import subprocess
from pathlib import Path


def apply_correction(
    band_path: str,
    dem_path: str,
    output_path: str,
    azimuth: float,
    height: float,
    method: int = 5,
    max_cells: int = 1000,
    max_value: int = 1,
) -> None:
    """
    Given a band's path and the DEMs' paths, performs the C topological correction (or others).
    Parameters for the algorithm can be passed as arguments.

    [0] Cosine Correction (Teillet et al. 1982)
    [1] Cosine Correction (Civco 1989)
    [2] Minnaert Correction
    [3] Minnaert Correction with Slope (Riano et al. 2003)
    [4] Minnaert Correction with Slope (Law & Nichol 2004)
    [5] C Correction
    [6] Normalization (after Civco, modified by Law & Nichol)

    Returns the corrected image's path.
    """
    subprocess.run(
        [
            "/usr/bin/saga_cmd",
            "ta_lighting",
            "4",
            "-DEM", f"{dem_path}",
            "-ORIGINAL", f"{band_path}",
            "-CORRECTED", f"{output_path}",
            "-AZI", f"{azimuth}",
            "-HGT", f"{height}",
            "-METHOD", f"{method}",
            "-MAXCELLS", f"{max_cells}",
            "-MAXVALUE", f"{max_value}",
        ]
    )

    # TODO: error control


if __name__ == "__main__":

    base_path = Path("/home/ubuntu/workflows/etc-uma/topographic-correction/")
    products_path = base_path / "products/"
    dem_path = base_path / "dem/denoised-example/"
    temp_path = base_path / "tmp/"

    for prod in os.listdir(products_path):
        p_path = products_path / prod
        dst_xml_path = glob.glob(str(p_path) + r"/*.SAFE/GRANULE/**/MTD_TL.xml")[0]

        dst_crs = get_crs(dst_xml_path)
        azimuth, height = get_sun(dst_xml_path)

        prod_r10, prod_r20, prod_r60 = sorted(
            glob.glob(str(p_path) + r"/*.SAFE/GRANULE/**/IMG_DATA/**")
        )
        sample_band = os.listdir(p_path / prod_r10)[0]
        (
            top_left,
            top_right,
            bottom_left,
            bottom_right,
        ) = get_corners(p_path / prod_r10 / sample_band)

        left_bound = get_bound(bottom_left, top_left)
        right_bound = get_bound(bottom_right, top_right)
        upper_bound = get_bound(top_left, top_right, is_up_down=True)
        lower_bound = get_bound(bottom_left, bottom_right, is_up_down=True)

        overlapping_dem = gather_aster(
            dem_path, left_bound, right_bound, upper_bound, lower_bound
        )

        dem = merge_dem(
            dem_paths=overlapping_dem,
            outfile=temp_path / prod / "merge.sdat",
        )

        r_dem = reproject_dem(dem, dst_crs)

        apply_correction(
            p_path / prod_r10 / sample_band,
            r_dem,
            temp_path / prod / "test.sdat",
            azimuth,
            height,
        )

        # TODO: apply topographic correction to every band for all products
        # for band in os.listdir(prod_r10):
        #    apply_correction(band_path=band, dem_path=dem)
