import json
import logging
import os
import shutil
from datetime import datetime, timedelta
from pathlib import Path

from ds_download.download_using_sentinel_api import download_product_using_sentinel_api
from landcoverpy.utilities.geometries import (
    _csv_to_geojson,
    _group_validated_data_points_by_tile,
    _kmz_to_geojson,
)

logger = logging.getLogger(__name__)


def _cleanup_tmp_dir():
    """Garantiza la limpieza de archivos temporales residuales en TMP_DIR."""
    tmp_dir = Path(os.getenv("TMP_DIR", "/app/tmp/"))
    if tmp_dir.exists():
        for item in tmp_dir.iterdir():
            try:
                if item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)
                else:
                    item.unlink(missing_ok=True)
            except Exception as e:
                print(f"Advertencia al limpiar archivo temporal {item}: {e}")


def download_products():
    with open(os.getenv("SEASONS_FILE"), "r") as f:
        seasons = json.load(f)

    data_file = os.getenv("DB_FILE")
    if data_file.endswith(".kmz"):
        data_file = _kmz_to_geojson(data_file)
    if data_file.endswith(".csv"):
        data_file = _csv_to_geojson(data_file, sep=";")

    polygons_per_tile = _group_validated_data_points_by_tile(data_file)
    tiles_to_train = set(polygons_per_tile.keys())

    tiles_to_predict_env = os.getenv("TILES_TO_PREDICT", "[]")
    if tiles_to_predict_env.lower() != "prediction":
        try:
            tiles_to_predict = set(json.loads(tiles_to_predict_env))
        except json.JSONDecodeError:
            raise ValueError(
                f"Error parsing TILES_TO_PREDICT: {tiles_to_predict_env}. "
                f"Expected format: '['NNLLL', 'NNLLL', ...]'"
            )
    else:
        tiles_to_predict = set()

    # Ordenamiento determinista: evita la aleatoriedad de iterar sobre sets
    tiles_to_download = sorted(tiles_to_train.union(tiles_to_predict))
    total_tiles = len(tiles_to_download)

    # Modo de ejecución: "TILE_FIRST" (recomendado) o "SEASON_FIRST" (legacy)
    pipeline_mode = os.getenv("DOWNLOAD_PIPELINE_MODE", "TILE_FIRST").upper()
    print(f"Total tiles a procesar: {total_tiles}")
    print(f"Modo de descarga: {pipeline_mode}")

    if pipeline_mode == "TILE_FIRST":
        # Paradigma Tile-First: completa todas las estaciones de un tile antes de pasar al siguiente
        for idx, tile in enumerate(tiles_to_download, 1):
            print(f"\n==================================================")
            print(f"[{idx}/{total_tiles}] Procesando Tile: {tile}")
            print(f"==================================================")

            for season, dates in seasons.items():
                start_date = datetime.strptime(dates["start"], "%Y-%m-%d")
                end_date = datetime.strptime(dates["end"], "%Y-%m-%d")

                print(f" -> Tile {tile} | Estación {season} ({dates['start']} a {dates['end']})")
                try:
                    download_product_using_sentinel_api(
                        False, True, start_date, end_date, tile_id=tile
                    )
                finally:
                    # Garantiza que los temporales del producto se borren inmediatamente
                    _cleanup_tmp_dir()

    else:
        # Modo Season-First (legacy, pero ordenado y con limpieza)
        for season in seasons:
            start_date = datetime.strptime(seasons[season]["start"], "%Y-%m-%d")
            end_date = datetime.strptime(seasons[season]["end"], "%Y-%m-%d")

            print(f"\n>>> Procesando estación {season} ({start_date.date()} a {end_date.date()})")
            for idx, tile in enumerate(tiles_to_download, 1):
                print(f"[{idx}/{total_tiles}] Tile {tile} para estación {season}")
                try:
                    download_product_using_sentinel_api(
                        False, True, start_date, end_date, tile_id=tile
                    )
                finally:
                    _cleanup_tmp_dir()
