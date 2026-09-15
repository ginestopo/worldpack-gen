"""Lectura por ventanas de COGs remotos (sin descargar teselas completas).

Ambas fuentes estan en EPSG:4326 igual que nuestra rejilla, asi que no hay
reproyeccion: solo recortar y remuestrear."""
import math
import os

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.errors import RasterioIOError
from rasterio.windows import from_bounds

from . import grid

WORLDCOVER = os.environ.get(
    "WP_WORLDCOVER",
    "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
    "ESA_WorldCover_10m_2021_v200_{NS}{alat:02d}{EW}{alon:03d}_Map.tif")
DEM = os.environ.get(
    "WP_DEM",
    "https://copernicus-dem-30m.s3.amazonaws.com/"
    "Copernicus_DSM_COG_10_{NS}{alat:02d}_00_{EW}{alon:03d}_00_DEM/"
    "Copernicus_DSM_COG_10_{NS}{alat:02d}_00_{EW}{alon:03d}_00_DEM.tif")

GDAL_ENV = dict(
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
    GDAL_HTTP_MULTIRANGE="YES",
    GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
    GDAL_HTTP_MAX_RETRY=6,
    GDAL_HTTP_RETRY_DELAY=3,
    VSI_CACHE=True,
    VSI_CACHE_SIZE=256 * 1024 * 1024,
    GDAL_CACHEMAX=512,
)

_missing = set()


def _tile_url(template, lat, lon):
    return template.format(
        NS="N" if lat >= 0 else "S", alat=abs(lat),
        EW="E" if lon >= 0 else "W", alon=abs(lon),
        lat=lat, lon=lon)


def _open(url):
    if url in _missing:
        return None
    try:
        return rasterio.open(url)
    except RasterioIOError as e:
        msg = str(e)
        # 404 / inexistente = mar abierto. Cualquier otro error debe romper el build.
        # S3 puede responder 403 en vez de 404 para claves inexistentes. La orden
        # "probe" del CLI verifica antes que las teselas de tierra si se leen.
        if any(s in msg for s in ("404", "403", "No such file", "does not exist",
                                  "not recognized")):
            _missing.add(url)
            return None
        raise


def read_gh3(template, tile_deg, gid, oversample, resampling, fill=0, dtype="float32"):
    """Devuelve un array (1024*k, 1024*k) con la fila 0 al SUR."""
    k = oversample
    n = grid.GH3 * k
    ds_deg = grid.CELL_DEG / k
    gx0, gy0 = grid.gh3_origin(gid)
    lon0 = gx0 * grid.CELL_DEG - 180.0
    lat0 = gy0 * grid.CELL_DEG - 90.0
    lon1 = lon0 + grid.GH3 * grid.CELL_DEG
    lat1 = lat0 + grid.GH3 * grid.CELL_DEG
    out = np.full((n, n), fill, dtype=dtype)
    found = False

    t_lat0 = math.floor(lat0 / tile_deg) * tile_deg
    t_lon0 = math.floor(lon0 / tile_deg) * tile_deg
    with rasterio.Env(**GDAL_ENV):
        for tlat in range(int(t_lat0), int(math.ceil(lat1)), tile_deg):
            for tlon in range(int(t_lon0), int(math.ceil(lon1)), tile_deg):
                src = _open(_tile_url(template, tlat, tlon))
                if src is None:
                    continue
                with src:
                    b = src.bounds
                    il0, il1 = max(lon0, b.left), min(lon1, b.right)
                    ib0, ib1 = max(lat0, b.bottom), min(lat1, b.top)
                    if il1 <= il0 or ib1 <= ib0:
                        continue
                    c0 = math.ceil((il0 - lon0) / ds_deg - 0.5)
                    c1 = math.floor((il1 - lon0) / ds_deg - 0.5)
                    r0 = math.ceil((ib0 - lat0) / ds_deg - 0.5)
                    r1 = math.floor((ib1 - lat0) / ds_deg - 0.5)
                    if c1 < c0 or r1 < r0:
                        continue
                    left = max(lon0 + c0 * ds_deg, b.left)
                    right = min(lon0 + (c1 + 1) * ds_deg, b.right)
                    bottom = max(lat0 + r0 * ds_deg, b.bottom)
                    top = min(lat0 + (r1 + 1) * ds_deg, b.top)
                    win = from_bounds(left, bottom, right, top, src.transform)
                    data = src.read(1, window=win, out_shape=(r1 - r0 + 1, c1 - c0 + 1),
                                    resampling=resampling, masked=True)
                    data = data.filled(fill)[::-1]  # norte arriba -> sur en fila 0
                    out[r0:r1 + 1, c0:c1 + 1] = data
                    found = True
    return out, found


def worldcover_gh3(gid, oversample=4):
    arr, found = read_gh3(WORLDCOVER, 3, gid, oversample, Resampling.nearest, 0, "uint8")
    return arr, found


def dem_gh3(gid):
    arr, found = read_gh3(DEM, 1, gid, 1, Resampling.average, 0.0, "float32")
    arr = np.nan_to_num(arr, nan=0.0)
    return np.rint(arr).astype(np.int32), found


def probe(lon, lat):
    """Lee una tesela de tierra conocida de cada fuente. Falla si no hay acceso."""
    import math as _m
    out = {}
    for name, tmpl, deg in (("worldcover", WORLDCOVER, 3), ("dem", DEM, 1)):
        tlat = int(_m.floor(lat / deg) * deg)
        tlon = int(_m.floor(lon / deg) * deg)
        url = _tile_url(tmpl, tlat, tlon)
        with rasterio.Env(**GDAL_ENV), rasterio.open(url) as src:
            ov = src.overviews(1)
            px = src.read(1, window=((src.height // 2, src.height // 2 + 1),
                                     (src.width // 2, src.width // 2 + 1)))
            out[name] = {"url": url, "size": [src.width, src.height],
                         "overviews": ov, "center_value": float(px[0, 0])}
    return out
