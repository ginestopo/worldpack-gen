"""Rejilla geohash-7 como enteros (capitulo 2 del spec)."""
import numpy as np

GRID_W = 262144            # 2^18 celdas en longitud
GRID_H = 131072            # 2^17 celdas en latitud
CELL_DEG = 360.0 / GRID_W  # == 180 / GRID_H
CHUNK = 32                 # celdas por lado de chunk
GH3 = 1024                 # celdas por lado de gh3


def xy_from_gps(lon, lat):
    x = int((lon + 180.0) / 360.0 * GRID_W)
    y = int((lat + 90.0) / 180.0 * GRID_H)
    return min(max(x, 0), GRID_W - 1), min(max(y, 0), GRID_H - 1)


def zorder5(a, b):
    """Intercala 5 bits: a (lon) en bits pares, b (lat) en impares."""
    r = 0
    for i in range(5):
        r |= ((a >> i) & 1) << (2 * i)
        r |= ((b >> i) & 1) << (2 * i + 1)
    return r


def local_idx(x, y):
    return zorder5(x & 31, y & 31)


def sub_idx(x, y):
    return zorder5((x >> 5) & 31, (y >> 5) & 31)


def gh3_id(x, y):
    return ((x >> 10) << 7) | (y >> 10)


def gh3_origin(gid):
    """Celda (x, y) de la esquina suroeste del gh3."""
    return (gid >> 7) << 10, (gid & 127) << 10


# ZPERM[i] = (lx, ly) de la celda con indice local i (vale igual para sub-indices)
ZPERM = np.zeros((1024, 2), dtype=np.int32)
for _lx in range(32):
    for _ly in range(32):
        ZPERM[zorder5(_lx, _ly)] = (_lx, _ly)
ZPERM_X = ZPERM[:, 0].copy()
ZPERM_Y = ZPERM[:, 1].copy()


def gh3_ids_for_bbox(lon0, lat0, lon1, lat1):
    x0, y0 = xy_from_gps(lon0, lat0)
    x1, y1 = xy_from_gps(lon1, lat1)
    ids = [(gx << 7) | gy
           for gx in range(x0 >> 10, (x1 >> 10) + 1)
           for gy in range(y0 >> 10, (y1 >> 10) + 1)]
    return sorted(ids), (x0, y0, x1, y1)
