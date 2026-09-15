"""Etapas 3 y 4: clasificacion y encoding de un gh3 completo."""
import numpy as np

from . import grid, fmt, rasters
from .osm import cell_keys, split_keys
from .taxonomy import WC_CLASSES, worldcover_to_layers, B_SEA

K = 4  # sobremuestreo de WorldCover: 16 subpixeles de ~38 m por celda


class OsmData:
    def __init__(self, npz_path=None):
        if npz_path is None:
            z = np.zeros(0, np.int64)
            self.poi_keys = self.path_keys = self.water_keys = self.prot_keys = z
            self.poi_types = z
            self.path_cnt = self.water_val = np.zeros(0, np.uint8)
            self.poi_counts = np.zeros(64, np.int64)
            return
        d = np.load(npz_path)
        order = np.argsort(d["poi_keys"], kind="stable")
        self.poi_keys = d["poi_keys"][order]
        self.poi_types = d["poi_types"][order]
        self.poi_counts = np.zeros(64, np.int64)
        self.poi_counts[:len(d["poi_counts"])] = d["poi_counts"]
        self.path_keys, self.path_cnt = d["path_keys"], d["path_cnt"]
        self.water_keys, self.water_val = d["water_keys"], d["water_val"]
        self.prot_keys = d["prot_keys"]

    @staticmethod
    def _slice(keys, gx0, gy0, *vals):
        lo = np.searchsorted(keys, int(cell_keys(gx0, 0)))
        hi = np.searchsorted(keys, int(cell_keys(gx0 + grid.GH3, 0)))
        x, y = split_keys(keys[lo:hi])
        m = (y >= gy0) & (y < gy0 + grid.GH3)
        return (x[m] - gx0, y[m] - gy0) + tuple(v[lo:hi][m] for v in vals)


def classify_gh3(gid, osm, wc=None, dem=None):
    """Devuelve (capas 1024x1024 con fila 0 al sur, pois) o (None, None) si es todo mar."""
    gx0, gy0 = grid.gh3_origin(gid)
    if wc is None:
        wc, wc_found = rasters.worldcover_gh3(gid, K)
    else:
        wc_found = True
    if dem is None:
        dem, _ = rasters.dem_gh3(gid)
    if not wc_found:
        return None, None

    k = wc.shape[0] // grid.GH3
    blocks = wc.reshape(grid.GH3, k, grid.GH3, k)
    counts = np.stack([(blocks == c).sum(axis=(1, 3), dtype=np.uint16) for c in WC_CLASSES])
    biome, water, built = worldcover_to_layers(counts, dem, np.zeros(dem.shape, bool))

    paths = np.zeros_like(biome)
    flags = np.zeros_like(biome)
    lx, ly, cnt = osm._slice(osm.path_keys, gx0, gy0, osm.path_cnt)
    paths[ly, lx] = cnt
    lx, ly, wv = osm._slice(osm.water_keys, gx0, gy0, osm.water_val)
    cur = water[ly, lx]
    water[ly, lx] = np.where(cur == 3, 3, np.maximum(cur, wv))
    lx, ly = osm._slice(osm.prot_keys, gx0, gy0)
    flags[ly, lx] |= 1

    # POIs: una por celda, gana la mas rara
    lx, ly, pt = osm._slice(osm.poi_keys, gx0, gy0, osm.poi_types)
    rarity = osm.poi_counts[pt]
    order = np.lexsort((pt, rarity))
    lx, ly, pt, rarity = lx[order], ly[order], pt[order], rarity[order]
    _, first = np.unique(ly * grid.GH3 + lx, return_index=True)
    pois = (lx[first], ly[first], pt[first], rarity[first])

    elev = dem
    return {"biome": biome, "elev": elev, "water": water, "built": built,
            "paths": paths, "flags": flags}, pois


def encode_gh3(gid, layers, pois, stats):
    """Parte el gh3 en 1024 chunks y los codifica. Devuelve [(gid, sub_idx, blob)]."""
    plx, ply, ptyp, prar = pois
    pcx, pcy = plx >> 5, ply >> 5
    out = []
    for cy in range(32):
        for cx in range(32):
            sl = (slice(cy * 32, cy * 32 + 32), slice(cx * 32, cx * 32 + 32))
            ch = {n: layers[n][sl][grid.ZPERM_Y, grid.ZPERM_X] for n in fmt.LAYER_NAMES}
            m = (pcx == cx) & (pcy == cy)
            if (ch["biome"] == B_SEA).all() and (ch["water"] == 3).all() and not m.any():
                stats["sea_chunks"] = stats.get("sea_chunks", 0) + 1
                continue
            idx = [grid.zorder5(int(a) & 31, int(b) & 31) for a, b in zip(plx[m], ply[m])]
            cand = sorted(zip(prar[m].tolist(), ptyp[m].tolist(), idx))
            chunk_pois = [(i, t) for _, t, i in cand][:fmt.POI_MAX]
            stats["pois_dropped"] = stats.get("pois_dropped", 0) + max(0, len(cand) - fmt.POI_MAX)
            blob = fmt.encode_blob(ch, chunk_pois, stats.setdefault("modes", {}))
            out.append((gid, grid.zorder5(cx, cy), blob))
            hist = stats.setdefault("biome_cells", {})
            for b, c in zip(*np.unique(ch["biome"], return_counts=True)):
                hist[int(b)] = hist.get(int(b), 0) + int(c)
            pw = stats.setdefault("poi_written", {})
            for _, t in chunk_pois:
                pw[int(t)] = pw.get(int(t), 0) + 1
    return out
