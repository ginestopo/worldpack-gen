"""Extrae de un .osm.pbf lo que necesita el pack, ya rasterizado a celdas:
POIs, agua fina (rios/arroyos), sendas y zonas protegidas.

Salida: un .npz con claves de celda (key = x << 17 | y)."""
import numpy as np
import osmium
from osmium.geom import WKBFactory
import shapely.wkb
from rasterio import features
from rasterio.transform import Affine

from . import grid
from .taxonomy import poi_type, POI_KEYS, POI_MAX_V1

PATH_HW = {"path", "footway", "track", "bridleway", "steps", "cycleway"}
RIVER_WW = {"river", "canal"}
STREAM_WW = {"stream", "brook"}
FILTER_KEYS = sorted(set(POI_KEYS) | {"highway", "waterway", "boundary", "leisure", "protect_class"})


def cell_keys(x, y):
    return (np.asarray(x, np.int64) << 17) | np.asarray(y, np.int64)


def split_keys(k):
    k = np.asarray(k, np.int64)
    return (k >> 17).astype(np.int64), (k & 0x1FFFF).astype(np.int64)


def _to_cells(lon, lat):
    fx = (np.asarray(lon) + 180.0) / 360.0 * grid.GRID_W
    fy = (np.asarray(lat) + 90.0) / 180.0 * grid.GRID_H
    return fx, fy


class _Lines:
    def __init__(self):
        self.lon, self.lat, self.wid = [], [], []
        self.n = 0

    def add(self, way):
        try:
            pts = [(nd.lon, nd.lat) for nd in way.nodes]
        except osmium.InvalidLocationError:
            return
        if len(pts) < 2:
            return
        a = np.asarray(pts)
        self.lon.append(a[:, 0])
        self.lat.append(a[:, 1])
        self.wid.append(np.full(len(a), self.n, np.int64))
        self.n += 1

    def cells(self):
        """Devuelve (cell_key, way_idx) unicos para todas las lineas."""
        if not self.n:
            return np.zeros(0, np.int64), np.zeros(0, np.int64)
        fx, fy = _to_cells(np.concatenate(self.lon), np.concatenate(self.lat))
        wid = np.concatenate(self.wid)
        same = wid[1:] == wid[:-1]
        x0, y0, x1, y1 = fx[:-1][same], fy[:-1][same], fx[1:][same], fy[1:][same]
        w = wid[:-1][same]
        nseg = np.ceil(np.hypot(x1 - x0, y1 - y0) * 2).astype(np.int64) + 1
        rep = np.repeat(np.arange(len(nseg)), nseg)
        # t en [0,1] dentro de cada segmento
        first = np.repeat(np.cumsum(nseg) - nseg, nseg)
        t = (np.arange(len(rep)) - first) / np.maximum(np.repeat(nseg, nseg) - 1, 1)
        sx = np.floor(x0[rep] + (x1[rep] - x0[rep]) * t).astype(np.int64)
        sy = np.floor(y0[rep] + (y1[rep] - y0[rep]) * t).astype(np.int64)
        combo = np.unique((cell_keys(sx, sy) << 28) | w[rep])
        return combo >> 28, combo & ((1 << 28) - 1)


def _rasterize_polygon(geom, bbox_xy):
    minx, miny, maxx, maxy = geom.bounds
    x0, y0 = grid.xy_from_gps(minx, miny)
    x1, y1 = grid.xy_from_gps(maxx, maxy)
    x0, y0 = max(x0, bbox_xy[0]), max(y0, bbox_xy[1])
    x1, y1 = min(x1, bbox_xy[2]), min(y1, bbox_xy[3])
    if x1 < x0 or y1 < y0:
        return np.zeros(0, np.int64)
    w, h = x1 - x0 + 1, y1 - y0 + 1
    d = grid.CELL_DEG
    # fila 0 = norte
    tr = Affine(d, 0, x0 * d - 180.0, 0, -d, (y1 + 1) * d - 90.0)
    mask = features.rasterize([(geom, 1)], out_shape=(h, w), transform=tr,
                              dtype="uint8", all_touched=False)
    rows, cols = np.nonzero(mask)
    return cell_keys(x0 + cols, y1 - rows)


def _is_protected(tags):
    b = tags.get("boundary")
    return (b in ("national_park", "protected_area")
            or tags.get("leisure") == "nature_reserve")


def extract(pbf_path, out_npz, bbox, log=print):
    _, bbox_xy = grid.gh3_ids_for_bbox(*bbox)
    wkb = WKBFactory()
    paths, rivers, streams = _Lines(), _Lines(), _Lines()
    poi_x, poi_y, poi_t = [], [], []
    prot = []
    seen = 0

    def add_poi(lon, lat, typ):
        x, y = grid.xy_from_gps(lon, lat)
        if bbox_xy[0] <= x <= bbox_xy[2] and bbox_xy[1] <= y <= bbox_xy[3]:
            poi_x.append(x)
            poi_y.append(y)
            poi_t.append(typ)

    fp = (osmium.FileProcessor(str(pbf_path))
          .with_locations()
          .with_areas()
          .with_filter(osmium.filter.KeyFilter(*FILTER_KEYS)))
    for obj in fp:
        seen += 1
        if seen % 500000 == 0:
            log(f"  osm: {seen} objetos")
        tags = obj.tags
        if obj.is_node():
            t = poi_type(tags)
            if t:
                add_poi(obj.location.lon, obj.location.lat, t)
        elif obj.is_way():
            hw, ww = tags.get("highway"), tags.get("waterway")
            if hw in PATH_HW:
                paths.add(obj)
            if ww in RIVER_WW:
                rivers.add(obj)
            elif ww in STREAM_WW:
                streams.add(obj)
            if not obj.is_closed():
                t = poi_type(tags)
                if t and len(obj.nodes) >= 2:
                    try:
                        nd = obj.nodes[len(obj.nodes) // 2]
                        add_poi(nd.lon, nd.lat, t)
                    except osmium.InvalidLocationError:
                        pass
        elif obj.is_area():
            t = poi_type(tags)
            prot_ = _is_protected(tags)
            if not (t or prot_):
                continue
            try:
                geom = shapely.wkb.loads(wkb.create_multipolygon(obj), hex=True)
            except Exception:
                continue
            if t:
                c = geom.representative_point()
                add_poi(c.x, c.y, t)
            if prot_:
                prot.append(_rasterize_polygon(geom, bbox_xy))

    log(f"  osm: {seen} objetos leidos; rasterizando lineas")
    pk, pw = paths.cells()
    path_keys, path_cnt = np.unique(pk, return_counts=True)
    rk, _ = rivers.cells()
    sk, _ = streams.cells()
    water_keys = np.concatenate((np.unique(rk), np.unique(sk)))
    water_val = np.concatenate((np.full(len(np.unique(rk)), 2, np.uint8),
                                np.full(len(np.unique(sk)), 1, np.uint8)))
    # si una celda tiene rio y arroyo, gana el rio (orden estable: rio primero)
    water_keys, first = np.unique(water_keys, return_index=True)
    water_val = water_val[first]
    prot_keys = np.unique(np.concatenate(prot)) if prot else np.zeros(0, np.int64)

    poi_keys = cell_keys(poi_x, poi_y)
    poi_t = np.asarray(poi_t, np.int64)
    poi_counts = np.bincount(poi_t, minlength=POI_MAX_V1 + 1)
    np.savez_compressed(
        out_npz, bbox=np.asarray(bbox, np.float64),
        poi_keys=poi_keys, poi_types=poi_t, poi_counts=poi_counts,
        path_keys=path_keys, path_cnt=np.minimum(path_cnt, 3).astype(np.uint8),
        water_keys=water_keys, water_val=water_val, prot_keys=prot_keys)
    log(f"  osm: {len(poi_keys)} POIs, {len(path_keys)} celdas con senda, "
        f"{len(water_keys)} celdas con agua, {len(prot_keys)} celdas protegidas")
    return poi_counts
