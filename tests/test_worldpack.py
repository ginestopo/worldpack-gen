import math
import struct

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from worldpack import grid, fmt, container, rasters
from worldpack.build import OsmData, classify_gh3, encode_gh3
from worldpack.taxonomy import poi_type

RNG = np.random.default_rng(1234)


# ---------- capitulo 5: el ejemplo del spec ----------
def test_spec_example_indices():
    x, y = grid.xy_from_gps(-0.5, 41.5)
    assert (x, y) == (130707, 95755)
    assert grid.gh3_id(x, y) == 16349
    assert grid.sub_idx(x, y) == 784
    assert grid.local_idx(x, y) == 399
    assert (399 & 0x3FF) | (5 << 10) == 0x158F


# ---------- test 5: paridad con geohash base32 ----------
B32 = "0123456789bcdefghjkmnpqrstuvwxyz"


def geohash_ref(lat, lon, prec=7):
    lat_r, lon_r = [-90.0, 90.0], [-180.0, 180.0]
    bits, out, even, ch = 0, "", True, 0
    while len(out) < prec:
        r, v = (lon_r, lon) if even else (lat_r, lat)
        mid = (r[0] + r[1]) / 2
        if v >= mid:
            ch = (ch << 1) | 1
            r[0] = mid
        else:
            ch <<= 1
            r[1] = mid
        even = not even
        bits += 1
        if bits == 5:
            out += B32[ch]
            bits = ch = 0
    return out


def geohash_from_xy(x, y):
    v = 0
    for i in range(35):  # empieza por longitud
        if i % 2 == 0:
            v = (v << 1) | ((x >> (17 - i // 2)) & 1)
        else:
            v = (v << 1) | ((y >> (16 - i // 2)) & 1)
    return "".join(B32[(v >> (30 - 5 * k)) & 31] for k in range(7))


@pytest.mark.parametrize("lon,lat", [(-0.5, 41.5), (21.0122, 52.2297), (14.0, 49.0),
                                     (-9.14, 38.72), (24.94, 60.17), (0.0, 0.0),
                                     (-3.7038, 40.4168), (19.94, 50.06)])
def test_geohash_parity(lon, lat):
    x, y = grid.xy_from_gps(lon, lat)
    gh = geohash_ref(lat, lon)
    assert geohash_from_xy(x, y) == gh
    # prefijos: gh3 y chunk coinciden con los primeros caracteres
    assert geohash_from_xy(x, y)[:3] == gh[:3]


# ---------- test 1: round-trip sintetico forzando cada modo ----------
def _pattern(kind, bits):
    top = (1 << bits) - 1
    if kind == "const":
        return np.full(1024, top), fmt.M_CONST
    if kind == "two":
        return RNG.choice([0, top], 1024), fmt.M_PAL1
    if kind == "four":
        return RNG.choice([0, 1, 2, top], 1024), fmt.M_PAL2
    if kind == "sixteen":
        return RNG.choice(np.arange(16) % (top + 1) if bits > 4 else [0], 1024), fmt.M_PAL4
    if kind == "runs":
        return np.repeat([1, top, 1], [300, 300, 424]), fmt.M_RLE
    if kind == "noise":
        return RNG.integers(0, top + 1, 1024), fmt.M_RAW


@pytest.mark.parametrize("kind", ["const", "two", "four", "sixteen", "runs", "noise"])
def test_roundtrip_modes(kind):
    v, expect = _pattern(kind, 6)
    enc = fmt.encode_layer(v, 6)
    assert enc[0] == expect
    dec, off = fmt.decode_layer(enc, 0, 6)
    assert off == len(enc)
    assert np.array_equal(dec, v)


def test_roundtrip_blob():
    for _ in range(50):
        layers = {n: RNG.integers(0, 1 << b, 1024) for n, b in zip(fmt.LAYER_NAMES, fmt.LAYER_BITS)}
        base = int(RNG.integers(0, 3000))
        layers["elev"] = base + RNG.integers(0, 63, 1024) * 4
        layers["water"][:10] = 3
        pois = [(int(i), int(t)) for i, t in zip(RNG.integers(0, 1024, 5), RNG.integers(1, 64, 5))]
        pois = list({i: (i, t) for i, t in pois}.values())
        blob = fmt.encode_blob(layers, pois)
        d = fmt.decode_blob(blob)
        for n in fmt.LAYER_NAMES:
            if n == "elev":
                assert np.array_equal(d["elev"], layers["elev"])
            else:
                assert np.array_equal(d[n], layers[n]), n
        assert d["pois"] == pois


# ---------- test 2: barrido de bits ----------
@pytest.mark.parametrize("bits", sorted(set(fmt.LAYER_BITS)))
def test_bit_sweep(bits):
    vals = np.arange(1 << bits)
    for pos in range(0, 1024, 97):
        for val in vals:
            for other in (0, (1 << bits) - 1):
                v = np.full(1024, other)
                v[pos] = val
                for enc in _all_modes(v, bits):
                    dec, _ = fmt.decode_layer(enc, 0, bits)
                    assert dec[pos] == val and np.array_equal(dec, v)


def _all_modes(v, bits):
    """Codifica en todos los modos aplicables, no solo el ganador."""
    out = [bytes([fmt.M_RAW]) + fmt.pack_bits(v, bits)]
    uniq = np.unique(v)
    for mode, (psize, pbits) in fmt.PAL_SIZE.items():
        if len(uniq) <= psize and pbits < bits:
            pal = np.concatenate((uniq, np.full(psize - len(uniq), uniq[0])))
            out.append(bytes([mode]) + pal.astype(np.uint8).tobytes()
                       + fmt.pack_bits(np.searchsorted(uniq, v), pbits))
    out.append(fmt.encode_layer(v, bits))
    return out


def test_elevation_quantization():
    for rng_m in (0, 10, 63, 64, 500, 2016, 4000, 8064):
        e = 1000 + np.linspace(0, rng_m, 1024).astype(int)
        packed, off = fmt.quantize_elevation(e)
        base, step = fmt.elev_unpack(packed)
        assert base == 1000 and rng_m <= 63 * step and step >= fmt.ELEV_MIN_STEP
        assert np.abs(base + off * step - e).max() <= step // 2
    packed, _ = fmt.quantize_elevation(np.full(1024, -5))
    assert fmt.elev_unpack(packed)[0] == 0  # bajo el nivel del mar se recorta


# ---------- pack sintetico con capas conocidas ----------
def _synthetic_pack(tmp_path):
    gids = [16349, 16350]
    entries, truth = [], {}
    for gid in gids:
        gx0, gy0 = grid.gh3_origin(gid)
        for sidx in RNG.choice(1024, 40, replace=False):
            layers = {n: RNG.integers(0, 1 << b, 1024) for n, b in zip(fmt.LAYER_NAMES, fmt.LAYER_BITS)}
            layers["elev"] = RNG.integers(0, 60, 1024)
            if sidx % 3 == 0:
                layers["biome"][:] = 17
            entries.append((gid, int(sidx), fmt.encode_blob(layers, [(5, 7)])))
            cx, cy = grid.ZPERM[sidx]
            truth[(gx0 + cx * 32, gy0 + cy * 32)] = layers
    # duplicados para probar dedup
    entries.append((gids[0], 1023, entries[0][2]))
    path = tmp_path / "t.pack"
    info = container.assemble(entries, path, (0, 0, 1, 1), 20260915, 25, 52)
    assert info["dedup_hits"] >= 1
    return path, truth


def test_pack_lookup(tmp_path):
    path, truth = _synthetic_pack(tmp_path)
    p = container.Pack(path)
    for (x0, y0), layers in list(truth.items())[:20]:
        for i in (0, 399, 1023):
            lx, ly = grid.ZPERM[i]
            c = p.at(x0 + lx, y0 + ly)
            assert c["biome"] == layers["biome"][i]
            assert abs(c["elevation"] - layers["elev"][i]) <= fmt.ELEV_MIN_STEP // 2
            assert c["poi"] == (7 if i == 5 else 0)


# ---------- test 3: frontera de chunk ----------
def test_chunk_boundary(tmp_path):
    path, truth = _synthetic_pack(tmp_path)
    p = container.Pack(path)
    x0, y0 = next(iter(truth))
    for cx, cy in ((x0, y0), (x0 + 32, y0), (x0, y0 + 32), (x0 + 32, y0 + 32)):
        g = p.grid(cx, cy, 9)
        ref = [p.at(cx + dx, cy + dy) for dy in range(-4, 5) for dx in range(-4, 5)]
        assert g == ref
    # coste de SD: celdas del mismo chunk no generan lecturas nuevas
    p2 = container.Pack(path)
    p2.at(x0, y0)
    r = p2.sd_reads
    p2.at(x0 + 31, y0 + 31)
    assert p2.sd_reads == r


# ---------- test 6: pack degradado ----------
def test_degraded(tmp_path):
    path, _ = _synthetic_pack(tmp_path)
    data = path.read_bytes()
    cases = {
        "trunc": data[:len(data) // 2],
        "crc": data[:-1] + bytes([data[-1] ^ 0xFF]),
        "version": data[:4] + struct.pack("<H", 99) + data[6:],
        "tiny": data[:10],
    }
    for name, d in cases.items():
        bad = tmp_path / f"{name}.pack"
        bad.write_bytes(d)
        with pytest.raises(container.PackError):
            container.Pack(bad)
    with pytest.raises(FileNotFoundError):
        container.Pack(tmp_path / "nope.pack")


# ---------- taxonomia OSM ----------
def test_poi_rules():
    assert poi_type({"historic": "castle"}) == 1
    assert poi_type({"man_made": "watchtower"}) == 5
    assert poi_type({"amenity": "place_of_worship", "religion": "jewish"}) == 20
    assert poi_type({"amenity": "place_of_worship", "religion": "christian"}) == 13
    assert poi_type({"natural": "tree"}) == 0
    assert poi_type({"natural": "tree", "denotation": "natural_monument"}) == 25
    assert poi_type({"amenity": "hospital"}) == 0
    assert poi_type({"amenity": "shelter", "shelter_type": "basic_hut"}) == 41
    assert poi_type({"amenity": "shelter", "shelter_type": "public_transport"}) == 0
    assert poi_type({"amenity": "shelter", "public_transport": "platform"}) == 0


# ---------- extremo a extremo con GeoTIFF locales ----------
def _write_tile(path, arr, lon0, lat1, res, dtype, nodata):
    with rasterio.open(path, "w", driver="GTiff", width=arr.shape[1], height=arr.shape[0],
                       count=1, dtype=dtype, crs="EPSG:4326", nodata=nodata,
                       transform=from_origin(lon0, lat1, res, res)) as ds:
        ds.write(arr.astype(dtype), 1)


def test_end_to_end(tmp_path, monkeypatch):
    gid = grid.gh3_id(*grid.xy_from_gps(21.0, 52.2))
    gx0, gy0 = grid.gh3_origin(gid)
    lon0 = gx0 * grid.CELL_DEG - 180
    lat0 = gy0 * grid.CELL_DEG - 90
    # WorldCover sintetico a ~50 m: mitad oeste bosque, este cultivo, franja urbana, un lago
    res = 1 / 2000
    n = int(3 / res)
    tl_lat, tl_lon = 51, 18  # tesela 3x3 que contiene el gh3
    lon = tl_lon + (np.arange(n) + 0.5) * res
    lat = tl_lat + 3 - (np.arange(n) + 0.5) * res
    LON, LAT = np.meshgrid(lon, lat)
    wc = np.where(LON < 20.4, 10, 40)
    wc[(LAT > 52.15) & (LAT < 52.25) & (LON > 20.6) & (LON < 20.8)] = 50
    wc[(LAT > 52.4) & (LAT < 52.45) & (LON > 20.2) & (LON < 20.3)] = 80
    _write_tile(tmp_path / "wc_N51E018.tif", wc, tl_lon, tl_lat + 3, res, "uint8", 0)
    # DEM a 1/300 de grado con pendiente
    for tlat in range(51, 55):
        for tlon in range(18, 24):
            m = 300
            dem = np.full((m, m), 100.0) + np.linspace(0, 200, m)[None, :]
            _write_tile(tmp_path / f"dem_N{tlat}E{tlon:03d}.tif", dem, tlon, tlat + 1, 1 / m,
                        "float32", -9999)
    monkeypatch.setattr(rasters, "WORLDCOVER", str(tmp_path / "wc_{NS}{alat:02d}{EW}{alon:03d}.tif"))
    monkeypatch.setattr(rasters, "DEM", str(tmp_path / "dem_{NS}{alat:02d}{EW}{alon:03d}.tif"))
    rasters._missing.clear()

    osm = OsmData()
    wx, wy = grid.xy_from_gps(20.0, 52.3)
    osm.poi_keys = np.array([(wx << 17) | wy], np.int64)
    osm.poi_types = np.array([1])
    osm.poi_counts[1] = 3

    layers, pois = classify_gh3(gid, osm)
    stats = {}
    entries = encode_gh3(gid, layers, pois, stats)
    path = tmp_path / "e2e.pack"
    container.assemble(entries, path, (gx0, gy0, gx0 + 1023, gy0 + 1023), 20260915, 25, 52)
    p = container.Pack(path)

    assert p.at_gps(20.0, 52.3)["biome"] == 12       # bosque
    assert p.at_gps(20.0, 52.3)["poi"] == 1          # castillo
    assert p.at_gps(20.9, 52.3)["biome"] == 17       # cultivo
    assert p.at_gps(20.7, 52.2)["biome"] == 23       # urbano denso
    assert p.at_gps(20.7, 52.2)["built"] == 7
    assert p.at_gps(21.05, 52.3)["biome"] == 1       # sin tesela WorldCover = mar
    lake = p.at_gps(20.25, 52.42)
    assert lake["biome"] == 2 and lake["water"] == 2
    e_w, e_e = p.at_gps(19.8, 52.3)["elevation"], p.at_gps(20.9, 52.3)["elevation"]
    assert 100 <= e_w < e_e <= 300
    assert abs(e_e - (100 + 0.9 * 200)) <= 3
    assert stats["modes"]["biome"].get(fmt.M_CONST, 0) > 0


# ---------- hemisferio sur y oeste ----------
@pytest.mark.parametrize("lon,lat,wc_tile,dem_tile", [
    (21.0, 52.2, "N51E021", "N52_00_E021_00"),      # Varsovia
    (-74.07, 4.71, "N03W075", "N04_00_W075_00"),    # Bogota
    (-69.94, -4.21, "S06W072", "S05_00_W070_00"),   # Leticia
    (0.5, -0.5, "S03E000", "S01_00_E000_00"),       # justo al sur del ecuador
])
def test_tile_url_hemispheres(lon, lat, wc_tile, dem_tile):
    """Ambas fuentes nombran la tesela por su esquina SUROESTE, asi que al sur
    del ecuador o al oeste de Greenwich el indice es el floor, no el truncado."""
    for tmpl, deg, tok in ((rasters.WORLDCOVER, 3, wc_tile), (rasters.DEM, 1, dem_tile)):
        tlat = int(math.floor(lat / deg) * deg)
        tlon = int(math.floor(lon / deg) * deg)
        assert tok in rasters._tile_url(tmpl, tlat, tlon)


def test_end_to_end_southern(tmp_path, monkeypatch):
    """Mismo recorrido que test_end_to_end pero al sur del ecuador y al oeste de
    Greenwich, donde el bbox de la ventana y los indices de tesela son negativos."""
    gid = grid.gh3_id(*grid.xy_from_gps(-69.94, -4.21))
    gx0, gy0 = grid.gh3_origin(gid)
    lon0 = gx0 * grid.CELL_DEG - 180
    lat0 = gy0 * grid.CELL_DEG - 90
    lon1 = lon0 + grid.GH3 * grid.CELL_DEG
    lat1 = lat0 + grid.GH3 * grid.CELL_DEG
    mid = (lon0 + lon1) / 2

    def tile_name(tlat, tlon):
        return (f"{'N' if tlat >= 0 else 'S'}{abs(tlat):02d}"
                f"{'E' if tlon >= 0 else 'W'}{abs(tlon):03d}")

    def tile_range(v0, v1, deg):
        return range(int(math.floor(v0 / deg) * deg), int(math.ceil(v1)), deg)

    # WorldCover 3x3 a ~110 m: mitad oeste selva, mitad este cultivo, un rio al norte
    res, lake_lat = 1 / 1000, lat1 - 0.08
    n = int(3 / res)
    for tlat in tile_range(lat0, lat1, 3):
        for tlon in tile_range(lon0, lon1, 3):
            LON, LAT = np.meshgrid(tlon + (np.arange(n) + 0.5) * res,
                                   tlat + 3 - (np.arange(n) + 0.5) * res)
            wc = np.where(LON < mid, 10, 40).astype("uint8")
            wc[(LAT > lake_lat) & (LAT < lake_lat + 0.04) & (LON > mid) & (LON < mid + 0.2)] = 80
            _write_tile(tmp_path / f"wc_{tile_name(tlat, tlon)}.tif", wc,
                        tlon, tlat + 3, res, "uint8", 0)
            del LON, LAT, wc
    # DEM de 1 grado: rampa de 100 a 300 m de oeste a este dentro de cada tesela
    m = 300
    for tlat in tile_range(lat0, lat1, 1):
        for tlon in tile_range(lon0, lon1, 1):
            dem = np.full((m, m), 100.0) + np.linspace(0, 200, m)[None, :]
            _write_tile(tmp_path / f"dem_{tile_name(tlat, tlon)}.tif", dem,
                        tlon, tlat + 1, 1 / m, "float32", -9999)
    monkeypatch.setattr(rasters, "WORLDCOVER", str(tmp_path / "wc_{NS}{alat:02d}{EW}{alon:03d}.tif"))
    monkeypatch.setattr(rasters, "DEM", str(tmp_path / "dem_{NS}{alat:02d}{EW}{alon:03d}.tif"))
    rasters._missing.clear()

    osm = OsmData()
    wx, wy = grid.xy_from_gps(lon0 + 0.3, lat0 + 0.3)
    osm.poi_keys = np.array([(wx << 17) | wy], np.int64)
    osm.poi_types = np.array([30])  # volcan
    osm.poi_counts[30] = 2

    layers, pois = classify_gh3(gid, osm)
    assert layers is not None
    entries = encode_gh3(gid, layers, pois, {})
    path = tmp_path / "sur.pack"
    container.assemble(entries, path, (gx0, gy0, gx0 + 1023, gy0 + 1023), 20260915, 25, 52)
    p = container.Pack(path)

    assert p.at_gps(lon0 + 0.3, lat0 + 0.3)["biome"] == 12   # bosque
    assert p.at_gps(lon0 + 0.3, lat0 + 0.3)["poi"] == 30
    assert p.at_gps(lon1 - 0.3, lat0 + 0.3)["biome"] == 17   # cultivo
    lake = p.at_gps(mid + 0.1, lake_lat + 0.02)
    assert lake["biome"] == 2 and lake["water"] == 2         # agua interior, no mar
    lon_q = lon0 + 0.3
    frac = lon_q - math.floor(lon_q)
    assert abs(p.at_gps(lon_q, lat0 + 0.3)["elevation"] - (100 + frac * 200)) <= 4


def test_elevation_above_base_ceiling():
    """Colombia pasa de 4095 m (el techo de la base de 12 bits): por encima solo
    se pierde resolucion, nunca se recorta el valor."""
    for lo, hi in ((4800, 5000), (5200, 5775), (0, 5775)):
        e = np.linspace(lo, hi, 1024).astype(int)
        packed, off = fmt.quantize_elevation(e)
        base, step = fmt.elev_unpack(packed)
        assert off.max() <= 63
        assert np.abs(base + off.astype(np.int64) * step - e).max() <= step // 2 + 1


# ---------- tabla de regiones ----------
def test_region_table():
    from worldpack.cli import REGIONS
    for name, r in REGIONS.items():
        lon0, lat0, lon1, lat1 = r["bbox"]
        assert lon0 < lon1 and lat0 < lat1, name
        ids, _ = grid.gh3_ids_for_bbox(*r["bbox"])
        assert 1 <= r["shards"] <= len(ids), name
        assert r["pbf"].endswith("-latest.osm.pbf"), name
        for lon, lat, label in r["checks"]:
            assert lon0 <= lon <= lon1 and lat0 <= lat <= lat1, (name, label)
            assert " " not in label, label
