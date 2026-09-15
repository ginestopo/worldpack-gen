"""Taxonomias del capitulo 4.

Version sin CORINE: el bioma sale solo de ESA WorldCover (11 clases), asi que
de momento no se distinguen frondosas/coniferas, vinedo/olivar, etc.
"""
import numpy as np

# ---- WorldCover -> bioma ----
WC_CLASSES = (0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100)
WC_TREE, WC_SHRUB, WC_GRASS, WC_CROP, WC_BUILT = 10, 20, 30, 40, 50
WC_BARE, WC_SNOW, WC_WATER, WC_WETLAND, WC_MANGROVE, WC_MOSS = 60, 70, 80, 90, 95, 100

B_UNKNOWN, B_SEA, B_LAKE, B_RIVER, B_WETLAND = 0, 1, 2, 3, 4
B_BEACH, B_ROCK, B_GLACIER, B_TUNDRA = 6, 7, 8, 9
B_FOREST_MIXED, B_SHRUB_TRANS = 12, 13
B_GRASS, B_CROP = 16, 17
B_URBAN_DENSE, B_URBAN_SPARSE = 23, 24
BIOME_MAX_V1 = 25

WC_TO_BIOME = {
    WC_TREE: B_FOREST_MIXED, WC_SHRUB: B_SHRUB_TRANS, WC_GRASS: B_GRASS,
    WC_CROP: B_CROP, WC_BARE: B_ROCK, WC_SNOW: B_GLACIER, WC_WETLAND: B_WETLAND,
    WC_MANGROVE: B_WETLAND, WC_MOSS: B_TUNDRA,
}

# ---- POIs: (tipo, [(clave, valor | None)]) ----  None = cualquier valor
POI_RULES = [
    (1, [("historic", "castle"), ("historic", "fort"), ("castle_type", None)]),
    (8, [("historic", "roman_road"), ("historic:civilization", "roman")]),
    (3, [("historic", "archaeological_site")]),
    (4, [("megalith_type", None), ("historic", "megalith")]),
    (5, [("man_made", "watchtower"), ("historic", "tower"), ("tower:type", "defensive")]),
    (6, [("historic", "city_walls"), ("barrier", "city_wall")]),
    (7, [("bridge", "aqueduct"), ("historic", "aqueduct")]),
    (9, [("military", "bunker"), ("historic", "bunker"), ("building", "bunker")]),
    (10, [("historic", "mine"), ("abandoned:man_made", "mineshaft"), ("disused:man_made", "mineshaft")]),
    (12, [("abandoned:place", None), ("historic", "village")]),
    (2, [("historic", "ruins"), ("ruins", "yes")]),
    (14, [("building", "cathedral"), ("building", "basilica")]),
    (15, [("amenity", "monastery"), ("building", "monastery"), ("historic", "monastery")]),
    (16, [("building", "chapel"), ("historic", "wayside_chapel"), ("place_of_worship", "chapel")]),
    (17, [("historic", "wayside_cross"), ("historic", "wayside_shrine")]),
    (18, [("amenity", "grave_yard"), ("landuse", "cemetery")]),
    (19, [("religion", "muslim")]),
    (20, [("religion", "jewish")]),
    (13, [("building", "church"), ("amenity", "place_of_worship")]),
    (21, [("natural", "cave_entrance")]),
    (22, [("waterway", "waterfall"), ("natural", "waterfall")]),
    (23, [("natural", "spring")]),
    (29, [("natural", "hot_spring")]),
    (24, [("tourism", "viewpoint")]),
    (25, [("natural", "tree"), ("denotation", "natural_monument")]),
    (30, [("natural", "volcano")]),
    (26, [("natural", "peak")]),
    (27, [("natural", "gorge"), ("natural", "canyon")]),
    (28, [("natural", "rock"), ("natural", "stone"), ("natural", "arch")]),
    (31, [("natural", "cliff")]),
    (32, [("place", "island"), ("place", "islet")]),
    (33, [("man_made", "lighthouse")]),
    (34, [("harbour", None), ("leisure", "marina"), ("man_made", "pier")]),
    (35, [("leisure", "beach_resort")]),
    (36, [("waterway", "dam"), ("man_made", "dam")]),
    (37, [("waterway", "lock_gate"), ("lock", "yes")]),
    (38, [("man_made", "watermill")]),
    (39, [("man_made", "windmill")]),
    (40, [("place", "farm"), ("building", "farm")]),
    (41, [("tourism", "alpine_hut"), ("tourism", "wilderness_hut"), ("amenity", "shelter")]),
    (42, [("historic", "boundary_stone"), ("historic", "milestone")]),
    (44, [("ford", "yes")]),
    (45, [("amenity", "watering_place"), ("man_made", "water_well")]),
    (46, [("building", "dovecote"), ("man_made", "dovecote")]),
    (48, [("amenity", "marketplace")]),
    (49, [("railway", "station")]),
    (50, [("tourism", "artwork"), ("historic", "memorial")]),
    (51, [("man_made", "observatory")]),
    (52, [("leisure", "garden"), ("garden:type", "botanical")]),
    (47, [("place", "square")]),
]
POI_MAX_V1 = 52

# Claves que el filtro de osmium debe conservar para nodos/areas de POIs
POI_KEYS = sorted({k for _, rules in POI_RULES for k, _ in rules})


def poi_type(tags):
    """tags: dict-like. Devuelve el tipo o 0. Respeta el orden de POI_RULES."""
    # descartes: POIs modernos que no queremos aunque coincidan
    if tags.get("railway") == "station" and tags.get("station") == "subway":
        return 0
    if tags.get("amenity") == "shelter" and (
            tags.get("shelter_type") in ("public_transport", "picnic_shelter", "sun_shelter",
                                         "field_shelter", "changing_rooms")
            or tags.get("public_transport") or tags.get("bus") == "yes"):
        return 0  # marquesinas y similares, no refugios
    if tags.get("leisure") == "garden" and tags.get("garden:type") != "botanical":
        return 0
    for typ, rules in POI_RULES:
        for k, v in rules:
            val = tags.get(k)
            if val is not None and (v is None or val == v):
                if typ == 25 and k == "natural" and tags.get("denotation") not in (
                        "natural_monument", "landmark"):
                    continue  # solo arboles singulares, no cada arbol mapeado
                if typ == 12 and k == "historic" and tags.get("abandoned") != "yes":
                    continue
                return typ
    return 0


def worldcover_to_layers(counts, elev, sea_hint):
    """counts: (len(WC_CLASSES), H, W) conteo de subpixeles por clase.
    Devuelve biome, water(base), built (arrays HxW)."""
    total = counts.sum(axis=0)
    total_nz = np.maximum(total, 1)
    cls_idx = counts.argmax(axis=0)
    cls = np.asarray(WC_CLASSES)[cls_idx]
    lut = np.zeros(256, dtype=np.uint8)
    for k, v in WC_TO_BIOME.items():
        lut[k] = v
    biome = lut[cls]

    built_frac = counts[WC_CLASSES.index(WC_BUILT)] / total_nz
    built = np.clip(np.rint(built_frac * 7), 0, 7).astype(np.uint8)
    urban = cls == WC_BUILT
    biome[urban & (built_frac >= 0.5)] = B_URBAN_DENSE
    biome[urban & (built_frac < 0.5)] = B_URBAN_SPARSE

    water_frac = counts[WC_CLASSES.index(WC_WATER)] / total_nz
    nodata = (cls == 0) | (total == 0)
    is_water = (cls == WC_WATER) | nodata
    sea = is_water & ((elev <= 0) | nodata | sea_hint)
    biome[is_water & ~sea] = B_LAKE
    biome[sea] = B_SEA
    water = np.zeros_like(biome)
    water[(water_frac >= 0.5) & ~sea] = 2
    water[sea] = 3
    return biome, water, built
