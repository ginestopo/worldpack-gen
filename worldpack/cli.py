"""worldpackgen: python -m worldpack.cli <subcomando> ..."""
import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

from . import grid, container, osm as osm_mod
from .build import OsmData, classify_gh3, encode_gh3
from .taxonomy import BIOME_MAX_V1, POI_MAX_V1

REGIONS = {
    "poland": {"bbox": (14.0, 49.0, 24.2, 55.0),
               "pbf": "https://download.geofabrik.de/europe/poland-latest.osm.pbf",
               "shards": 4,
               "checks": [(21.0122, 52.2297, "Varsovia"), (20.0883, 49.1794, "Rysy"),
                          (19.9450, 50.0647, "Cracovia"), (18.6466, 54.3520, "Gdansk")]},
    # El bbox llega hasta 82 W para incluir San Andres y Providencia; el mar
    # sobrante no cuesta casi nada porque los gh3 sin tesela se omiten.
    "colombia": {"bbox": (-82.0, -4.3, -66.8, 13.5),
                 "pbf": "https://download.geofabrik.de/south-america/colombia-latest.osm.pbf",
                 "shards": 12,
                 "checks": [(-74.0721, 4.7110, "Bogota"), (-75.5636, 6.2518, "Medellin"),
                            (-75.3222, 4.8925, "Nevado_del_Ruiz"),
                            (-75.5144, 10.3910, "Cartagena"),
                            (-69.9406, -4.2153, "Leticia"),
                            (-81.7006, 12.5847, "San_Andres")]},
    "zaragoza": {"bbox": (-1.2, 41.4, -0.6, 41.8),
                 "pbf": "https://download.geofabrik.de/europe/spain/aragon-latest.osm.pbf",
                 "shards": 1,
                 "checks": [(-0.8773, 41.6488, "Zaragoza")]},
}
REGION_FIELDS = ("bbox", "pbf", "shards", "shard-matrix", "checks")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_bbox(args):
    if args.bbox:
        return tuple(float(v) for v in args.bbox.split(","))
    if args.region:
        return REGIONS[args.region]["bbox"]
    sys.exit("hace falta --bbox o --region")


def merge_stats(into, s):
    for k, v in s.items():
        if isinstance(v, dict):
            merge_stats(into.setdefault(k, {}), v)
        else:
            into[k] = into.get(k, 0) + v


def cmd_region(args):
    r = REGIONS[args.region]
    if args.field == "shard-matrix":
        print(json.dumps(list(range(r["shards"]))))
        return
    if args.field == "checks":
        for lon, lat, name in r["checks"]:
            print(f"{lon} {lat} {name}")
        return
    val = r[args.field]
    print(",".join(map(str, val)) if isinstance(val, tuple) else val)


def cmd_osm(args):
    bbox = parse_bbox(args)
    log(f"extrayendo OSM de {args.pbf}")
    osm_mod.extract(args.pbf, args.out, bbox, log)


def run_shard(bbox, osm_npz, shard, shards, log_=log, allow_empty=False):
    ids, _ = grid.gh3_ids_for_bbox(*bbox)
    mine = ids[shard::shards]
    osm = OsmData(osm_npz)
    entries, stats = [], {}
    for i, gid in enumerate(mine):
        t = time.time()
        layers, pois = classify_gh3(gid, osm)
        if layers is None:
            stats["sea_gh3"] = stats.get("sea_gh3", 0) + 1
            log_(f"gh3 {gid} ({i + 1}/{len(mine)}): mar, omitido")
            continue
        e = encode_gh3(gid, layers, pois, stats)
        entries += e
        log_(f"gh3 {gid} ({i + 1}/{len(mine)}): {len(e)} chunks, "
             f"{sum(len(b) for _, _, b in e) / 1e6:.2f} MB, {time.time() - t:.0f}s")
    if mine and stats.get("sea_gh3", 0) == len(mine) and not allow_empty:
        raise SystemExit("ningun gh3 con datos de WorldCover: revisa el acceso a las fuentes "
                         "(o usa --allow-empty si el shard es realmente todo mar)")
    return entries, stats


def cmd_probe(args):
    from . import rasters
    print(json.dumps(rasters.probe(args.lon, args.lat), indent=2))


def cmd_filter_expr(args):
    from .taxonomy import POI_RULES
    from .osm import PATH_HW
    exprs = {}
    for _, rules in POI_RULES:
        for k, v in rules:
            cur = exprs.setdefault(k, set())
            if v is None or cur is None:
                exprs[k] = None
            else:
                cur.add(v)
    out = []
    for k, vals in sorted(exprs.items()):
        if k == "highway":
            continue
        out.append(f"nwr/{k}" if vals is None else f"nwr/{k}=" + ",".join(sorted(vals)))
    out += ["w/highway=" + ",".join(sorted(PATH_HW)), "w/waterway",
            "wr/boundary=national_park,protected_area", "wr/leisure=nature_reserve"]
    print(" ".join(out))


def cmd_shard(args):
    bbox = parse_bbox(args)
    entries, stats = run_shard(bbox, args.osm, args.shard, args.shards,
                               allow_empty=args.allow_empty)
    container.write_part(args.out, entries)
    Path(args.out + ".json").write_text(json.dumps(stats))


def write_report(path, bbox, info, stats, osm_npz):
    import numpy as np
    rep = {"bbox": bbox, "pack": info, **stats}
    if osm_npz:
        c = np.load(osm_npz)["poi_counts"]
        rep["poi_extracted"] = {int(i): int(v) for i, v in enumerate(c) if v}
    modes = rep.get("modes", {})
    from .fmt import MODE_NAMES
    rep["modes"] = {layer: {MODE_NAMES[int(m)]: n for m, n in d.items()} for layer, d in modes.items()}
    Path(path).write_text(json.dumps(rep, indent=2, sort_keys=True))


def content_version(args):
    return int(args.content_version or dt.date.today().strftime("%Y%m%d"))


def cmd_assemble(args):
    bbox = parse_bbox(args)
    _, bbox_xy = grid.gh3_ids_for_bbox(*bbox)
    stats = {}
    entries = []
    for p in args.parts:
        entries += list(container.read_part(p))
        sj = Path(p + ".json")
        if sj.exists():
            merge_stats(stats, json.loads(sj.read_text()))
    info = container.assemble(entries, args.out, bbox_xy, content_version(args),
                              BIOME_MAX_V1, POI_MAX_V1)
    log(f"pack escrito: {info}")
    write_report(args.report, bbox, info, stats, args.osm)


def cmd_build(args):
    bbox = parse_bbox(args)
    _, bbox_xy = grid.gh3_ids_for_bbox(*bbox)
    entries, stats = run_shard(bbox, args.osm, 0, 1)
    info = container.assemble(entries, args.out, bbox_xy, content_version(args),
                              BIOME_MAX_V1, POI_MAX_V1)
    log(f"pack escrito: {info}")
    write_report(args.report, bbox, info, stats, args.osm)


def cmd_query(args):
    p = container.Pack(args.pack)
    x, y = grid.xy_from_gps(args.lon, args.lat)
    print(json.dumps({"x": x, "y": y, "gh3": grid.gh3_id(x, y), "sub": grid.sub_idx(x, y),
                      "local": grid.local_idx(x, y), **p.at(x, y)}))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="worldpackgen")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def region_args(s):
        s.add_argument("--region", choices=sorted(REGIONS))
        s.add_argument("--bbox", help="lon0,lat0,lon1,lat1")

    s = sub.add_parser("region")
    s.add_argument("region", choices=sorted(REGIONS))
    s.add_argument("field", choices=list(REGION_FIELDS))
    s.set_defaults(fn=cmd_region)

    s = sub.add_parser("osm")
    region_args(s)
    s.add_argument("--pbf", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_osm)

    s = sub.add_parser("shard")
    region_args(s)
    s.add_argument("--osm")
    s.add_argument("--shard", type=int, required=True)
    s.add_argument("--shards", type=int, required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--allow-empty", action="store_true")
    s.set_defaults(fn=cmd_shard)

    s = sub.add_parser("probe")
    s.add_argument("lon", type=float)
    s.add_argument("lat", type=float)
    s.set_defaults(fn=cmd_probe)

    s = sub.add_parser("filter-expr")
    s.set_defaults(fn=cmd_filter_expr)

    for name, fn in (("assemble", cmd_assemble), ("build", cmd_build)):
        s = sub.add_parser(name)
        region_args(s)
        s.add_argument("--osm")
        s.add_argument("--out", required=True)
        s.add_argument("--report", default="build_report.json")
        s.add_argument("--content-version")
        if name == "assemble":
            s.add_argument("parts", nargs="+")
        s.set_defaults(fn=fn)

    s = sub.add_parser("query")
    s.add_argument("pack")
    s.add_argument("lon", type=float)
    s.add_argument("lat", type=float)
    s.set_defaults(fn=cmd_query)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
