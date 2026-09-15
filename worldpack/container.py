"""Layout del archivo .pack: header, directorio raiz, sub-indices y blobs."""
import hashlib
import struct
import zlib
from pathlib import Path

import numpy as np

from . import FORMAT_VERSION
from . import grid, fmt

MAGIC = b"TRWP"
HEADER_FMT = "<4sHHIIIIIBBBB4II76s"
HEADER_SIZE = 128
assert struct.calcsize(HEADER_FMT) == HEADER_SIZE
SUBINDEX_SIZE = 1024 * 6
PART_MAGIC = b"WPPT"


# ---------- archivos parciales (salida de cada shard) ----------
def write_part(path, entries):
    """entries: iterable de (gh3, sub_idx, blob)."""
    with open(path, "wb") as f:
        f.write(PART_MAGIC)
        for gid, sidx, blob in entries:
            f.write(struct.pack("<HHH", gid, sidx, len(blob)))
            f.write(blob)


def read_part(path):
    data = Path(path).read_bytes()
    if data[:4] != PART_MAGIC:
        raise ValueError(f"{path}: no es un archivo parcial")
    off = 4
    while off < len(data):
        gid, sidx, ln = struct.unpack_from("<HHH", data, off)
        off += 6
        yield gid, sidx, data[off:off + ln]
        off += ln


# ---------- ensamblado ----------
def assemble(entries, out_path, bbox_xy, content_version, biome_max, poi_max):
    """entries: iterable de (gh3, sub_idx, blob). Deduplica por contenido."""
    by_gh3 = {}
    for gid, sidx, blob in entries:
        by_gh3.setdefault(gid, {})[sidx] = blob
    gh3s = sorted(g for g, d in by_gh3.items() if d)

    root_off = HEADER_SIZE
    sub_base = root_off + 8 * len(gh3s)
    data_off = sub_base + SUBINDEX_SIZE * len(gh3s)

    blob_offsets = {}
    data = bytearray()
    root = bytearray()
    subs = bytearray()
    dedup_hits = total = 0
    for i, gid in enumerate(gh3s):
        root += struct.pack("<HHI", gid, 0, sub_base + i * SUBINDEX_SIZE)
        table = bytearray(SUBINDEX_SIZE)
        for sidx, blob in by_gh3[gid].items():
            total += 1
            h = hashlib.blake2b(blob, digest_size=16).digest()
            if h in blob_offsets:
                dedup_hits += 1
            else:
                blob_offsets[h] = len(data)
                data += blob
            struct.pack_into("<IH", table, sidx * 6, blob_offsets[h], len(blob))
        subs += table
    if len(data) >= 1 << 32:
        raise ValueError("zona de datos > 4 GB")

    body = bytes(root) + bytes(subs) + bytes(data)
    header = bytearray(struct.pack(
        HEADER_FMT, MAGIC, FORMAT_VERSION, 0, content_version,
        len(gh3s), root_off, len(blob_offsets), data_off,
        len(fmt.LAYER_BITS), biome_max, poi_max, fmt.POI_MAX,
        *bbox_xy, 0, b"\0" * 76))
    crc = zlib.crc32(bytes(header[0x40:]) + body) & 0xFFFFFFFF
    struct.pack_into("<I", header, 0x30, crc)
    tmp = Path(str(out_path) + ".tmp")
    tmp.write_bytes(bytes(header) + body)
    tmp.replace(out_path)
    return {"gh3": len(gh3s), "chunks": total, "unique_blobs": len(blob_offsets),
            "dedup_hits": dedup_hits, "bytes": HEADER_SIZE + len(body)}


# ---------- lector de referencia ----------
class PackError(Exception):
    pass


class Pack:
    def __init__(self, path, verify_crc=True):
        self.data = Path(path).read_bytes()
        if len(self.data) < HEADER_SIZE:
            raise PackError("archivo truncado")
        h = struct.unpack_from(HEADER_FMT, self.data, 0)
        (magic, self.format_version, self.flags, self.content_version,
         self.root_count, self.root_offset, self.chunk_count, self.chunk_data_offset,
         self.layer_count, self.biome_max, self.poi_max, self.poi_per_chunk_max) = h[:12]
        self.bbox = h[12:16]
        crc = h[16]
        if magic != MAGIC:
            raise PackError("magic incorrecto")
        if self.format_version != FORMAT_VERSION:
            raise PackError(f"format_version {self.format_version} no soportada")
        if self.flags & 1:
            raise PackError("capa LZ4 no soportada por este lector")
        if self.chunk_data_offset > len(self.data):
            raise PackError("archivo truncado")
        if verify_crc and zlib.crc32(self.data[0x40:]) & 0xFFFFFFFF != crc:
            raise PackError("CRC incorrecto")
        self.root = {}
        for i in range(self.root_count):
            gid, _, off = struct.unpack_from("<HHI", self.data, self.root_offset + 8 * i)
            self.root[gid] = off
        self._cache = {}
        self.sd_reads = 0

    def _chunk(self, x, y):
        key = (x >> 5, y >> 5)
        if key in self._cache:
            return self._cache[key]
        gid = grid.gh3_id(x, y)
        res = None
        if gid in self.root:
            self.sd_reads += 1
            off, ln = struct.unpack_from("<IH", self.data, self.root[gid] + grid.sub_idx(x, y) * 6)
            if ln:
                self.sd_reads += 1
                start = self.chunk_data_offset + off
                if start + ln > len(self.data):
                    raise PackError("blob fuera del archivo")
                res = fmt.decode_blob(self.data[start:start + ln])
        self._cache[key] = res
        return res

    def at(self, x, y):
        c = self._chunk(x, y)
        if c is None:  # ausente = mar
            return {"biome": 1, "water": 3, "built": 0, "paths": 0, "flags": 0,
                    "poi": 0, "elevation": 0}
        i = grid.local_idx(x, y)
        return {"biome": int(c["biome"][i]), "water": int(c["water"][i]),
                "built": int(c["built"][i]), "paths": int(c["paths"][i]),
                "flags": int(c["flags"][i]), "poi": int(c["poi"][i]),
                "elevation": int(c["elev"][i])}

    def grid(self, cx, cy, n):
        h = n // 2
        return [self.at(cx + dx, cy + dy)
                for dy in range(-h, n - h) for dx in range(-h, n - h)]

    def at_gps(self, lon, lat):
        return self.at(*grid.xy_from_gps(lon, lat))
