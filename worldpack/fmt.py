"""Codificacion y decodificacion de blobs de chunk (capitulo 3 del spec)."""
import struct
import numpy as np

LAYER_NAMES = ("biome", "elev", "water", "built", "paths", "flags")
LAYER_BITS = (5, 6, 2, 3, 2, 2)
M_CONST, M_PAL1, M_PAL2, M_PAL4, M_RLE, M_RAW = range(6)
MODE_NAMES = ("CONST", "PAL1", "PAL2", "PAL4", "RLE", "RAW")
PAL_SIZE = {M_PAL1: (2, 1), M_PAL2: (4, 2), M_PAL4: (16, 4)}
ELEV_STEPS = (1, 2, 4, 8, 16, 32, 64, 128)
POI_MAX = 8
CF_COAST, CF_PROTECTED, CF_NAVWATER = 1, 2, 4


# ---------- bits: LSB primero, little-endian ----------
def pack_bits(values, bits):
    v = np.asarray(values, dtype=np.uint32)
    bitarr = ((v[:, None] >> np.arange(bits, dtype=np.uint32)) & 1).astype(np.uint8).ravel()
    return np.packbits(bitarr, bitorder="little")[: (len(v) * bits + 7) // 8].tobytes()


def unpack_bits(buf, off, count, bits):
    nbytes = (count * bits + 7) // 8
    if off + nbytes > len(buf):
        raise ValueError("blob truncado")
    raw = np.frombuffer(buf, dtype=np.uint8, count=nbytes, offset=off)
    bitarr = np.unpackbits(raw, bitorder="little")[: count * bits].reshape(count, bits)
    return (bitarr.astype(np.uint32) << np.arange(bits, dtype=np.uint32)).sum(axis=1)


def varint(n):
    out = bytearray()
    while True:
        b, n = n & 0x7F, n >> 7
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def read_varint(buf, off):
    n = shift = 0
    while True:
        b = buf[off]
        off += 1
        n |= (b & 0x7F) << shift
        shift += 7
        if not b & 0x80:
            return n, off


# ---------- capas ----------
def encode_layer(values, bits, stats=None):
    """values: 1024 enteros en orden Z. Prueba los seis modos y devuelve el menor."""
    v = np.asarray(values, dtype=np.uint32)
    if len(v) != 1024 or int(v.max()) >= (1 << bits):
        raise ValueError("valores fuera de rango para la capa")
    uniq = np.unique(v)
    cands = []
    if len(uniq) == 1:
        cands.append(bytes([M_CONST, int(uniq[0])]))
    for mode, (psize, pbits) in PAL_SIZE.items():
        if len(uniq) <= psize and pbits < bits:
            pal = np.concatenate((uniq, np.full(psize - len(uniq), uniq[0], np.uint32)))
            idx = np.searchsorted(uniq, v).astype(np.uint32)
            cands.append(bytes([mode]) + pal.astype(np.uint8).tobytes() + pack_bits(idx, pbits))
    starts = np.concatenate(([0], np.flatnonzero(np.diff(v)) + 1))
    if len(starts) <= 400:  # por encima nunca gana a RAW
        lens = np.diff(np.concatenate((starts, [1024])))
        body = bytearray(struct.pack("<BH", M_RLE, len(starts)))
        for s, ln in zip(starts, lens):
            body.append(int(v[s]))
            body += varint(int(ln))
        cands.append(bytes(body))
    cands.append(bytes([M_RAW]) + pack_bits(v, bits))
    best = min(cands, key=len)  # en empate gana el de codigo menor
    if stats is not None:
        stats[best[0]] = stats.get(best[0], 0) + 1
    return best


def decode_layer(buf, off, bits):
    """Devuelve (valores[1024], nuevo_offset)."""
    mode = buf[off]
    off += 1
    if mode == M_CONST:
        return np.full(1024, buf[off], dtype=np.uint32), off + 1
    if mode in PAL_SIZE:
        psize, pbits = PAL_SIZE[mode]
        pal = np.frombuffer(buf, dtype=np.uint8, count=psize, offset=off).astype(np.uint32)
        off += psize
        return pal[unpack_bits(buf, off, 1024, pbits)], off + (1024 * pbits) // 8
    if mode == M_RLE:
        (n,) = struct.unpack_from("<H", buf, off)
        off += 2
        out = np.empty(1024, dtype=np.uint32)
        pos = 0
        for _ in range(n):
            val = buf[off]
            ln, off = read_varint(buf, off + 1)
            if pos + ln > 1024:
                raise ValueError("RLE corrupto")
            out[pos:pos + ln] = val
            pos += ln
        if pos != 1024:
            raise ValueError("RLE corrupto")
        return out, off
    if mode == M_RAW:
        return unpack_bits(buf, off, 1024, bits), off + (1024 * bits + 7) // 8
    raise ValueError(f"modo de capa desconocido {mode}")


# ---------- altitud ----------
def quantize_elevation(elev):
    """Devuelve (elev_packed, offsets). Altitudes negativas se recortan a 0."""
    e = np.clip(np.asarray(elev, dtype=np.int32), 0, 4095 + 63 * 128)
    base = int(min(e.min(), 4095))
    rng = int(e.max()) - base
    step_i = next((i for i, s in enumerate(ELEV_STEPS) if rng <= 63 * s), 7)
    off = np.clip(np.rint((e - base) / ELEV_STEPS[step_i]), 0, 63).astype(np.uint32)
    return base | (step_i << 12), off


def elev_unpack(packed):
    return packed & 0xFFF, ELEV_STEPS[(packed >> 12) & 7]


# ---------- blob ----------
def encode_blob(layers, pois, stats=None):
    """layers: dict LAYER_NAMES -> 1024 valores en orden Z ('elev' en metros).
    pois: lista de (local_idx, tipo) ya ordenada por prioridad."""
    packed, eoff = quantize_elevation(layers["elev"])
    water = np.asarray(layers["water"])
    flags = 0
    if (water == 3).any() and (water != 3).any():
        flags |= CF_COAST
    if (np.asarray(layers["flags"]) & 1).any():
        flags |= CF_PROTECTED
    if (water == 2).any():
        flags |= CF_NAVWATER
    pois = pois[:POI_MAX]
    out = bytearray(struct.pack("<BHB", flags, packed, len(pois)))
    for i, (name, bits) in enumerate(zip(LAYER_NAMES, LAYER_BITS)):
        arr = eoff if name == "elev" else layers[name]
        out += encode_layer(arr, bits, None if stats is None else stats.setdefault(name, {}))
    for idx, typ in pois:
        out += struct.pack("<H", (idx & 0x3FF) | (typ << 10))
    return bytes(out)


def decode_blob(buf):
    buf = bytes(buf)
    flags, packed, npoi = struct.unpack_from("<BHB", buf, 0)
    off = 4
    vals = []
    for bits in LAYER_BITS:
        v, off = decode_layer(buf, off, bits)
        vals.append(v)
    base, step = elev_unpack(packed)
    pois = []
    for _ in range(npoi):
        (p,) = struct.unpack_from("<H", buf, off)
        off += 2
        pois.append((p & 0x3FF, p >> 10))
    if off != len(buf):
        raise ValueError("longitud de blob inconsistente")
    poi_arr = np.zeros(1024, dtype=np.uint32)
    for idx, typ in reversed(pois):  # el primero de la lista gana
        poi_arr[idx] = typ
    return {
        "chunk_flags": flags,
        "biome": vals[0], "elev": base + vals[1].astype(np.int32) * step,
        "water": vals[2], "built": vals[3], "paths": vals[4], "flags": vals[5],
        "poi": poi_arr, "pois": pois,
    }
