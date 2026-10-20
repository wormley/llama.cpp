#!/usr/bin/env python3
# gguf-rename-arch.py — rewrite a metadata-only GGUF shard, renaming an
# architecture string in general.architecture AND every "<old>." KV key prefix.
# Only metadata is touched; no tensor bytes are involved (refuses shards that
# carry tensor data). Exact-preserving: quant data is never read or modified.
#
# Usage: gguf-rename-arch.py <in.gguf> <out.gguf> <old-arch> <new-arch>
import struct, sys

def parse(path):
    f = open(path, "rb")
    def rd(n):
        b = f.read(n)
        assert len(b) == n, "EOF"
        return b
    def u32(): return struct.unpack("<I", rd(4))[0]
    def u64(): return struct.unpack("<Q", rd(8))[0]
    def s():
        n = u64(); return rd(n).decode("utf-8")
    def val(t):
        if t == 0: return struct.unpack("<B", rd(1))[0]
        if t == 1: return struct.unpack("<b", rd(1))[0]
        if t == 2: return struct.unpack("<H", rd(2))[0]
        if t == 3: return struct.unpack("<h", rd(2))[0]
        if t == 4: return u32()
        if t == 5: return struct.unpack("<i", rd(4))[0]
        if t == 6: return struct.unpack("<f", rd(4))[0]
        if t == 7: return bool(rd(1)[0])
        if t == 8: return s()
        if t == 9:
            et = u32(); n = u64()
            if et == 8:
                return (8, [s() for _ in range(n)])
            sz = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}[et]
            return (et, rd(n * sz))
        if t == 10: return u64()
        if t == 11: return struct.unpack("<q", rd(8))[0]
        if t == 12: return struct.unpack("<d", rd(8))[0]
        raise ValueError(t)
    assert rd(4) == b"GGUF", "not gguf"
    ver = u32(); nt = u64(); nk = u64()
    assert nt == 0, "shard carries tensor data; refusing (metadata-only patcher)"
    kvs = []
    for _ in range(nk):
        k = s(); t = u32(); v = val(t)
        kvs.append([k, t, v])
    return ver, kvs

def enc_str(x):
    b = x.encode("utf-8")
    return struct.pack("<Q", len(b)) + b

def ser(ver, kvs):
    parts = [b"GGUF", struct.pack("<IQQ", ver, 0, len(kvs))]
    for k, t, v in kvs:
        parts.append(enc_str(k)); parts.append(struct.pack("<I", t))
        if t == 0: parts.append(struct.pack("<B", v))
        elif t == 1: parts.append(struct.pack("<b", v))
        elif t == 2: parts.append(struct.pack("<H", v))
        elif t == 3: parts.append(struct.pack("<h", v))
        elif t == 4: parts.append(struct.pack("<I", v))
        elif t == 5: parts.append(struct.pack("<i", v))
        elif t == 6: parts.append(struct.pack("<f", v))
        elif t == 7: parts.append(struct.pack("<B", 1 if v else 0))
        elif t == 8: parts.append(enc_str(v))
        elif t == 9:
            et, payload = v
            if et == 8:
                parts.append(struct.pack("<IQ", 8, len(payload)))
                for s in payload: parts.append(enc_str(s))
            else:
                sz = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:1,10:8,11:8,12:8}[et]
                parts.append(struct.pack("<IQ", et, len(payload) // sz))
                parts.append(payload)
        elif t == 10: parts.append(struct.pack("<Q", v))
        elif t == 11: parts.append(struct.pack("<q", v))
        elif t == 12: parts.append(struct.pack("<d", v))
        else: raise ValueError(t)
    return b"".join(parts)

def main():
    inp, outp, old, new = sys.argv[1:5]
    ver, kvs = parse(inp)
    renamed = 0
    for ent in kvs:
        k, t, v = ent
        if k == "general.architecture" and t == 8 and v == old:
            ent[2] = new; renamed += 1
        elif k.startswith(old + "."):
            ent[0] = new + "." + k[len(old)+1:]; renamed += 1
    assert renamed > 0, "nothing renamed — arch string not found"
    data = ser(ver, kvs)
    with open(outp, "wb") as f:
        f.write(data)
    print("renamed %d strings; wrote %d bytes -> %s" % (renamed, len(data), outp))
    ver2, kvs2 = parse(outp)
    arch2 = [v for k, t, v in kvs2 if k == "general.architecture"][0]
    assert arch2 == new, arch2
    assert not any(k.startswith(old + ".") for k, t, v in kvs2)
    print("verify OK: general.architecture=%s, %d KV pairs, %d bytes" %
          (arch2, len(kvs2), len(data)))

main()
