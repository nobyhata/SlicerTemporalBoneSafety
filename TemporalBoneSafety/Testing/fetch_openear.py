"""
Download only the files TemporalBoneSafety needs from one OpenEar specimen
(Zenodo record 1473724, CC BY 4.0) using HTTP range requests, instead of the
whole 4-14 GB zip.

  python fetch_openear.py ZETA  ./openear_ZETA

Specimens: ALPHA BETA GAMMA DELTA EPSILON ZETA ETA THETA
Files (~290 MB per specimen):
  05_Registred_Slicer_Volumes/CBCT_Unembedded.nrrd   registered CBCT
  06_Segmentation/Segmentation.seg.nrrd              13 segments incl. bone
  07_3D_Models/Bone.nrrd                             bone label

Data: Sieber D, et al. The OpenEar library of 3D models of the human temporal
bone based on computed tomography and micro-slicing. Sci Data 6, 180297 (2019).
"""
import io
import json
import os
import struct
import sys
import urllib.request
import zipfile
import zlib

RECORD = "https://zenodo.org/api/records/1473724"
WANT = ["05_Registred_Slicer_Volumes/CBCT_Unembedded.nrrd",
        "06_Segmentation/Segmentation.seg.nrrd",
        "07_3D_Models/Bone.nrrd"]


def _get(url, start=None, end=None):
    headers = {"User-Agent": "TemporalBoneSafety-fetch"}
    if start is not None:
        headers["Range"] = "bytes=%d-%d" % (start, end)
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=600)


class _RemoteFile(io.RawIOBase):
    def __init__(self, url):
        r = urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60)
        self.url, self.size, self.pos = r.geturl(), int(r.headers["Content-Length"]), 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def readinto(self, b):
        if self.pos >= self.size:
            return 0
        end = min(self.pos + len(b), self.size) - 1
        d = _get(self.url, self.pos, end).read()
        b[:len(d)] = d
        self.pos += len(d)
        return len(d)


def main(specimen, outdir):
    specimen = specimen.upper()
    files = {f["key"]: f["links"]["self"] for f in json.load(_get(RECORD))["files"]}
    url = files.get(specimen + ".zip")
    if not url:
        sys.exit("Unknown specimen %s; choose from %s" % (specimen, sorted(k[:-4] for k in files)))
    rf = _RemoteFile(url)
    z = zipfile.ZipFile(io.BufferedReader(rf, buffer_size=1 << 16))
    os.makedirs(outdir, exist_ok=True)
    for info in z.infolist():
        if info.filename not in WANT:
            continue
        out = os.path.join(outdir, os.path.basename(info.filename))
        if os.path.exists(out) and os.path.getsize(out) == info.file_size:
            print("exists", out)
            continue
        start = info.header_offset
        head = _get(rf.url, start, start + 29).read()
        nl, xl = struct.unpack("<HH", head[26:30])
        dstart = start + 30 + nl + xl
        print("downloading %s (%.0f MB)..." % (info.filename, info.compress_size / 1e6))
        raw = _get(rf.url, dstart, dstart + info.compress_size - 1).read()
        data = raw if info.compress_type == 0 else zlib.decompress(raw, -15)
        if len(data) != info.file_size or (zlib.crc32(data) & 0xFFFFFFFF) != info.CRC:
            sys.exit("Checksum mismatch for " + info.filename)
        with open(out, "wb") as f:
            f.write(data)
        print("ok", out)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
