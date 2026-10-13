"""Parallel, resumable downloader + in-order streaming extractor for the Nordland
tarballs hosted at https://huggingface.co/datasets/Somayeh-h/Nordland .

The network link to HF is slow per-connection (~0.5 MB/s), so each tarball is
fetched as fixed-size HTTP range chunks by a thread pool (lowest chunk index
first, interleaved across seasons). A per-season extractor thread feeds the
contiguous prefix of finished chunks into `tar -xz`, so images from the start
of each traverse (all that the E1 experiments need) are available early.

Usage: python scripts/fetch_nordland.py summer spring fall
Output: data/nordland/<season>/images-XXXXX.png ; chunks in data/raw/
The dataset is not redistributed; this only mirrors it locally.
"""
import os
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = "https://huggingface.co/datasets/Somayeh-h/Nordland/resolve/main/data/{}.tar.gz"
CHUNK = 32 * 1024 * 1024
RAW = "data/raw"
OUT = "data/nordland"
N_WORKERS = int(os.environ.get("FETCH_WORKERS", "24"))


def content_length(url):
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req, timeout=60) as r:
        return int(r.headers["Content-Length"])


def chunk_path(season, k):
    return os.path.join(RAW, f"{season}.part{k:05d}")


def fetch_chunk(season, url, k, size):
    path = chunk_path(season, k)
    lo, hi = k * CHUNK, min(size, (k + 1) * CHUNK) - 1
    want = hi - lo + 1
    if os.path.exists(path) and os.path.getsize(path) == want:
        return
    for attempt in range(20):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={lo}-{hi}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if len(data) != want:
                raise IOError(f"short read {len(data)} != {want}")
            with open(path + ".tmp", "wb") as f:
                f.write(data)
            os.replace(path + ".tmp", path)
            return
        except Exception as e:  # retry with backoff
            print(f"[{season} {k}] attempt {attempt}: {e}", flush=True)
            time.sleep(min(60, 2 ** attempt))
    raise RuntimeError(f"failed chunk {season} {k}")


def extractor(season, n_chunks):
    done_flag = os.path.join(OUT, f".{season}.extracted")
    if os.path.exists(done_flag):
        return
    os.makedirs(OUT, exist_ok=True)
    p = subprocess.Popen(["tar", "-xzf", "-", "-C", OUT], stdin=subprocess.PIPE)
    for k in range(n_chunks):
        path = chunk_path(season, k)
        while not os.path.exists(path):
            time.sleep(2)
        with open(path, "rb") as f:
            p.stdin.write(f.read())
        p.stdin.flush()
        if k % 20 == 0:
            print(f"[extract {season}] fed chunk {k}/{n_chunks}", flush=True)
    p.stdin.close()
    rc = p.wait()
    print(f"[extract {season}] tar exit {rc}", flush=True)
    if rc == 0:
        open(done_flag, "w").close()


def main(seasons):
    os.makedirs(RAW, exist_ok=True)
    jobs, ext = [], []
    for s in seasons:
        url = URL.format(s)
        size = content_length(url)
        n = (size + CHUNK - 1) // CHUNK
        print(f"{s}: {size} bytes, {n} chunks", flush=True)
        jobs += [(k, s, url, size) for k in range(n)]
        t = threading.Thread(target=extractor, args=(s, n), daemon=False)
        t.start()
        ext.append(t)
    jobs.sort()  # interleave seasons, earliest chunks first
    with ThreadPoolExecutor(N_WORKERS) as ex:
        list(ex.map(lambda j: fetch_chunk(j[1], j[2], j[0], j[3]), jobs))
    print("all chunks downloaded", flush=True)
    for t in ext:
        t.join()
    print("done", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
