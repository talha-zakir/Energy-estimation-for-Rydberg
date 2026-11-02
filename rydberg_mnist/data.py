"""Original MNIST IDX data with official train/test separation and MD5 checks."""
import gzip
import hashlib
from pathlib import Path
import struct
from urllib.request import urlopen

import numpy as np
from sklearn.model_selection import train_test_split

BASE = "https://storage.googleapis.com/cvdf-datasets/mnist/"
FILES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}


def load_mnist(cache="data/mnist"):
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    arrays = []
    for name, checksum in FILES.items():
        path = cache / name
        if not path.exists():
            print(f"Downloading {name}", flush=True)
            payload = urlopen(BASE + name, timeout=60).read()
            if hashlib.md5(payload).hexdigest() != checksum:
                raise ValueError(f"Download checksum mismatch: {name}")
            path.write_bytes(payload)
        if hashlib.md5(path.read_bytes()).hexdigest() != checksum:
            raise ValueError(f"Cached checksum mismatch: {path}")
        raw = gzip.decompress(path.read_bytes())
        magic, count = struct.unpack(">II", raw[:8])
        if magic == 2051:
            rows, cols = struct.unpack(">II", raw[8:16])
            array = np.frombuffer(raw, np.uint8, offset=16).reshape(count, rows * cols)
        elif magic == 2049:
            array = np.frombuffer(raw, np.uint8, offset=8).reshape(count)
        else:
            raise ValueError(f"Invalid IDX magic {magic}")
        arrays.append(array)
    return tuple(arrays)


def stratified_subset(indices, labels, count, seed):
    if not 0 < count <= len(indices):
        raise ValueError(f"Requested {count} samples from {len(indices)} available")
    if count == len(indices):
        return indices.copy()
    return train_test_split(indices, train_size=count, stratify=labels[indices],
                            random_state=seed)[0]


def split_indices(train_labels, test_labels, digits, n_train, n_val, n_test, seed):
    eligible = np.flatnonzero(np.isin(train_labels, digits))
    pool = stratified_subset(eligible, train_labels, n_train + n_val, seed)
    train, val = train_test_split(pool, test_size=n_val, stratify=train_labels[pool],
                                 random_state=seed + 1)
    test = stratified_subset(np.flatnonzero(np.isin(test_labels, digits)),
                             test_labels, n_test, seed + 2)
    return train, val, test
