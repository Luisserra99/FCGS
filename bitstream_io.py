"""Shared helpers for the encode / decode entry points.

The seed used to shuffle the gaussians before they are split into chunks
(FCGS.compress) has to be reproduced exactly by the decoder, otherwise the
bitstream decodes to garbage without raising. encode_single_scene.py records it
in encode_meta.json next to the run; the decoders read it back from there.
"""

import json
import os


def dir_size_bytes(path):
    """Total bytes actually written to disk under `path` (recursive)."""
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            total += os.path.getsize(os.path.join(root, name))
    return total


def count_steps(lmd_dir):
    """Number of independent compression steps stored in `lmd_dir`.

    Only numerically named subdirectories count, so stray files (a stale log, a
    metadata dump) cannot inflate the step count.
    """
    assert os.path.isdir(lmd_dir), f"Bitstream directory {lmd_dir} not found."
    steps = [d for d in os.listdir(lmd_dir)
             if d.isdigit() and os.path.isdir(os.path.join(lmd_dir, d))]
    assert steps, f"No compression steps found under {lmd_dir}."
    return len(steps)


def load_encode_meta(out_dir):
    """encode_meta.json written by the encoder, or None if it is not there."""
    meta_path = os.path.join(out_dir, "encode_meta.json")
    if not os.path.exists(meta_path):
        return None
    with open(meta_path) as f:
        return json.load(f)


def resolve_seed(cli_seed, meta):
    """The seed to hand to FCGS.decomprss: --seed wins, else encode_meta.json."""
    if cli_seed is not None:
        return int(cli_seed)
    if meta is not None and "seed" in meta:
        return int(meta["seed"])
    raise SystemExit(
        "Cannot determine the chunk-shuffle seed: pass --seed explicitly, or "
        "point --out_dir at the directory holding the encode_meta.json this "
        "bitstream was written with. Decoding with the wrong seed silently "
        "produces a corrupted model."
    )
