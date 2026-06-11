"""Sweep G-PCC (tmc3) encoder configs on the xyz stream of a 3DGS ply.

For each config: voxelize -> tmc3 encode -> tmc3 decode -> verify exact voxel match
(after Morton sort) = lossless. Reports compressed size and encode/decode times.

Run:  python tools/gpcc_sweep.py --ply_path <point_cloud.ply> [--bit_depths 16 15 14]
"""
import argparse
import os
import sys
import time
from tempfile import TemporaryDirectory

import numpy as np
from plyfile import PlyData

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model.gpcc_utils import (GPCC_CODEC_PATH, GPCC_ENC_FLAGS_DEFAULT, gpcc_encode, gpcc_decode,
                              write_ply_geo_ascii, read_ply_geo_bin, sorted_voxels)

BASE = GPCC_ENC_FLAGS_DEFAULT

CONFIGS = {
    'baseline (octree)': BASE,
    'planar': BASE.replace('--planarEnabled=0', '--planarEnabled=1'),
    'planar+idcm': BASE.replace('--planarEnabled=0', '--planarEnabled=1')
                       .replace('--planarModeIdcmUse=0', '--planarModeIdcmUse=1'),
    'cabac (no bypass)': BASE.replace('--cabac_bypass_stream_enabled_flag=1',
                                      '--cabac_bypass_stream_enabled_flag=0'),
    'planar+cabac': BASE.replace('--planarEnabled=0', '--planarEnabled=1')
                        .replace('--cabac_bypass_stream_enabled_flag=1',
                                 '--cabac_bypass_stream_enabled_flag=0'),
    'intra6': BASE.replace('--intra_pred_max_node_size_log2=3', '--intra_pred_max_node_size_log2=6'),
    'planar+cabac+intra6': BASE.replace('--planarEnabled=0', '--planarEnabled=1')
                               .replace('--cabac_bypass_stream_enabled_flag=1',
                                        '--cabac_bypass_stream_enabled_flag=0')
                               .replace('--intra_pred_max_node_size_log2=3',
                                        '--intra_pred_max_node_size_log2=6'),
    'predgeom': ('--trisoupNodeSizeLog2=0 --mergeDuplicatedPoints=0 --positionQuantizationScale=1 '
                 '--geomTreeType=1 --cabac_bypass_stream_enabled_flag=1'),
    'predgeom+cabac': ('--trisoupNodeSizeLog2=0 --mergeDuplicatedPoints=0 --positionQuantizationScale=1 '
                       '--geomTreeType=1 --cabac_bypass_stream_enabled_flag=0'),
}


def load_means(ply_path):
    ply = PlyData.read(ply_path).elements[0]
    return np.stack([ply.data[n] for n in ['x', 'y', 'z']], axis=1).astype(np.float32)


def voxelize_bits(means, bits):
    means_min, means_max = means.min(axis=0), means.max(axis=0)
    v = (means - means_min) / (means_max - means_min)
    return np.round(v * (2 ** bits - 1))


def run_config(name, flags, vox_sorted, tmc3):
    with TemporaryDirectory() as td:
        ply_in = os.path.join(td, 'in.ply')
        bin_path = os.path.join(td, 'out.bin')
        ply_out = os.path.join(td, 'rec.ply')
        write_ply_geo_ascii(vox_sorted, ply_in)
        try:
            t0 = time.time()
            gpcc_encode(encoder_path=tmc3, ply_path=ply_in, bin_path=bin_path, flags=flags)
            t_enc = time.time() - t0
            size = os.path.getsize(bin_path)
            t0 = time.time()
            gpcc_decode(decoder_path=tmc3, bin_path=bin_path, recon_path=ply_out)
            t_dec = time.time() - t0
        except AssertionError as e:
            print(f'{name:24s} CODEC FAILED: {e}')
            return None
        rec = sorted_voxels(read_ply_geo_bin(ply_out).astype(np.float32))
        lossless = (rec.shape == vox_sorted.shape) and np.array_equal(rec, vox_sorted.astype(np.float32))
    return size, t_enc, t_dec, lossless


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ply_path', required=True)
    ap.add_argument('--tmc3', default=GPCC_CODEC_PATH)
    ap.add_argument('--bit_depths', nargs='+', type=int, default=[16])
    args = ap.parse_args()

    means = load_means(args.ply_path)
    print(f'{args.ply_path}: {means.shape[0]} points')
    for bits in args.bit_depths:
        vox = sorted_voxels(voxelize_bits(means, bits))
        print(f'\n--- bit depth {bits} ---')
        print(f'{"config":24s} {"bytes":>12s} {"enc_s":>7s} {"dec_s":>7s}  lossless  vs_baseline')
        base_size = None
        for name, flags in CONFIGS.items():
            res = run_config(name, flags, vox, args.tmc3)
            if res is None:
                continue
            size, t_enc, t_dec, lossless = res
            if base_size is None:
                base_size = size
            print(f'{name:24s} {size:12d} {t_enc:7.2f} {t_dec:7.2f}  {str(lossless):8s}  '
                  f'{100.0 * (size - base_size) / base_size:+6.2f}%')


if __name__ == '__main__':
    main()
