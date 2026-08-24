import gc
import json
import os
import random
import sys
import time
from argparse import ArgumentParser

import numpy as np
import torch

from bitstream_io import dir_size_bytes
from gaussian_renderer import GaussianModel
from model.FCGS_model import FCGS


def train(args):
    # Seed every RNG so repeated runs with different --seed values give the
    # independent samples needed for variance / significance analysis. The
    # chunk shuffle inside FCGS.compress re-seeds torch itself with the same
    # value; this covers everything else.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    timings = {}

    t0 = time.time()
    with torch.no_grad():
        gaussians = GaussianModel(3)  # dataset.sh_degree = 3
        gaussians.load_ply(path=args.ply_path_from)
    g_xyz = gaussians._xyz.detach()
    N_gaussian = g_xyz.shape[0]

    per_step_size = int(args.per_step_size)
    # Avoid a tiny trailing step: fold it into the previous one instead.
    if N_gaussian > per_step_size and N_gaussian < per_step_size * 1.1:
        per_step_size = int(per_step_size * 1.1)

    _features_dc = gaussians._features_dc.detach().view(N_gaussian, -1)  # [N, 1, 3] -> [N, 3]
    _features_rest = gaussians._features_rest.detach().view(N_gaussian, -1)  # [N, 15, 3] -> [N, 45]
    _opacity = gaussians._opacity.detach()  # [N, 1]
    _scaling = gaussians._scaling.detach()  # [N, 3]
    _rotation = gaussians._rotation.detach()  # [N, 4]
    g_fea = torch.cat([_features_dc, _features_rest, _opacity, _scaling, _rotation], dim=-1)  # [N, 56]

    # g_xyz/g_fea now hold everything the codec needs; the model itself is a
    # second full copy on the GPU. Dropping it here is what keeps the largest
    # scenes (bicycle, 6.1M gaussians) inside 24 GB.
    del _features_dc, _features_rest, _opacity, _scaling, _rotation, gaussians
    gc.collect()
    torch.cuda.empty_cache()
    timings["time_load_ply"] = time.time() - t0

    step_num = int(np.ceil(N_gaussian / per_step_size))
    lmd = args.lmd
    chunk_size_list = [200_0000, 100_0000, 100_0000]

    t0 = time.time()
    CM = FCGS(
        Q=1,
        resolutions_list=[300, 400, 500],
        resolutions_list_3D=[70, 80, 90],
        norm_radius=args.nr,
    ).cuda()
    CM.load_state_dict(torch.load(f'./checkpoints/checkpoint_{lmd}.pkl'), strict=True)
    timings["time_model_load"] = time.time() - t0

    ttl_size = 0
    CM.eval()
    torch.cuda.synchronize(); t1 = time.time()
    with torch.no_grad():
        for s in range(step_num):
            bit_save_path = os.path.join(args.bit_path_to, f"{lmd}/{s}")
            os.makedirs(bit_save_path, exist_ok=True)
            g_xyz_in = g_xyz[s*per_step_size:s*per_step_size+per_step_size]
            g_fea_in = g_fea[s*per_step_size:s*per_step_size+per_step_size]
            ttl_size += CM.compress(g_xyz_in, g_fea_in, root_path=bit_save_path, chunk_size_list=chunk_size_list, determ_codec=args.determ, seed=args.seed)[3]
            # Steps are independent, but the allocator holds on to every
            # intermediate; without this the 4th step OOMs on a 24 GB card.
            del g_xyz_in, g_fea_in
            gc.collect()
            torch.cuda.empty_cache()
    torch.cuda.synchronize(); t2 = time.time()
    timings["time_encode"] = t2 - t1
    # Compression proper, so it lines up with the C3DGS time_total: reading the
    # .ply and building the network are setup, not compression.
    timings["time_total"] = timings["time_encode"]
    print('time:', timings["time_encode"])

    print(f"{args.ply_path_from} compressed! Save bitstreams to {args.bit_path_to}.")
    orig_size = os.path.getsize(args.ply_path_from)/1024/1024
    # Two different numbers, both worth keeping: `size_MB` is what the
    # bitstreams actually occupy on disk, `size_estimated_MB` is the codec's own
    # bit accounting (what the paper reports).
    size_MB = dir_size_bytes(os.path.join(args.bit_path_to, str(lmd))) / 1024 / 1024
    print(f"Original size: {orig_size:.4f} MB. Compressed size: {size_MB:.4f} MB "
          f"(bit accounting: {ttl_size:.4f} MB). Compression ratio: {orig_size/size_MB:.4f} X")

    out_dir = args.out_dir or os.path.dirname(os.path.normpath(args.bit_path_to))
    os.makedirs(out_dir, exist_ok=True)

    # encode_meta.json is the handshake with the decoder: it must be able to
    # recover the seed, otherwise the chunk permutation cannot be reproduced.
    meta = {
        "seed": args.seed,
        "lmd": args.lmd,
        "nr": args.nr,
        "determ": args.determ,
        "per_step_size": per_step_size,
        "num_gaussians": int(N_gaussian),
        "step_num": step_num,
        "chunk_size_list": chunk_size_list,
        "size_MB": size_MB,
        "size_estimated_MB": float(ttl_size),
        "orig_ply_MB": orig_size,
        "compression_ratio": orig_size / size_MB if size_MB > 0 else 0.0,
        "ply_path_from": args.ply_path_from,
        "bit_path": args.bit_path_to,
    }
    with open(os.path.join(out_dir, "encode_meta.json"), "w") as f:
        json.dump(meta, f, indent=4)
    with open(os.path.join(out_dir, "times.json"), "w") as f:
        json.dump(timings, f, indent=4)


if __name__ == "__main__":
    parser = ArgumentParser(description="dataset_param")
    parser.add_argument("--lmd", default=1e-4, choices=[1e-4, 2e-4, 4e-4, 8e-4, 16e-4], type=float)
    parser.add_argument("--nr", default=3, type=float)
    parser.add_argument("--determ", default=1, type=float)
    parser.add_argument("--seed", default=0, type=int,
                        help="RNG seed; also seeds the gaussian shuffle that defines the chunk split")
    parser.add_argument("--per_step_size", default=100_0000, type=int,
                        help="gaussians per independent compression step (lower it if CUDA OOMs)")
    parser.add_argument("--bit_path_to", default="./bitstreams/tmp/", type=str)
    parser.add_argument("--ply_path_from", default="./xxx/point_cloud.ply", type=str)
    parser.add_argument("--out_dir", default="", type=str,
                        help="where times.json / encode_meta.json go (default: parent of --bit_path_to)")
    args = parser.parse_args(sys.argv[1:])
    train(args)
