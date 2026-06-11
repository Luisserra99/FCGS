import time

import torch
from model.FCGS_model import FCGS
import os
from gaussian_renderer import GaussianModel
import numpy as np
from argparse import ArgumentParser
import sys


def train(args):
    # assert False, 'check scene dataloader first!'

    with torch.no_grad():
        gaussians = GaussianModel(3)  # dataset.sh_degree = 3
        gaussians.load_ply(path=args.ply_path_from)
    g_xyz = gaussians._xyz.detach()
    N_gaussian = g_xyz.shape[0]

    per_step_size = args.per_step_size
    if per_step_size == 100_0000 and N_gaussian > 100_0000 and N_gaussian < 110_0000:
        per_step_size = 110_0000

    _features_dc = gaussians._features_dc.detach().view(N_gaussian, -1)  # [N, 1, 3] -> [N, 3]
    _features_rest = gaussians._features_rest.detach().view(N_gaussian, -1)  # [N, 15, 3] -> [N, 45]
    _opacity = gaussians._opacity.detach()  # [N, 1]
    _scaling = gaussians._scaling.detach()  # [N, 3]
    _rotation = gaussians._rotation.detach()  # [N, 4]
    g_fea = torch.cat([_features_dc, _features_rest, _opacity, _scaling, _rotation], dim=-1)  # [N, 56]

    step_num = int(np.ceil(N_gaussian / per_step_size))
    lmd = args.lmd
    chunk_size_list = [200_0000, 100_0000, 100_0000]

    CM = FCGS(
        Q=1,
        resolutions_list=[300, 400, 500],
        resolutions_list_3D=[70, 80, 90],
        norm_radius=args.nr,
    ).cuda()
    CM.load_state_dict(torch.load(f'./checkpoints/checkpoint_{lmd}.pkl'), strict=True)

    ttl_size = 0
    stream_sizes = {'xyz': 0, 'mask': 0, 'fea': 0, 'feq': 0, 'geo': 0}
    CM.eval()
    torch.cuda.synchronize(); t1 = time.time()
    with torch.no_grad():
        for s in range(step_num):
            bit_save_path = os.path.join(args.bit_path_to, f"{lmd}/{s}")
            os.makedirs(bit_save_path, exist_ok=True)
            g_xyz_in = g_xyz[s*per_step_size:s*per_step_size+per_step_size]
            g_fea_in = g_fea[s*per_step_size:s*per_step_size+per_step_size]
            out = CM.compress(g_xyz_in, g_fea_in, root_path=bit_save_path, chunk_size_list=chunk_size_list, determ_codec=args.determ, mask_mode=args.mask_mode, xyz_bits=args.xyz_bits)
            ttl_size += out[3]
            for key, idx in (('xyz', 4), ('mask', 5), ('fea', 6), ('feq', 9), ('geo', 12)):
                stream_sizes[key] += float(out[idx])
    torch.cuda.synchronize(); t2 = time.time()
    print('time:', t2-t1)

    print(f"{args.ply_path_from} compressed! Save bitstreams to {args.bit_path_to}.")
    orig_size = os.path.getsize(args.ply_path_from)/1024/1024
    print('Per-stream sizes (MB): ' + ', '.join(f'{k}: {v:.4f}' for k, v in stream_sizes.items()))
    print(f"Original size: {orig_size:.4f} MB. Compressed size: {ttl_size:.4f} MB. Compression ratio: {orig_size/ttl_size:.4f} X")

    if args.pack:
        from tools.fcgs_container import pack as fcgs_pack
        src_dir = os.path.join(args.bit_path_to, str(lmd))
        out_path = os.path.join(args.bit_path_to, f'{lmd}.fcgs')
        n_files, packed_size = fcgs_pack(src_dir, out_path)
        print(f'Packed {n_files} bitstream files into {out_path} ({packed_size/1024/1024:.4f} MB)')
        if args.pack_rm:
            import shutil
            shutil.rmtree(src_dir)


if __name__ == "__main__":
    parser = ArgumentParser(description="dataset_param")
    parser.add_argument("--lmd", default=1e-4, choices=[1e-4, 2e-4, 4e-4, 8e-4, 16e-4], type=float)
    parser.add_argument("--nr", default=3, type=float)
    parser.add_argument("--determ", default=1, type=float)
    parser.add_argument("--mask_mode", default="ctx", choices=["global", "ctx"], type=str)
    parser.add_argument("--xyz_bits", default=16, choices=[14, 15, 16], type=int)
    parser.add_argument("--pack", action="store_true", help="pack all bitstream files into a single .fcgs container")
    parser.add_argument("--pack_rm", action="store_true", help="remove loose bitstream files after packing")
    parser.add_argument("--per_step_size", default=100_0000, type=int, help="max Gaussians per encoding step; lower it if encoding runs out of GPU memory")
    parser.add_argument("--bit_path_to", default="./bitstreams/tmp/", type=str)
    parser.add_argument("--ply_path_from", default="./xxx/point_cloud.ply", type=str)
    args = parser.parse_args(sys.argv[1:])
    train(args)

