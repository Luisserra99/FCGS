import os
import sys
import time
from argparse import ArgumentParser

import torch
import torch.nn as nn

from bitstream_io import count_steps, load_encode_meta, resolve_seed
from gaussian_renderer import GaussianModel
from model.FCGS_model import FCGS


def train(args):

    lmd = args.lmd
    lmd_dir = os.path.join(args.bit_path_from, str(lmd))
    step_num = count_steps(lmd_dir)
    out_dir = args.out_dir or os.path.dirname(os.path.normpath(args.bit_path_from))
    seed = resolve_seed(args.seed, load_encode_meta(out_dir))
    print(f"Decoding {step_num} step(s) with chunk-shuffle seed {seed}")
    chunk_size_list = [200_0000, 100_0000, 100_0000]

    CM = FCGS(
        Q=1,
        resolutions_list=[300, 400, 500],
        resolutions_list_3D=[70, 80, 90],
        norm_radius=args.nr,
    ).cuda()
    CM.load_state_dict(torch.load(f'./checkpoints/checkpoint_{lmd}.pkl'), strict=True)

    g_xyz_list = []
    g_fea_list = []
    CM.eval()
    torch.cuda.synchronize(); t1 = time.time()
    with torch.no_grad():
        for s in range(step_num):
            bit_save_path = os.path.join(lmd_dir, str(s))
            g_xyz_out, g_fea_out = CM.decomprss(root_path=bit_save_path, chunk_size_list=chunk_size_list, seed=seed)
            g_xyz_list.append(g_xyz_out)
            g_fea_list.append(g_fea_out)

    g_xyz = torch.cat(g_xyz_list, dim=0)
    g_fea = torch.cat(g_fea_list, dim=0)
    torch.cuda.synchronize(); t2 = time.time()
    print('time:', t2 - t1)

    with torch.no_grad():
        gaussians = GaussianModel(3)  # dataset.sh_degree = 3
    f_dc, f_rst, op, sc, ro = torch.split(g_fea, split_size_or_sections=[3, 45, 1, 3, 4], dim=-1)
    gaussians._xyz = nn.Parameter(g_xyz)
    gaussians._features_dc = nn.Parameter(f_dc.view(-1, 1, 3))
    gaussians._features_rest = nn.Parameter(f_rst.view(-1, 15, 3))
    gaussians._opacity = nn.Parameter(op.view(-1, 1))
    gaussians._scaling = nn.Parameter(sc.view(-1, 3))
    gaussians._rotation = nn.Parameter(ro.view(-1, 4))
    # Assigning the tensors directly bypasses load_ply, which is the only place
    # that promotes active_sh_degree; left at 0 the rasterizer would drop all 45
    # SH coefficients and render the DC term alone.
    gaussians.active_sh_degree = gaussians.max_sh_degree

    gaussians.save_ply(args.ply_path_to)

    print(f"Decompressed ply file saved to {args.ply_path_to}!")


if __name__ == "__main__":
    parser = ArgumentParser(description="dataset_param")
    parser.add_argument("--lmd", default=1e-4, choices=[1e-4, 2e-4, 4e-4, 8e-4, 16e-4], type=float)
    parser.add_argument("--nr", default=3, type=float)
    parser.add_argument("--determ", default=1, type=float)
    parser.add_argument("--seed", default=None, type=int,
                        help="chunk-shuffle seed; defaults to the one in encode_meta.json")
    parser.add_argument("--bit_path_from", default="./bitstreams/tmp/", type=str)
    parser.add_argument("--ply_path_to", default="./bitstreams/tmp/point_cloud.ply", type=str)
    parser.add_argument("--out_dir", default="", type=str,
                        help="where encode_meta.json lives (default: parent of --bit_path_from)")
    args = parser.parse_args(sys.argv[1:])
    train(args)
