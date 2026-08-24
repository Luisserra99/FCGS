"""Decode a scene from bitstreams, measure decode time and render quality.

Writes results.json / times.json in the same shape the C3DGS runs use, so
c3dgs/scripts/extract_metrics.py aggregates both methods into one CSV:

    <out_dir>/results.json   {"fcgs_<lmd>": {PSNR, SSIM, LPIPS, size, seed, ...}}
    <out_dir>/times.json     encode timings (from the encoder) + decode timing

Usage:
    python decode_single_scene_validate.py --lmd 1e-4 \\
        --bit_path_from RUN_DIR/bitstreams --ply_path_to RUN_DIR/decoded.ply \\
        --model_path  /path/to/trained/model \\
        --source_path /path/to/scene/images
"""

import json
import os
import sys
import time
from argparse import ArgumentParser

import torch
import torch.nn as nn
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, get_combined_args
from bitstream_io import count_steps, dir_size_bytes, load_encode_meta, resolve_seed
from gaussian_renderer import GaussianModel, render
from model.FCGS_model import FCGS
from scene import Scene
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim


def train(args, dataset, pipeline):
    lmd = args.lmd
    lmd_dir = os.path.join(args.bit_path_from, str(lmd))
    step_num = count_steps(lmd_dir)
    out_dir = args.out_dir or os.path.dirname(os.path.normpath(args.bit_path_from))
    meta = load_encode_meta(out_dir)
    seed = resolve_seed(args.seed, meta)
    print(f"Decoding {step_num} step(s) with chunk-shuffle seed {seed}")
    chunk_size_list = [200_0000, 100_0000, 100_0000]

    with torch.no_grad():
        scene = Scene(dataset, shuffle=False)
        views = scene.getTestCameras()

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
    # Time the codec alone: camera loading above and LPIPS setup below stay out
    # of the measurement.
    torch.cuda.synchronize(); t1 = time.time()
    with torch.no_grad():
        for s in range(step_num):
            bit_save_path = os.path.join(lmd_dir, str(s))
            g_xyz_out, g_fea_out = CM.decomprss(root_path=bit_save_path, chunk_size_list=chunk_size_list, seed=seed)
            g_xyz_list.append(g_xyz_out)
            g_fea_list.append(g_fea_out)

    g_xyz = torch.cat(g_xyz_list, dim=0)
    g_fea = torch.cat(g_fea_list, dim=0)
    torch.cuda.synchronize()
    time_decode = time.time() - t1
    print(f"Decode time: {time_decode:.4f} s")

    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
    f_dc, f_rst, op, sc, ro = torch.split(g_fea, split_size_or_sections=[3, 45, 1, 3, 4], dim=-1)
    gaussians._xyz = nn.Parameter(g_xyz)
    gaussians._features_dc = nn.Parameter(f_dc.view(-1, 1, 3))
    gaussians._features_rest = nn.Parameter(f_rst.view(-1, 15, 3))
    gaussians._opacity = nn.Parameter(op.view(-1, 1))
    gaussians._scaling = nn.Parameter(sc.view(-1, 3))
    gaussians._rotation = nn.Parameter(ro.view(-1, 4))
    # Assigning the tensors directly bypasses load_ply, which is the only place
    # that promotes active_sh_degree; left at 0 the rasterizer would drop all 45
    # SH coefficients and evaluate a DC-only render.
    gaussians.active_sh_degree = gaussians.max_sh_degree

    if args.ply_path_to:
        gaussians.save_ply(args.ply_path_to)
        print(f"Decompressed ply file saved to {args.ply_path_to}!")

    # Imported late: constructing the VGG network allocates GPU memory, which
    # must not compete with the decode above.
    import lpips
    lpips_fn = lpips.LPIPS(net='vgg').to('cuda')

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    with torch.no_grad():
        ssim_test_sum = 0
        L1_test_sum = 0
        lpips_test_sum = 0
        psnr_test_sum = 0
        for _, view in enumerate(tqdm(views, desc="Rendering progress")):
            rendering = render(view, gaussians, pipe=pipeline, bg_color=background)[
                "render"]  # [3, H, W]
            gt = view.original_image[0:3, :, :].to("cuda")
            rendering = torch.round(rendering.mul(255).clamp_(0, 255)) / 255.0
            ssim_test_sum += (ssim(rendering, gt)).mean().double().item()
            L1_test_sum += l1_loss(rendering, gt).mean().double().item()
            lpips_test_sum += lpips_fn(rendering, gt).mean().double().item()
            psnr_test_sum += psnr(rendering, gt).mean().double().item()
        ssim_avg = ssim_test_sum / len(views)
        Ll1_avg = L1_test_sum / len(views)
        lpips_avg = lpips_test_sum / len(views)
        psnr_avg = psnr_test_sum / len(views)

        print(f"Evaluation results: psnr: {psnr_avg:.4f}, ssim: {ssim_avg:.4f}, lpips: {lpips_avg:.4f}, Ll1: {Ll1_avg:.4f}")

    # --- record ------------------------------------------------------------
    os.makedirs(out_dir, exist_ok=True)
    meta = meta or {}
    size_MB = dir_size_bytes(lmd_dir) / 1024 / 1024
    orig_ply_MB = meta.get("orig_ply_MB", "")

    times = {}
    times_path = os.path.join(out_dir, "times.json")
    if os.path.exists(times_path):
        with open(times_path) as f:
            times = json.load(f)
    times["time_decode"] = time_decode
    times["time_ac_decode"] = time_decode
    with open(times_path, "w") as f:
        json.dump(times, f, indent=4)

    metrics = {
        "PSNR": psnr_avg,
        "SSIM": ssim_avg,
        "LPIPS": lpips_avg,
        "Ll1": Ll1_avg,
        # `size` is the on-disk total, matching what compress.py reports for
        # C3DGS; `size_estimated_MB` is the codec's own bit accounting.
        "size": size_MB,
        "size_estimated_MB": meta.get("size_estimated_MB", ""),
        "orig_ply_MB": orig_ply_MB,
        "compression_ratio": orig_ply_MB / size_MB if orig_ply_MB and size_MB > 0 else "",
        "seed": seed,
        "lmd": lmd,
        "nr": args.nr,
        "num_gaussians": meta.get("num_gaussians", int(g_xyz.shape[0])),
        "per_step_size": meta.get("per_step_size", ""),
        "step_num": step_num,
        # The FCGS protocol quantizes the render to 8 bits before scoring;
        # C3DGS scores the float render. Recorded so the CSV is unambiguous.
        "metric_protocol": "uint8",
        "images": dataset.images,
        "resolution": dataset.resolution,
        "time_total": times.get("time_total", ""),
        "time_encode": times.get("time_encode", ""),
        "time_decode": time_decode,
        "time_ac_decode": time_decode,
    }
    with open(os.path.join(out_dir, "results.json"), "w") as f:
        json.dump({f"fcgs_{lmd}": metrics}, f, indent=4)
    print(f"Wrote {os.path.join(out_dir, 'results.json')}")


if __name__ == "__main__":
    parser = ArgumentParser(description="dataset_param")
    # sentinel=True: every flag left off the command line becomes None, so
    # get_combined_args keeps the value from the model's cfg_args (images_8 /
    # images_4 / images_2 depending on the scene, resolution, white_background).
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--lmd", default=1e-4, choices=[1e-4, 2e-4, 4e-4, 8e-4, 16e-4], type=float)
    parser.add_argument("--nr", default=3, type=float)
    parser.add_argument("--seed", default=None, type=int,
                        help="chunk-shuffle seed; defaults to the one in encode_meta.json")
    parser.add_argument("--bit_path_from", default="./bitstreams/tmp/", type=str)
    parser.add_argument("--ply_path_to", default="./bitstreams/tmp/point_cloud.ply", type=str,
                        help="empty string skips writing the decoded .ply")
    parser.add_argument("--out_dir", default="", type=str,
                        help="where results.json / times.json go (default: parent of --bit_path_from)")
    args = get_combined_args(parser)

    # cfg_args carries neither of these, and with sentinel=True they arrive as
    # None; Scene() would then die on a missing attribute.
    if getattr(args, "data_device", None) is None:
        args.data_device = "cuda"
    if getattr(args, "lod", None) is None:
        args.lod = 0

    dataset = model.extract(args)
    pipeline_params = pipeline.extract(args)
    print(f"Evaluating against images='{dataset.images}' resolution={dataset.resolution} "
          f"eval={dataset.eval} from {dataset.source_path}")
    train(args, dataset, pipeline_params)
