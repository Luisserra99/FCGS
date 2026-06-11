"""Round-trip unit test for the two-slice mask context coder (encoder_mask_ctx / decoder_mask_ctx).

Run:  python tools/test_mask_ctx.py   (from the FCGS repo root, FCGS conda env)
"""
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from model.encodings_cuda import encoder_mask_ctx, decoder_mask_ctx, encoder, get_binary_vxl_size


def correlated_mask(N, smooth=2000, thresh=0.0, device='cuda'):
    # smoothed random walk -> spatially correlated binary sequence
    torch.manual_seed(42)
    noise = torch.randn(N, device=device)
    kernel = torch.ones(1, 1, smooth, device=device) / smooth
    pad = smooth // 2
    sm = torch.nn.functional.conv1d(noise.view(1, 1, -1), kernel, padding=pad).view(-1)[:N]
    return (sm > thresh).float()


def run_case(name, x):
    with tempfile.TemporaryDirectory() as td:
        f = os.path.join(td, 'mask_ctx.b')
        bits = encoder_mask_ctx(x, file_name=f)
        dec = decoder_mask_ctx(x.numel(), file_name=f, device='cuda')
        ref = torch.floor(x.detach().view(-1)).to(dec.dtype).to(dec.device)
        ok = torch.equal(dec, ref)
        # compare against the legacy global-Bernoulli coder's actual size
        f2 = os.path.join(td, 'mask.b')
        bits_legacy = encoder(x, file_name=f2) if x.numel() > 0 else 0
    status = 'OK ' if ok else 'FAIL'
    print(f'[{status}] {name:34s} N={x.numel():9d}  ctx={bits:10d}b  legacy={bits_legacy:10d}b  '
          f'saving={100.0 * (1 - bits / max(bits_legacy, 1)):6.2f}%')
    return ok


def main():
    cases = []
    for N in [1, 2, 3, 17, 1_000_000, 1_099_999]:
        torch.manual_seed(N)
        for p in [0.02, 0.5, 0.98]:
            cases.append((f'iid p={p} N={N}', (torch.rand(N, device='cuda') < p).float()))
        cases.append((f'all-zeros N={N}', torch.zeros(N, device='cuda')))
        cases.append((f'all-ones N={N}', torch.ones(N, device='cuda')))
    for N in [1_000_000, 1_099_999]:
        cases.append((f'correlated N={N}', correlated_mask(N)))
        cases.append((f'correlated-sparse N={N}', correlated_mask(N, thresh=0.02)))
    # column-vector input shape as used by compress(): [N, 1]
    cases.append(('shape [N,1] correlated', correlated_mask(500_000).view(-1, 1)))

    fails = sum(0 if run_case(n, x) else 1 for n, x in cases)
    print(f'\n{len(cases) - fails}/{len(cases)} passed')
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
