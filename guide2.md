# FCGS Compression Improvements — Detailed Explanations, Sources, and Mask-Context Re-implementation Guide

Companion document to [GUIDE.md](GUIDE.md). Everything below is grounded in three kinds of
sources, tagged throughout:

- **[repo]** — code read in this repository (file:line),
- **[measured]** — experiments run during the implementation session (scripts and logs you can re-run),
- **[lit]** — published literature that the design ideas come from.

No external web sources were consulted during the work — the literature references are the
classical foundations these techniques are built on.

---

## 1. What the mask is, and what "mask spatial context" means

### 1.1 The mask in FCGS

FCGS (*Fast Feedforward 3D Gaussian Splatting Compression*, Chen et al., ICLR 2025,
[arXiv:2410.08017](https://arxiv.org/pdf/2410.08017) — linked from [README.md](README.md))
compresses each Gaussian's 56 attributes (48 SH color coefficients + opacity + 3 scales +
4 rotation quaternions). The paper's **Multi-path Entropy Module (MEM)** routes each Gaussian's
color features down one of two paths:

- **fea path**: an autoencoder compresses the 48 SH coefficients into a 256-d latent that is
  quantized and entropy-coded (`model/FCGS_model.py:377-380`);
- **feq path**: the raw SH coefficients are directly quantized and entropy-coded, skipping the
  autoencoder (`model/FCGS_model.py:389-391`).

A tiny MLP (`Encoder_mask`, Linear→LeakyReLU→…→Sigmoid) looks at each Gaussian's features and
emits a score; thresholding at 0.01 produces one **binary decision per Gaussian**
**[repo: FCGS_model.py:345-346]**. The decoder cannot recompute this decision (it does not have
the original features), so the mask — one bit per Gaussian, ~1.6 M bits for `room` — must be
transmitted in the bitstream. That is the `mask.b` file.

### 1.2 How it was coded before, and why that is improvable

The original coder (`model/encodings_cuda.py:436-465`, **[repo]**) computed one number — the
global fraction of 1-bits, `prob_1 = x.sum()/x.numel()` — and arithmetic-coded every element
against that same Bernoulli distribution. The cost is N·H(p) bits where H is the binary entropy
function. This is optimal **if and only if the bits are independent and identically
distributed** — Shannon's source coding theorem **[lit: Shannon, "A Mathematical Theory of
Communication", 1948]**.

But the mask bits are not independent. Each bit is a function of the Gaussian's SH content, and
neighboring Gaussians in space tend to have similar appearance (they paint the same surface),
so neighbors tend to make the same fea/feq decision. Information theory says that whenever
there is statistical dependence, **conditional entropy is lower than marginal entropy**:
H(X | neighbors) ≤ H(X), with equality only under independence **[lit: Shannon 1948; Cover &
Thomas, *Elements of Information Theory*]**. Coding each bit against a probability conditioned
on its already-decoded neighbors therefore never costs more, and saves exactly the mutual
information between a bit and its context. This is the principle behind every modern binary
entropy coder: JBIG's context templates for bilevel images **[lit: ISO/IEC 11544]** and CABAC's
context models in H.264/HEVC **[lit: Marpe, Schwarz & Wiegand, "Context-Based Adaptive Binary
Arithmetic Coding in the H.264/AVC Video Compression Standard", IEEE TCSVT 2003]**.

"**Spatial context**" simply means: the conditioning information is the values of spatially
nearby, already-decoded mask bits. The repo already provides a spatial neighborhood structure
for free: before anything is coded, the Gaussians are sorted by the **Morton (Z-order) curve**
of their voxelized coordinates (`model/gpcc_utils.py:113-131`, **[repo]**). A Morton curve maps
3-D positions to a 1-D order that preserves locality — consecutive indices are usually spatial
neighbors **[lit: G.M. Morton, "A computer oriented geodetic data base and a new technique in
file sequencing", IBM 1966]**. So "the two array neighbors at distance d" is a cheap,
decoder-reproducible proxy for "the two nearest spatial neighbors."

### 1.3 How much it can be improved — honest numbers

The gain equals the actual spatial redundancy in the mask, which is an empirical question:

- On synthetic strongly-correlated masks the hierarchical coder saved **~90%**
  **[measured: `tools/test_mask_ctx.py` output]**.
- On real FCGS masks the redundancy is much smaller: **−0.3% to −4.1%** of mask bits across
  room/train/bicycle × λ∈{1e-4, 16e-4} **[measured: `/d01/luis/bench_maskctx/subset.log`]**.
  The real masks are close to i.i.d. — `Encoder_mask` output varies at high spatial frequency.
  The earlier analysis' estimate of "30–60%" was wrong for real data; the adaptive design
  (below) is what guarantees we capture whatever redundancy exists without ever paying a
  penalty when there is none.

---

## 2. The implementations in rich detail

### 2.1 Hierarchical mask context coder

**The central engineering constraint** (this drove the whole design): the repo's CUDA range
coder decodes all N symbols **in parallel** and therefore needs the complete N×3 CDF table
*before* decoding starts — `arithmetic.arithmetic_decode(output_cdf, …)` takes the CDF tensor
as an input **[repo: encodings_cuda.py:485-492]**. A classical adaptive context coder
(JBIG/CABAC style) updates probabilities bit-by-bit, which is inherently serial and would
destroy FCGS's "fast" property. So the context for a bit may only depend on bits decoded in a
*previous pass*.

The standard solution in the learned-compression world is multi-pass conditioning: decode half
the symbols context-free, then decode the other half conditioned on the first — the
**checkerboard context model** **[lit: He, Zheng, Sun, Wang & Qin, "Checkerboard Context Model
for Efficient Learned Image Compression", CVPR 2021]**. The first version implemented here was
exactly that, in 1-D: even Morton indices coded globally, odd indices conditioned on their two
even neighbors.

The unit test then showed the even half (still coded context-free, 50% of all bits) dominated
the cost **[measured: test_mask_ctx.py, 2-slice = 527 kb vs 3-slice theoretical = 289 kb on
correlated input]**, so the scheme was generalized to the **hierarchical dyadic scheme** that
shipped:

- **Level 0**: bits at indices `0, m, 2m, …` with `m = 2^(L−1)`, coded with one global
  probability `p0`.
- **Level k (1…L−1)**: bits at odd multiples of `m_k = 2^(L−1−k)`; each is conditioned on the
  bits at `i−m_k` and `i+m_k`, which are guaranteed decoded in earlier levels. Two binary
  neighbors → **4 contexts**; the 4 conditional probabilities are simply counted from the data
  at encode time and stored in the header.

This recursion — transmit endpoints, then midpoints conditioned on their endpoints, halving the
stride each level — is structurally the same idea as **binary interpolative coding**
**[lit: Moffat & Stuiver, "Binary Interpolative Coding for Effective Index Compression",
*Information Retrieval* 3(1), 2000]** and the dyadic refinement used in progressive schemes
(e.g., PNG's Adam7 interlacing). Each level is one fully-parallel GPU pass, so decode is L
kernel launches instead of 1 — negligible (measured encode time 17.3 s → 17.4 s on `room`
**[measured: subset.log]**).

**Adaptive depth L.** Before encoding, the encoder computes the *theoretical* cost of L = 1…6
directly from bit-count histograms (no trial encodes — for each candidate, the level-0 cost is
N₀·H(p₀) and each refinement level costs Σ_ctx n_ctx·H(p_ctx), plus the real per-substream
overhead), and picks the cheapest **[repo: `_mask_level_bits_estimate`, encodings_cuda.py]**.
Choosing the model that minimizes total description length (model parameters + data given
model) is the **MDL principle** **[lit: Rissanen, "Modeling by shortest data description",
*Automatica* 1978]**. This is why the coder is safe: for an i.i.d. mask it selects small L and
costs within 0.06% of the old coder; for a correlated mask it selects L=5–6 and wins big
**[measured: test_mask_ctx.py — i.i.d. overhead −0.01…−0.06%, correlated saving 90%]**.

**Numerical correctness details**, all forced by how arithmetic coding works:

- Encoder and decoder must build **bit-identical CDFs**, or decoding diverges. So probabilities
  are stored in the header as the exact float32 values used at encode, and both sides rebuild
  the CDF from those bytes (no re-deriving, no double-rounding).
- Probabilities are clamped to [1e−6, 1−1e−6]: a zero-probability symbol would get zero code
  space **[lit: Witten, Neal & Cleary, "Arithmetic Coding for Data Compression", CACM 1987]**.
  The repo's kernel additionally adds `+ sym` to each CDF integer boundary, guaranteeing every
  symbol ≥1 unit of code space — so extreme probabilities degrade efficiency but cannot break
  decoding **[repo: submodules/arithmetic/arithmetic_kernel.cu:123-125]**.
- The mask is coded **before** the deterministic shuffle. The shuffle
  (`torch.manual_seed(1); randperm`) **[repo: FCGS_model.py:357-360]** destroys spatial
  adjacency, so coding had to move before it; the mask MLP is per-row, so computing it
  pre-shuffle yields identical values (verified bitwise — the decoded PLYs match byte-for-byte
  **[measured]**). One subtlety: the mask decode must not consume any torch RNG, or the
  subsequent `randperm` would change — the new decode path uses no RNG ops and runs before
  `manual_seed(1)` is even called.

**Where the decision evidence came from:** the i.i.d. coding waste is
`encodings_cuda.py:440-441` **[repo]**; the parallel-decoder constraint is
`encodings_cuda.py:485-492` **[repo]**; Morton order availability at decode time is
`gpcc_utils.py:225` (decoder re-sorts voxels by Morton code) **[repo]**; the
2-slice→hierarchical upgrade and the adaptive-L safety property are **[measured]**.

### 2.2 G-PCC tuning

The xyz coordinates are coded by MPEG's **G-PCC** reference codec TMC13 (standard: ISO/IEC
23090-9; binary bundled at `mpeg-pcc-tmc13/build/tmc3/tmc3`, release-v23.0-rc2 **[repo]**).
The repo invoked it with a fixed flag string **[repo: gpcc_utils.py:17-21]**. The flags were
parameterized and `tools/gpcc_sweep.py` was built, which for every candidate config voxelizes →
encodes → decodes → **verifies exact voxel equality** (so a config can only be adopted if it is
provably lossless on that data). Results **[measured: sweep output on room + train at 3 bit
depths]**:

| Config | Effect | Decision |
|---|---|---|
| `cabac_bypass_stream_enabled_flag=0` | −0.6% size, same speed | **adopted** as default |
| `planarEnabled=1`, `intra_pred…=6` | ±0.00% | rejected (neutral) |
| `geomTreeType=1` (predictive geometry) | **+18…31% size, ~6× slower encode** | rejected |

The flag meanings come from the TMC13 documentation bundled in the `mpeg-pcc-tmc13` source tree
**[repo]**. The bypass flag routes "bypass" (equiprobable) bits into a separate raw stream for
speed; disabling it sends them through the arithmetic coder, recovering a little compression.
Predictive geometry is designed for sparse, scan-ordered LiDAR clouds; dense, volumetrically
distributed 3DGS clouds favor octree coding — the earlier analysis claimed predictive geometry
would win 10–20%, and the measurement refuted it. This is why every adoption was gated on the
sweep rather than on received wisdom.

### 2.3 xyz bit depth (the one lossy lever)

xyz is quantized to a 16-bit grid per axis before G-PCC **[repo: gpcc_utils.py:10,
`VOXELIZE_SCALE_FACTOR`]**. Fewer bits = coarser grid = smaller stream (15-bit: −23%, 14-bit:
−45% of xyz **[measured: sweep]**) but real geometric error — a classic rate-distortion
trade-off. The approved gate was ΔPSNR ≤ 0.05 dB. End-to-end renders showed
**[measured: xyzbits.log, traingate.log]**: on `room`, 15-bit costs only −0.02 dB; on `train`
it costs −0.14 dB and fails. The physical reason: the voxel grid spans the scene's bounding
box, so absolute voxel size grows with scene extent — a fixed bit budget that is sub-millimeter
in a room is centimeters on a large outdoor scene. Hence: default stays 16-bit, `--xyz_bits 15`
available per-scene, 14-bit rejected. The format was made self-describing (magic `FCGSXYZ1` +
bit-depth byte in `xyz_gpcc.bin`, only when ≠16) so decoders need no side information.

### 2.4 `.fcgs` container

Pure engineering, no literature needed: a tar-like single file (magic, version, entry table of
name/offset/size, concatenated blobs) in `tools/fcgs_container.py`. It does not change the
information content — its value is one artifact per scene instead of 200+ loose files, and
eliminating filesystem block slack. Decoders auto-detect it. Verified byte-identical round-trip
**[measured]**.

---

## 3. Guide: re-implementing the mask spatial context yourself

Written so it can be redone from a clean FCGS checkout.

### Step 0 — understand the two invariants

1. At decode time, before anything else is decoded, you have: xyz in Morton order (G-PCC
   decode + Morton re-sort, `gpcc_utils.py:225`) and N. You do *not* have features. So mask
   context may only use *position/order* and *previously decoded mask bits*.
2. The CUDA decoder needs all per-element CDFs up front (`encodings_cuda.py:485-492`) —
   context must come from earlier *passes*, never from the same pass.

### Step 1 — per-element Bernoulli primitives

Copy the existing `encoder`/`decoder` pair (`encodings_cuda.py:436-492`) and change one thing:
accept a probability *tensor* `p1[M]` instead of a scalar. The CDF construction is the same:
`output_cdf = cat([zeros, 1−p1.unsqueeze(-1), ones], dim=-1)`. Handle M=0 by returning empty
byte strings. (~25 lines.)

### Step 2 — the context function

For target bits at odd multiples of stride `m` (positions `i = m + j·2m`), their neighbors live
at `i±m`, which are exactly the elements of the array sliced as `sym[0::2m]`:

```python
def mask_ctx_ids(sym_A, M_B):          # sym_A = sym[0::2m], M_B = len(sym[m::2m])
    left  = sym_A[:M_B].long()
    right = sym_A[1:M_B+1].long()
    if right.shape[0] < M_B:           # last target lacks a right neighbor (N even)
        right = torch.cat([right, left[-1:]])
    return left * 2 + right            # context id in {0,1,2,3}
```

Both sides must call this *identical* function — any asymmetry desynchronizes the range coder.

### Step 3 — encoder

Binarize the mask (`floor`, int16), then:

1. For each candidate L in 1…6 (capped so `2^(L−1) ≤ N`), estimate cost from histograms only:
   level 0 = `N₀·H(p₀)`; each level k = `Σ_ctx [ −n₁ log₂ p − n₀ log₂(1−p) ]` with `p = n₁/n`
   clamped; add real per-substream overhead (8 bytes + 4·⌈M/chunk⌉ bytes for the coder's
   chunk-count array). Pick argmin (MDL).
2. For the chosen L: encode level 0 with `p0` broadcast; for each level k, `bincount` the 4
   contexts, clamp probabilities to [1e−6, 1−1e−6], **cast to float32 first**, build
   `p1 = p_ctx_tensor[ctx]`, encode.
3. Write header: `uint8 version, uint8 L, int32 N, float32 p0, float32 p_ctx[(L−1)×4]`, then
   per level `int32 len_cnt | cnt | int32 len_stream | stream`.

### Step 4 — decoder

Read header; allocate `out[N]`; decode level 0 into `out[0::2^(L−1)]`; then for k = 1…L−1:
`ctx = mask_ctx_ids(out[0::2m_k], M_k)` (that slice is fully populated by induction), build
`p1` from the *stored* float32 table, decode, scatter into `out[m_k::2m_k]`. The CDFs match the
encoder's bit-for-bit because both were built from the same stored bytes.

### Step 5 — pipeline integration

In `compress()`: compute the mask immediately after the Morton sort and encode it there (before
G-PCC and before `torch.manual_seed(1)`); after the shuffle, add
`mask = mask[shuffled_indices]`. In `decomprss()`: decode the mask right after G-PCC xyz
decode, then shuffle it together with xyz. Dispatch on filename (`mask_ctx.b` vs `mask.b`) for
backward compatibility. Do not put any torch-RNG op in the new code path.

### Step 6 — verify, in this order

Each stage catches a different failure class:

1. **Unit round-trip** on synthetic masks — i.i.d. at p∈{0.02, 0.5, 0.98}, all-0/all-1,
   N∈{1,2,3,10⁶}, and a smoothed-noise correlated mask — assert exact equality (catches coder
   bugs, padding, empty contexts).
2. **Mode A/B**: encode one scene with old and new coder; the two decoded PLYs must be
   **byte-identical** (catches integration bugs: ordering, shuffle, RNG).
3. **Backward compat**: decode an old-format bitstream with the new code (catches dispatch
   bugs).
4. Only then look at bits: compare `bits_mask`, and check encode/decode wall time.

### Pitfalls actually hit during implementation

- GPU contention silently kills concurrent encodes — run benchmarks strictly serially.
- `grep`-filtered logs swallow stderr — capture full output of anything that can fail.
- The biggest conceptual one — *measure the redundancy before promising gains*: the synthetic
  90% became 0.3–4% on real masks, and only the adaptive-L design made the change
  unconditionally worthwhile.

---

## 4. Source summary

| Decision | Source |
|---|---|
| Mask is i.i.d.-coded, one global prob | **[repo]** encodings_cuda.py:440-441 |
| Decoder needs full CDF up front → multi-pass design | **[repo]** encodings_cuda.py:485-492; arithmetic_kernel.cu |
| Morton order available pre-mask at decode | **[repo]** gpcc_utils.py:225; FCGS_model.py decode flow |
| Conditional entropy ≤ entropy (why context helps) | **[lit]** Shannon 1948; Cover & Thomas |
| Binary context modeling practice | **[lit]** JBIG (ISO/IEC 11544); CABAC, Marpe et al., IEEE TCSVT 2003 |
| Two-pass parallel context | **[lit]** He et al., "Checkerboard Context Model…", CVPR 2021 |
| Dyadic midpoint refinement | **[lit]** Moffat & Stuiver, binary interpolative coding, 2000 |
| Adaptive L selection | **[lit]** Rissanen, MDL, 1978 |
| Zero-probability clamping | **[lit]** Witten, Neal & Cleary, CACM 1987 |
| Morton/Z-order locality | **[lit]** Morton, IBM 1966 |
| FCGS architecture (MEM, two paths) | **[lit/repo]** Chen et al., ICLR 2025, arXiv:2410.08017 + model code |
| G-PCC flags & codec | **[repo]** mpeg-pcc-tmc13 (TMC13 v23, ISO/IEC 23090-9) |
| cabac flag adoption, predgeom rejection, bit-depth gate, real mask gains | **[measured]** tools/gpcc_sweep.py, tools/test_mask_ctx.py, `/d01/luis/bench_maskctx/*.log`, `full_suite.csv` |
