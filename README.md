# Residual → U-Net rewiring

Turn a plain residual stack (ResNet / transformer style, every layer at full
resolution) into a U-Net whose middle layers run on a 4x smaller token grid,
**without pooling the residual stream**.

The trick uses the residual structure: a residual layer only *adds* an update
to the stream. So the stream itself can go around the low-resolution part
unchanged, and only the middle layers' *update* is computed at low resolution
and upsampled back:

```
original:   x = h + g(h)            g = what the middle layers add
rewired:    x = h + U g(P h)        P = downsample, U = upsample
```

What you get:

- **Exact when the middle layers are identity.** If the middle adds nothing,
  the rewired network equals the original bit for bit. A naive U-Net doesn't.
- **One known error term for any weights.** For linear layers the error is
  exactly $(UP - I)\,g(h)$: the pool → upsample roundtrip applied to the
  update, never to the stream.
- **Cheaper middle.** Each compressed layer processes 1/4 of the tokens, and
  attention in those layers costs about 1/16.

[`residual_unet_7_layers.py`](residual_unet_7_layers.py) is a standalone
reference implementation. It uses 7 matmul-only layers so every identity
below can be checked numerically.

```
python residual_unet_7_layers.py
```

Requires only PyTorch.

---

## Setup and notation

Tokens sit on an $H \times W$ grid, so spatial pooling is well defined. The
activations are $x \in \mathbb{R}^{B \times N \times C}$ with $N = HW$.

| symbol | meaning | mixes |
|---|---|---|
| $h$ | residual stream entering the middle | – |
| $W_i$ | weight of middle layer $i$ (`x ← x + x @ W_i`) | channels, per token |
| $P$ | 2x2 average pool, $N \to N/4$ | tokens, per channel |
| $U$ | bilinear 2x upsample, $N/4 \to N$ | tokens, per channel |
| $f$ | the whole middle stack, $f(h) = h + g(h)$ | – |
| $g$ | the middle's total update, $g(h) = f(h) - h$ | – |

The reference model has 7 layers: `L0` (full res), `L1..L5` (middle,
compressed), `L6` (full res).

## The picture

Original: one resolution the whole way.

```
x0 -> L0 -> L1 -> L2 -> L3 -> L4 -> L5 -> L6 -> out
      \____________ all on N = 64 tokens ______/
```

Rewired: the U shape.

```
full res   x0 -> L0 ----------- skip: h -----------> (+) -> L6 -> out
                  \                                  /
                   P                                U
                    \                              /
low res              L1 -> L2 -> L3 -> L4 -> L5 -> (- low)
```

Rewired, in detail:

```
x0 -> L0 -> h ----------- skip: h itself, never pooled -----------+
            |                                                     |
            | P: 2x2 avg pool          (64 -> 16 tokens)          |
            v                                                     |
           low = P h                                              |
            |                                                     |
            | L1 -> L2 -> L3 -> L4 -> L5,  all on 16 tokens       |
            v                                                     |
           mid_out = low + u1 + u2 + u3 + u4 + u5                 |
            |        (each layer adds its update u_i)             |
            |                                                     |
            | subtract what went in                               |
            v                                                     |
           delta = mid_out - low = u1 + ... + u5   <- updates     |
            |                                                     |
            | U: bilinear x2           (16 -> 64 tokens)          |
            v                                                     v
           U delta ---------------------------------------------> (+)
                                                                  |
                                                 x = h + U delta  |
                                                                  v
                                                            L6 -> out
```

The key step is **"subtract what went in"**. The middle's input (a pooled,
blurred copy of the stream) is thrown away. Only the sum of what the middle
layers *added* goes back up.

In code, the whole rewire is:

```python
low     = pool(h)                  # P h
mid_out = middle(low)              # L1..L5 on N/4 tokens
x       = h + up(mid_out - low)    # h + U g(P h)
```

---

## The math

### 1. Linear layers: the middle is one matrix

Each layer is $x \mapsto x(I + W_i)$, so the five middle layers compose into

```math
f(h) = h\,M, \qquad M = (I + W_1)(I + W_2)(I + W_3)(I + W_4)(I + W_5),
```

and their total update is $g(h) = h\,(M - I)$.

### 2. Pooling commutes with the middle

$P$ acts on the token axis and $M$ acts on the channel axis. Linear maps on
different axes commute:

```math
(P h)\,(M - I) = P\,\big(h\,(M - I)\big) \quad\Longrightarrow\quad g(P h) = P\,g(h).
```

In words: "pool, then run five layers" gives the same update as "run five
layers, then pool".

### 3. The rewired output

```math
\text{original:}\quad x = h + g(h)
\qquad\qquad
\text{rewired:}\quad x = h + U\,g(P h) = h + U P\,g(h).
```

### 4. The error is the roundtrip applied to the update only

Subtract the two:

```math
x_\text{rewired} - x_\text{original} = (UP - I)\,g(h).
```

This gives two results:

- **Exactness.** If the middle is identity ($W_i = 0$), then $g = 0$ and the
  error is exactly zero. The stream $h$ never goes through $P$ or $U$.
- **Where error comes from.** $UP - I$ is zero on constant signals and small
  on smooth ones. It only affects fine spatial detail. So the error is small
  when the middle's *update* is spatially smooth, whatever the stream
  contains.

### 5. Why the naive U-Net is worse

A naive U-Net upsamples the middle's whole output: $x = U f(Ph)$. Expand
$f = \mathrm{id} + g$:

```math
x_\text{naive} = U P h + U\,g(P h) = x_\text{rewired} + (UP - I)\,h.
```

So the naive U-Net has **the same error plus the pool/upsample roundtrip
applied to the whole stream**. That second term is not zero even when the
middle does nothing: pooling throws away the stream's fine detail, and every
later layer sees the blurred version. The residual skip removes that term.
This identity is pure algebra: it holds for any $g$, linear or not, because
$U$ is linear.

### 6. General (nonlinear) blocks

Real blocks are $x + \mathrm{MLP}(x)$, $x + \mathrm{Attn}(x)$, and so on.
The bookkeeping is the same (`delta = mid_out - mid_in`), but $g$ is now
nonlinear. Add and subtract $UP\,g(h)$:

```math
x_\text{rewired} - x_\text{original}
= \underbrace{(UP - I)\,g(h)}_{\text{update blurred}}
+ \underbrace{U\big[g(Ph) - P\,g(h)\big]}_{\text{pooling doesn't commute with } g}.
```

- The first term is the same as in the linear case.
- The second term measures how much the middle's update changes when it sees
  a pooled input instead of pooling its full-res output. It is zero for
  per-token linear maps (section 2). For MLPs it is small when the input is
  locally smooth. For attention it depends on how much the attention pattern
  relies on fine detail.
- Exactness at an identity middle still holds: $g = 0$ makes both terms
  vanish.
- Section 5 still holds exactly: the naive U-Net always adds $(UP - I)\,h$
  on top.

---

## Cost

Matmul work scales with token count, so a compressed layer costs 1/4. With
$F$ full-res layers and $K$ compressed ones:

```math
\text{token cost} = \frac{F + K/4}{F + K}.
```

| layers | full-res | compressed | token cost |
|---|---|---|---|
| 7 (reference impl) | 2 | 5 | 0.46x |
| 12 | 4 | 8 | 0.50x |

Attention in compressed layers scales as $(N/4)^2 = N^2/16$, so on
attention-heavy models the savings in the middle are larger.

---

## Applying it to a real model

The reference implementation covers the core identity. Adapting a real
transformer also involves these choices:

- **Which layers to compress.** Keep at least the first and last blocks at
  full resolution. They are where the stream is read from and written back
  to at full detail.
- **Learnable P and U.** Use depthwise convs initialised to exactly 2x2
  average pool (down, 2x2 kernel of `0.25`) and bilinear 2x upsampling (up,
  transposed conv with 4x4 kernel `[.25,.75,.75,.25]` outer product, stride
  2). The rewire starts at the analytic version, and fine-tuning can sharpen
  both.
- **Optional learnable merge.** Instead of `h + U delta`, use
  `Linear([U delta, h])` initialised to `[I, I]`. It starts as the plain sum,
  so exactness is kept.
- **Non-grid tokens** (CLS, registers, text tokens). Don't pool them. Pass
  them through the middle next to the pooled grid tokens, and add their
  `delta` back directly without upsampling.
- **Positions: re-index, don't rescale.** Pooled tokens need positions on
  the coarse grid. For RoPE-style encodings, give them plain integer
  positions on the coarse grid ($0 \ldots H/2 - 1$). Don't rescale or
  average the fine positions (e.g. $0.5, 2.5, \ldots$) to keep the original
  extent. Re-indexing works better, at least for diffusion models, which are
  usually trained at low resolution first and high resolution after. A
  $H/2 \times W/2$ grid with integer positions is then an input the model
  has already seen, while fractional or stretched positions are not. Neither
  choice is exact for a pretrained model, so this is still part of what
  fine-tuning recovers.
- **Grid size.** $H$ and $W$ must be even (or be padded) for 2x2 pooling.
- **Fine-tune.** Start from the pretrained weights with the rewire applied.
  The stream path is untouched, so training only has to fix the
  low-resolution error terms above, not relearn the model.

---

## The reference implementation

[`residual_unet_7_layers.py`](residual_unet_7_layers.py):

| function | role |
|---|---|
| `Stack` | 7 residual matmul layers, `x ← x + x @ W` |
| `run` | apply a range of layers |
| `pool`, `up` | $P$ (2x2 avg pool) and $U$ (bilinear x2) on the token grid |
| `compressed_middle` | the rewire: `h + up(middle(pool(h)) - pool(h))`; `skip=False` gives the naive U-Net |
| `compressed` | full rewired model: `L0` → rewired middle → `L6` |
| `make_inputs` | smooth field + small noise, the regime where low-res compute works well |

It checks:

1. **Identity middle** ($W_1..W_5 = 0$): the rewired output equals the
   original bit for bit. The naive U-Net is off by about 15%.
2. **Random middle**: the rewired middle equals exactly
   $h + UP\,g(h)$ (sections 2–4). It also reports the error against the
   original and against the naive U-Net.
3. **Cost**: token cost relative to the original.

Expected output:

```
PASS identity middle (W1..W5=0): compressed == original bit-exact; without the skip it is off by 0.149
PASS random W1..W5: error == (UP - I) applied to the update only (0.0874 after the middle, 0.0870 after L6); pooling the whole stream instead: 0.1463

token cost vs original: 0.46x ((2 full + 5 quarter-res layers)/7; attention on the middle layers ~16x cheaper)
```

The random weights are scaled down (`Stack(scale=0.3)`) so each layer's
update is small next to the stream, as in a trained residual net. At
`scale=1` the five middle updates add up to about 4x the stream. Almost
everything is then "update", and the skip has little left to protect.
