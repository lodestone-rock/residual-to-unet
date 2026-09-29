r"""Residual -> U-Net rewiring: reference implementation in 7 matmuls.

WHAT THIS IS
------------
A way to turn a plain residual stack (ResNet / transformer style: every layer
does x = x + f(x), all at the same resolution) into a U-Net (the middle layers
run on a downsampled token grid, which makes them cheaper) while keeping the
full-resolution stream intact. The stream is never pooled; only what the
middle layers ADD to it is computed at low resolution. So the rewired network
is exactly the original when the middle layers are identity, and for any
trained weights the error is exactly one term: the middle's update, blurred
by the pool/upsample roundtrip. Fine-tuning a pretrained model only has to
recover that term; it does not have to relearn the model.

The layers here are plain matmuls so every claim can be checked by hand:

    a layer is ONE residual matmul        x <- x + x @ W
    tokens sit on an H x W grid           (so 2x2 pooling is well defined)
    7 layers: 1 start, 5 middle, 1 end    the 5 middle ones go low-res

NOTATION   (x is [batch, tokens N, channels C]; W acts on the right)

    h        the stream after the start layer L0     [B, N, C]
    L1..L5   the middle layers, weights W1..W5       mix CHANNELS, per token
    P        2x2 average pool,     N   -> N/4        mixes TOKENS, per channel
    U        bilinear 2x upsample, N/4 -> N          mixes TOKENS, per channel

ORIGINAL  (ResNet style: one resolution the whole way)

    x0 -> L0 -> L1 -> L2 -> L3 -> L4 -> L5 -> L6 -> out
          \____________ all on N = 64 tokens ______/

REWIRED, THE SHAPE  (why it is called a U-Net)

    full res   x0 -> L0 ----------- skip: h -----------> (+) -> L6 -> out
                      \                                  /
                       P                                U
                        \                              /
    low res              L1 -> L2 -> L3 -> L4 -> L5 -> (- low)

REWIRED, IN DETAIL

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

The key move is "subtract what went in": the middle's INPUT (the stream) is
not sent back up, only the sum of what the five middle layers ADDED to it.

THE ONE-LINE VERSION
--------------------
All five middle layers together are one linear map: h -> h @ M with
M = (I + W1)(I + W2)(I + W3)(I + W4)(I + W5). Their total update is
h @ (M - I). So:

    original:   h + h @ (M - I)          every token computes its own update
    rewired:    h + U P (h @ (M - I))    update computed on N/4 tokens,
                                         then upsampled

(Running the middle on P h gives (P h) @ (M - I) == P (h @ (M - I)), because
P only mixes tokens and M only mixes channels: linear maps acting on different
axes commute. That is why "pool, then run five layers" equals "run five
layers, then pool".)

WHY THE SKIP MAKES IT EXACT
---------------------------
A residual layer only ADDS an update to the stream. The rewire exploits that:
the stream h goes around the low-res part untouched, and only the updates are
pooled and upsampled.

  * Middle does nothing (W1..W5 = 0): delta = 0, so x = h and the rewired
    network equals the original bit for bit. A naive U-Net that upsamples the
    whole middle output instead (x = U mid_out) would give U P h != h even
    here, because pooling throws away fine detail in the stream itself.

  * Any W1..W5: subtracting the two lines above, the only error is

        U P (h @ (M - I)) - h @ (M - I)  =  (U P - I)(h @ (M - I))

    i.e. the pool -> upsample roundtrip applied to the UPDATE alone. U P - I
    is ~0 on smooth signals and only bites on fine detail, so the error is
    small when the middle's update is smooth (the regime make_inputs builds).

In a real model P and U can be learnable depthwise convs initialised to
exactly avg-pool and bilinear, so fine-tuning can sharpen P, U and the middle
weights while the skip keeps the starting point exact.

(Real blocks are nonlinear, e.g. x + MLP(x) or x + Attn(x). The same
delta = mid_out - mid_in bookkeeping applies, so exactness at an identity
middle still holds; the error just has no closed form like the one above.
See README.md for the general version.)

COST
----
Matmul work is proportional to token count, so each middle layer costs 1/4.
Seven layers with five compressed: (2 + 5/4)/7 = 0.46x. In general, with F
full-res layers and K compressed ones: (F + K/4)/(F + K); e.g. 12 layers
with the middle 8 compressed: (4 + 8/4)/12 = 0.50x. With attention the middle
saves even more: N^2 -> (N/4)^2 is ~16x cheaper.
"""

import torch
import torch.nn.functional as F
from torch import nn

START, MIDDLE, END = 1, 5, 1
LAYERS = START + MIDDLE + END      # 7
MID = range(START, START + MIDDLE)  # L1..L5: the compressed layers


class Stack(nn.Module):
    """LAYERS layers, each x + x @ W. That is the entire model.

    scale keeps each layer's update small next to the stream, as in a trained
    residual net. At scale=1 the five middle updates compound to ~4x the
    stream, so nearly everything is "update" and the skip has little to save.
    """

    def __init__(self, dim, layers=LAYERS, scale=0.3, seed=0):
        super().__init__()
        gen = torch.Generator().manual_seed(seed)
        self.W = nn.ParameterList(
            nn.Parameter(scale * torch.randn(dim, dim, generator=gen)
                         / dim ** 0.5)
            for _ in range(layers))

    def forward(self, x):
        return run(self, x, range(len(self.W)))


def run(stack, x, layers):
    """Apply the given layers in order: x <- x + x @ W_i."""
    for i in layers:
        x = x + x @ stack.W[i]
    return x


def to_grid(x, h, w):
    return x.transpose(1, 2).reshape(x.shape[0], x.shape[2], h, w)


def to_tokens(grid):
    return grid.flatten(2).transpose(1, 2)


def pool(x, h, w):
    """P: 2x2 average pool, N -> N/4 tokens."""
    return to_tokens(F.avg_pool2d(to_grid(x, h, w), 2))


def up(x, h, w):
    """U: bilinear x2 on an h x w grid -> 2h x 2w tokens."""
    return to_tokens(F.interpolate(to_grid(x, h, w), scale_factor=2,
                                   mode="bilinear", align_corners=False))


def compressed_middle(stack, full, h, w, skip=True):
    """The rewired middle: L1..L5 on pooled tokens, only their update goes up.

    skip=True is the residual rewire (exact when W1..W5 = 0); skip=False is
    the naive U-Net that upsamples the middle's whole output, for comparison.
    """
    low = pool(full, h, w)                     # P h: N -> N/4
    mid_out = run(stack, low, MID)             # L1..L5, all on N/4 tokens
    if skip:
        delta = mid_out - low                  # u1 + ... + u5: updates only
        return full + up(delta, h // 2, w // 2)    # h + U delta
    return up(mid_out, h // 2, w // 2)         # U P h + ...: stream blurred


def compressed(stack, x, h, w, skip=True):
    """L0 at full res; L1..L5 on pooled tokens; L6 at full res."""
    full = run(stack, x, range(START))         # h: saved for the skip
    x = compressed_middle(stack, full, h, w, skip)
    return run(stack, x, range(START + MIDDLE, LAYERS))


def make_inputs(batch, h, w, dim):
    """Smooth field + a little noise: where low-res compute works well."""
    field = F.interpolate(torch.randn(batch, dim, h // 4, w // 4), size=(h, w),
                          mode="bilinear")
    return to_tokens(field + 0.1 * torch.randn_like(field))


def rel_err(a, b):
    return ((a - b).norm() / b.norm()).item()


def main():
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    dim, h, w, batch = 16, 8, 8, 2
    x = make_inputs(batch, h, w, dim)
    stack = Stack(dim)

    with torch.no_grad():
        reference = stack(x)

        # 1. Identity middle: W1..W5 = 0 -> delta = 0 -> the rewire is exact.
        hollow = Stack(dim)
        hollow.load_state_dict(stack.state_dict())
        for i in MID:
            hollow.W[i].zero_()
        torch.testing.assert_close(compressed(hollow, x, h, w), hollow(x))
        no_skip = rel_err(compressed(hollow, x, h, w, skip=False), hollow(x))
        print(f"PASS identity middle (W1..W5=0): compressed == original "
              f"bit-exact; without the skip it is off by {no_skip:.3f}")

        # 2. For ANY W1..W5 the rewired middle is exactly h + U P (update),
        #    where update = what L1..L5 add at full res. The stream h is
        #    untouched; only the update goes through the lossy pool/up.
        full = run(stack, x, range(START))
        mid_original = run(stack, full, MID)
        update = mid_original - full               # h @ (M - I)
        mid_compressed = compressed_middle(stack, full, h, w)
        expected = full + up(pool(update, h, w), h // 2, w // 2)
        torch.testing.assert_close(mid_compressed, expected)
        err_mid = rel_err(mid_compressed, mid_original)
        err_out = rel_err(compressed(stack, x, h, w), reference)
        err_naive = rel_err(compressed(stack, x, h, w, skip=False), reference)
        print(f"PASS random W1..W5: error == (UP - I) applied to the "
              f"update only ({err_mid:.4f} after the middle, {err_out:.4f} "
              f"after L6); pooling the whole stream instead: {err_naive:.4f}")

        n = h * w
        cost = ((START + END) * n + MIDDLE * n / 4) / (LAYERS * n)
        print(f"\ntoken cost vs original: {cost:.2f}x "
              f"(({START + END} full + {MIDDLE} quarter-res layers)/{LAYERS}; "
              f"attention on the middle layers ~16x cheaper)")


if __name__ == "__main__":
    main()
