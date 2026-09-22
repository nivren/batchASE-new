
import tilelang
import tilelang.language as T
from tilelang.language.atomic import atomic_add

import tilelang.jit as tl_jit

import torch

# Scatter-add along dim=0:
#   src:   [N, F]
#   index: [N]  (int32/int64)
#   out:   [M, F]
#
# out[index[i], f] += src[i, f]
@T.prim_func
def scatter_add_2d(
    src:   T.Buffer,   # (N, F), float16/float32
    index: T.Buffer,   # (N,),   int32 preferred
    out:   T.Buffer,   # (M, F), same dtype as src
):
    N = src.shape[0]
    F = src.shape[1]

    BLOCK_N = 256
    THREADS = 256

    # 1D grid over N
    with T.Kernel(T.ceildiv(N, BLOCK_N), threads=THREADS) as (bx,):
        tx = T.get_thread_binding(0)  # threadIdx.x
        i = bx * BLOCK_N + tx

        if i < N:
            dst = index[i]
            # Bounds check (optional but safer)
            if 0 <= dst and dst < out.shape[0]:
                # accumulate feature dimension
                for f in range(F):
                    atomic_add(out[dst, f], src[i, f])

# Compile once (you can also use @tilelang.jit as a decorator style)
scatter_add_kernel = tl_jit.compile(
    scatter_add_2d,
    out_idx=None,            # we pass out in, so no need to return
    target="auto",           # or "auto"
    execution_backend="auto" # cuda typically picks tvm_ffi
)

def scatter_add_tl(src: torch.Tensor, index: torch.Tensor, M: int):
    """
    src:   [N, F] on CUDA
    index: [N] on CUDA (int32 recommended)
    M:     output rows
    """
    assert src.is_cuda and index.is_cuda
    assert src.ndim == 2 and index.ndim == 1
    assert src.shape[0] == index.shape[0]

    if index.dtype != torch.int32:
        index = index.to(torch.int32)

    out = torch.zeros((M, src.shape[1]), device=src.device, dtype=src.dtype)
    scatter_add_kernel(src, index, out)
    return out
