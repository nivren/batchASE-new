# pbc_distance_tilelang.py
import torch
import tilelang
import tilelang.language as T

# ---------------------------
# 1) TileLang kernel (returns two outputs)
# ---------------------------
def pbc_distance_tl_program(BLOCK: int = 512, dtype: str = "float32"):
    eps = 1e-12 if dtype == "float64" else 1e-8

    # N and B are dynamic so you can use arbitrary num_pairs / batch_size at runtime
    N = T.dynamic("N")
    B = T.dynamic("B")

    @T.prim_func
    def main(
        pos1: T.Tensor((N, 3), dtype),
        pos2: T.Tensor((N, 3), dtype),
        pbc_offsets: T.Tensor((B, 3), dtype),      # [batch_size, 3]
        batch_offsets: T.Tensor((B + 1,), "int64"),# [batch_size+1]
        radius_sq: T.Tensor((), dtype),            # scalar
        distances_sq: T.Tensor((N,), dtype),       # output
        valid_mask: T.Tensor((N,), "bool"),        # output
    ):
        with T.Kernel(T.ceildiv(N, BLOCK), threads=BLOCK) as bx:
            for tx in T.Parallel(BLOCK):
                idx = bx * BLOCK + tx
                if idx < N:
                    # Find batch index (linear scan; replace with binary search if B is large)
                    b_idx = T.alloc_local((1,), "int32")
                    b_idx[0] = 0
                    for b in T.serial(B):
                        if (idx >= batch_offsets[b]) and (idx < batch_offsets[b + 1]):
                            b_idx[0] = T.cast(b, "int32")

                    ox = pbc_offsets[b_idx[0], 0]
                    oy = pbc_offsets[b_idx[0], 1]
                    oz = pbc_offsets[b_idx[0], 2]

                    dx = pos2[idx, 0] - pos1[idx, 0] + ox
                    dy = pos2[idx, 1] - pos1[idx, 1] + oy
                    dz = pos2[idx, 2] - pos1[idx, 2] + oz

                    d2 = dx * dx + dy * dy + dz * dz
                    distances_sq[idx] = d2
                    valid_mask[idx] = (d2 <= radius_sq[()]) and (d2 > T.cast(eps, dtype))

    return main


# ---------------------------
# 2) Compile cache (compile once per dtype/BLOCK/device)
# ---------------------------
_KERNEL_CACHE = {}

def _get_kernel(dtype: torch.dtype, block: int, device: torch.device):
    if dtype not in (torch.float32, torch.float64):
        raise TypeError(f"pos dtype must be float32/float64, got {dtype}")

    key = (str(device), dtype, block)
    if key in _KERNEL_CACHE:
        return _KERNEL_CACHE[key]

    tl_dtype = "float64" if dtype == torch.float64 else "float32"
    prog = pbc_distance_tl_program(BLOCK=block, dtype=tl_dtype)

    # outputs are the last two args => out_idx=[-2, -1]
    # execution_backend can be "cython"/"torch"/"tvm_ffi"/"nvrtc" depending on your setup
    ker = tilelang.compile(
        prog,
        out_idx=[-2, -1],
        target="cuda",
        execution_backend="cython",
    )
    _KERNEL_CACHE[key] = ker
    return ker


# ---------------------------
# 3) PyTorch wrapper API
# ---------------------------
def pbc_distance_tl(pos1: torch.Tensor,
                    pos2: torch.Tensor,
                    pbc_offsets: torch.Tensor,
                    batch_offsets: torch.Tensor,
                    radius: float,
                    block: int = 512):
    """
    Args:
      pos1: [N,3] float32/float64 CUDA
      pos2: [N,3] float32/float64 CUDA
      pbc_offsets: [B,3] same dtype/device as pos*
      batch_offsets: [B+1] int64 CUDA (prefix sums; batch_offsets[0]=0)
      radius: float
    Returns:
      distances_sq: [N]
      valid_mask: [N] bool
    """
    if not pos1.is_cuda or not pos2.is_cuda:
        raise ValueError("pos1/pos2 must be CUDA tensors")
    if pos1.dtype != pos2.dtype:
        raise ValueError("pos1 and pos2 must have same dtype")
    if pbc_offsets.dtype != pos1.dtype or pbc_offsets.device != pos1.device:
        raise ValueError("pbc_offsets must match pos dtype/device")
    if batch_offsets.dtype != torch.int64 or batch_offsets.device != pos1.device:
        raise ValueError("batch_offsets must be int64 on same CUDA device")

    pos1 = pos1.contiguous()
    pos2 = pos2.contiguous()
    pbc_offsets = pbc_offsets.contiguous()
    batch_offsets = batch_offsets.contiguous()

    radius_sq = torch.tensor(radius * radius, device=pos1.device, dtype=pos1.dtype)

    ker = _get_kernel(pos1.dtype, block, pos1.device)

    # Because out_idx=[-2,-1], TileLang allocates distances_sq and valid_mask for you
    distances_sq, valid_mask = ker(pos1, pos2, pbc_offsets, batch_offsets, radius_sq)
    return distances_sq, valid_mask


# ---------------------------
# 4) Optional: autograd stub (forward-only)
# ---------------------------
class PBCDistanceTL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pos1, pos2, pbc_offsets, batch_offsets, radius: float, block: int = 512):
        # no backward (same as your CUDA code; add backward if you need grads)
        return pbc_distance_tl(pos1, pos2, pbc_offsets, batch_offsets, float(radius), block)

# convenience alias
pbc_distance = PBCDistanceTL.apply
