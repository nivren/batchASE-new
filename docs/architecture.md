# batchASE 架构规格说明书

本文档定义 `batchASE` 各核心模块的职责分工与调用契约。

## 1. 模块层次

```
batchase/
├── neighbors/     # 邻居搜索与周期性图构建 (AtomsToGraphs)
├── kernels/       # 硬件加速核心算子 (CUDA, HIP, Triton, TileLang, CPU)
├── potentials/    # 势函数推理后端 (MACE, SevenNet, CHGNet, MatRIS)
├── relaxation/    # 松弛状态维护 (OptimizableBatch) 与优化器 (FIRE2, BFGS, LBFGS)
└── engine/        # 动态批管理 (SlotManager), 工作进程 (Worker), 顶层调度 (Scheduler)
```

## 2. 动态批槽位契约 (`SlotManager` <-> `Optimizer`)

为了解决不同晶体收敛步数差异带来的 GPU 算力闲置问题，`SlotManager` 负责在迭代过程中替换已收敛或步数超限的晶体结构。

`Optimizer` 需实现标准更新契约：
```python
def update_slots(self, keep_indices: list[int], num_new_slots: int) -> None:
    """当批处理槽位发生变动时同步更新内部状态。

    Args:
        keep_indices: 保留在当前批次中的旧结构索引列表
        num_new_slots: 新移入槽位的结构数量
    """
```

- 对于 **FIRE2 / 速度 Verlet 类**：
  保持存活槽位的速度 `v = v[keep_indices]`，新槽位补 0；时间步长重置。
- 对于 **BFGS 类**：
  保持存活槽位的逆 Hessian 逼近矩阵 `H = H[keep_indices]`，新槽位初始化为单位矩阵。

## 3. 硬件算子分发契约 (`kernels/registry.py`)

统一提供：
```python
def get_pbc_graph_kernel(backend: str = "auto") -> Callable:
    """根据硬件环境或显式指定返回最优 PBC 构图核函数。"""
```
支持的后端：
- `cuda`: 原生 NVIDIA CUDA C++ 扩展 (最快)
- `hip`: 原生 AMD ROCm / 海光 DCU HIP 扩展
- `triton`: OpenAI Triton 统一跨平台实现
- `tilelang`: TVM TileLang DSL 实现
- `cpu`: 纯 PyTorch CPU 兜底实现
