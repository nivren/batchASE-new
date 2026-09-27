# 任务 2.1：修复 constant_volume 的 reshape 维度与标量退化错误实施文档

## 1. 背景与目标

在原工程中，当用户启用等体积晶胞弛豫（`constant_volume=True`）时，两处晶胞力计算存在确凿的张量维度与标量退化缺陷：
1. **`OptimizableUnitCellBatch.get_forces()`**：
   原实现通过 `virial[:, range(3), range(3)] -= (self._batch_trace(virial).view(3, -1) / 3.0)` 试图消除晶胞力中的流体静压成分（使迹归零）。但 `self._batch_trace(virial)` 形状为 `(B,)`，在批大小 $B \neq 3$（如 $B=1, 2, 4, 16, 25$）时直接抛出 `RuntimeError: shape '[3, -1]' is invalid for input of size B`，导致等体积弛豫完全不可用。
2. **`OptimizableFrechetCellBatch.get_forces()`**：
   原实现使用 `dglf_trace = self._batch_trace(deform_grad_log_force).view(-1, 1, 1)` 后通过 `self._batch_diag(dglf_trace.squeeze() / 3.0)` 计算对角阵。当单批次 $B=1$ 时，`dglf_trace.squeeze()` 会把唯一的维度也挤压掉，退化为 0-D 标量，使底层的 `torch.vmap` 抛出 `RuntimeError: vmap: Got in_dim=0 for some input, but that input is a Tensor of dimensionality 0`。

本任务的目标是彻底解决 `constant_volume=True` 在各类批次大小下的形状与维度错误，确保等体积晶胞弛豫严格保持代数无迹约束（$\text{Tr} = 0$）。

---

## 2. 核心架构设计与修改细节

### 2.1 原位对角投影算子 (`src/batchase/relaxation/optimizable.py`)

在 `OptimizableUnitCellBatch` 和 `OptimizableFrechetCellBatch` 中采用 PyTorch 原生高效的原位对角投影：

1. **`OptimizableUnitCellBatch`**（第 900-902 行）：
   ```python
   if self.constant_volume:
       diagonal = virial.diagonal(dim1=-2, dim2=-1)
       diagonal -= diagonal.sum(dim=-1, keepdim=True) / 3.0
   ```
2. **`OptimizableFrechetCellBatch`**（第 1233-1235 行）：
   ```python
   if self.constant_volume:
       diagonal = deform_grad_log_force.diagonal(dim1=-2, dim2=-1)
       diagonal -= diagonal.sum(dim=-1, keepdim=True) / 3.0
   ```

### 2.2 方案优势
- **普适支持所有批大小**：`diagonal` 形状恒为 `(B, 3)`，`sum(dim=-1, keepdim=True)` 形状恒为 `(B, 1)`，广播机制完美兼容 $B=1, 2, 3, 4, 25$ 及任意批大小。
- **杜绝标量退化**：彻底移除了 `vmap` 调用和 `squeeze()` 操作，单批次 $B=1$ 稳健运行。
- **零内存分配与精度保真**：直接对张量对角视图在原位执行减法，无需构造额外的单位张量或临时对象，严格保持张量的数据类型（如 float64），杜绝精度截断损失。

---

## 3. 测试覆盖与验证

1. **单元测试 (`tests/test_constant_volume.py`)**：
   - 构造无 GPU 依赖的轻量测试探针 `StressMockBackend`，注入具有非零迹的应力张量并按槽位线性放缩。
   - 覆盖批大小 $B = 1, 2, 3, 4, 25$。
   - 分别验证 `OptimizableUnitCellBatch` 与 `OptimizableFrechetCellBatch` 在 `constant_volume=True` 下：
     - 计算得到的晶胞力有限；
     - 晶胞力对角线元素之和（迹）相对误差 $< 10^{-14}$，数学上严格无迹 $\text{Tr}(\sigma) = 0$；
     - 归一化后各槽位的独立受力严格吻合。
   - 结果：**2/2 全部通过 (0.07s)**。

2. **全量测试套件历史回归**：
   - 包含第一阶段所有任务（1.1 三态与隔离、1.2 LBFGS 原生算子、1.3 CPU BFGS 掩码、1.4 cuSOLVER 恢复、Worker 端到端测试）及任务 2.1 专项单测：
   - 运行：`python -m unittest discover -s tests -p "test_*.py"`
   - 结果：**23/23 全部通过，零失败零报错 (3.25s)**。

3. **静态检查与编译自检**：
   - `python -m compileall src/ tests/`：100% 编译通过。
   - `git diff --check`：输出干净，0 格式违规。
