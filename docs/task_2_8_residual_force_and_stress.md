# Task 2.8 — 输出原子残余力与残余应力

## Background & Goal

`docs/debug_opt_report.md` (lines 190, 195, 312) 指出：
> **残余力/应力输出：P1 可观测性**
> “分别输出原子残余力和残余应力非常有价值。”
> 任务 2.8：输出原子残余力、残余应力。

### 原有实现问题

1. 在 `OptimizableUnitCellBatch` 与 `OptimizableFrechetCellBatch` 中，`get_forces()` 将真实原子力和变形梯度维里力拼合为增广力向量（`augmented_forces`，形状为 $[N + 3B, 3]$）。
2. `get_max_forces()` 仅在增广向量上计算最大范数，导致结构收敛或失败时仅输出单一的 `fmax` 标量，使用者无法区分是原子受力尚未收敛还是晶胞残余应力尚未松弛充分。
3. 优化结果缺少独立的物理原子残余力（$\text{eV/\AA}$）与晶胞柯西残余应力（$\text{eV/\AA}^3$ / $\text{GPa}$）指标。

## Core Design & Changes

### 1. `Optimizable` 体系解耦原子残余力与残余应力 — `optimizable.py`

在基类 [`OptimizableBatch`](file:///home/wangleping/codes/ICT-CSP/3rdparty/batchASE/src/batchase/relaxation/optimizable.py) 及派生类 [`OptimizableUnitCellBatch`](file:///home/wangleping/codes/ICT-CSP/3rdparty/batchASE/src/batchase/relaxation/optimizable.py) 和 [`OptimizableFrechetCellBatch`](file:///home/wangleping/codes/ICT-CSP/3rdparty/batchASE/src/batchase/relaxation/optimizable.py) 中增加标准可观测性接口：

```python
def get_max_atom_forces(
    self, forces: torch.Tensor | None = None, apply_constraint: bool = False
) -> torch.Tensor:
    """Get the maximum atomic force magnitude for each structure in the batch (in eV/Å).
    
    Evaluates purely the physical forces on atoms, excluding any virtual/cell DOFs.
    """

def get_max_stresses(self) -> torch.Tensor:
    """Get the maximum residual Cauchy stress component for each structure in the batch (in eV/Å^3).
    
    Residual stress takes into account external hydrostatic pressure:
        sigma_res = sigma + P * I
    and projects out any constrained or fixed strain components.
    """
```

- **原子残余力**：通过 `self.get_property("forces")` 提取纯原子物理 Cartesian 受力，以 `self.batch.batch` 索引进行 `scatter_reduce(reduce="amax")` 聚合，得到每个结构的 $\max_{j} \|F_{i, j}\|_2$（eV/Å）。
- **残余应力**：
  - 固定晶胞（`OptimizableBatch`）：晶胞自由度不参与松弛，残余应力默认为 0（若开启 `compute_stress=True` 则计算柯西应力最大绝对值分量）；
  - 晶胞滤波（`OptimizableUnitCellBatch` / `OptimizableFrechetCellBatch`）：考虑外加压强偏置 $\sigma_{\text{res}} = \sigma + P \cdot I$，并正确投影 `hydrostatic_strain`、`mask` 及 `constant_volume` 约束分量，计算 $\max_{k, l} |\sigma_{\text{res}, k, l}|$（eV/Å³）。

### 2. `Worker._run_stage` 输出独立残余指标 — `worker.py:365-420, 500-525`

- 在每个批次完成时提取 `max_atom_forces` 与 `max_stresses`。
- 计算 `fmax_stress_gpa`（换算比例 $1\text{ eV/\AA}^3 \approx 160.21766208\text{ GPa}$）。
- 在结构 JSON 结果字典中新增：
  - `"fmax_atom"`: 最大原子残余力（$\text{eV/\AA}$）
  - `"fmax_stress"`: 最大残余应力分量（$\text{eV/\AA}^3$）
  - `"fmax_stress_gpa"`: 最大残余应力（$\text{GPa}$）
  - 保留 `"fmax"`: 综合增广残余力（向下兼容）
- 在 DONE 日志中丰富输出信息：
  `fmax=0.0098 (atom=0.0098, stress=0.015GPa)`

### 3. `Scheduler._write_summary_csv` 增补残余力与应力列 — `scheduler.py:285-335`

在 `results_scheduler.csv` 中增补以下列：
- `stage1_fmax`
- `stage1_fmax_atom`
- `stage1_fmax_stress`
- `stage2_fmax`
- `stage2_fmax_atom`
- `stage2_fmax_stress`

## Tests & Verification

### 新增测试 — `tests/test_residual_observability.py`

| 测试方法 | 覆盖点 |
|---|---|
| `test_optimizable_atom_and_stress_separation` | 验证多体系批次中原子残余力与残余应力的精确数值分离、固定晶胞与可变晶胞行为 |
| `test_unit_cell_batch_external_pressure_stress` | 验证加压滤波下目标平衡应力偏置 $\sigma + P \cdot I$ 的准确性 |
| `test_frechet_cell_batch_residual_stress` | 验证 Fréchet 晶胞滤波器的原子残余力与应力分量提取 |
| `test_worker_and_scheduler_output_residual_metrics` | 验证 Worker 输出 JSON 及 Scheduler 生成的 `results_scheduler.csv` 完整包含原子受力与应力列 |

### 全量回归测试

- 专项测试 4/4 通过。
- 全量测试 **52/52** 通过（含 GPU MACE 真实模型测试）。
- `git diff --check` 通过。
