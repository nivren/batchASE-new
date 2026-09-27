# Task 2.8：输出原子残余力与残余应力

## 目标

在保留综合增广 `fmax` 和既有收敛判据的前提下，分别输出：

- 真实原子自由度的最大残余力，单位 `eV/Å`；
- 晶胞自由度对应的最大有效残余应力分量，单位 `eV/Å³` 和 `GPa`。

该任务对应 `docs/debug_opt_report.md` 中的 P1 可观测性项，不改变优化器的步进逻辑或 `converged()` 判定。

## 指标语义

### 原子残余力

`get_max_atom_forces()` 只统计真实原子力：

\[
f_{\mathrm{atom},i}=\max_j\lVert F_{i,j}\rVert_2
\]

晶胞滤波器返回的增广晶胞力不会参与该统计。接口支持：

- 传入纯原子力或增广力；
- `apply_constraint=True` 时将固定原子力置零；
- `numpy=True` 时返回 `numpy.ndarray`。

### 残余应力

晶胞滤波器的应力指标来自最近一次有效 `get_forces()` 计算得到的有效应力，已经包含：

- 外加压力偏置 `σ + P·I`；
- 形变梯度变换；
- `hydrostatic_strain` 和 lattice `mask` 投影；
- `constant_volume` 的无迹投影。

统计量为所有 `3×3` 分量绝对值的最大值，不引入新的应力收敛阈值。

固定晶胞时：

- `compute_stress=False` 返回 `None`，表示晶胞应力未计算/不适用；
- `compute_stress=True` 且后端提供 stress 时返回模型应力最大分量。

不使用 `0.0` 伪装“未计算”的应力。

## 实施内容

### `optimizable.py`

- 新增 `get_max_atom_forces()`；
- 新增 `get_max_stresses()`；
- 通过 `self.batch.batch` 聚合原子自由度，避免误用增广 `batch_indices`；
- 修正 Fréchet cell filter 在 `constant_volume=True` 时的应力投影，使诊断应力与约束后的晶胞力一致。

### `worker.py`

Worker 在每个批次完成时先取得一次最终增广力，再从同一次评估派生：

- `fmax`：旧的综合增广力；
- `fmax_atom`：最大原子残余力；
- `fmax_stress`：最大残余应力，单位 `eV/Å³`；
- `fmax_stress_gpa`：最大残余应力，单位 `GPa`。

应力换算使用：

\[
1\ \mathrm{eV/Å^3}=160.21766208\ \mathrm{GPa}
\]

不可用指标写入 JSON `null`，DONE 日志显示为 `n/a`，避免将缺失数据误认为零残余应力。

同时修复 `_run_stage()` 中局部变量遮蔽 NumPy 模块名称的问题。

### `scheduler.py`

保留已有 `stage1_fmax` 和 `stage2_fmax`，新增：

- `stage1_fmax_atom`、`stage1_fmax_stress`、`stage1_fmax_stress_gpa`；
- `stage2_fmax_atom`、`stage2_fmax_stress`、`stage2_fmax_stress_gpa`。

旧 JSON 缺少新字段时保持 CSV 为空，不从综合 `fmax` 推断原子力或应力。

最终结构摘要新增收敛结构的：

- 原子最大残余力 `median / p95 / max`；
- 残余应力 `median / p95 / max`（GPa）。

失败结构和不可用指标不参与物理量统计。

## 测试覆盖

新增 `tests/test_residual_observability.py`，覆盖：

1. 多结构批次的原子力/晶胞应力分离；
2. 外压平衡关系 `σ=-P·I` 和 `σ=0`；
3. `mask`、`constant_volume`、UnitCell/Frechet 滤波器；
4. 非单位形变下有效应力，而非原始模型应力；
5. `numpy=True`、固定原子约束和增广力输入；
6. Worker JSON、Scheduler CSV 和最终摘要。

该任务不增加 GPU 推理、不改变优化器收敛逻辑，也不增加新的 stress tolerance。

## 验证结果

- 专项测试：`5/5` 通过；
- 全量 unittest：`59` 项通过，`7` 项因当前环境缺少可用 CUDA runtime 而跳过；
- `git diff --check`：通过。
