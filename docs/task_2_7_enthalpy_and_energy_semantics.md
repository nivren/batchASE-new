# Task 2.7 — 区分 `enthalpy_kj_mol` 与 `energy_kj_mol`

## Background & Goal

`docs/debug_opt_report.md` (lines 209, 311) 指出：
> **C1 能量语义**：Stage 1 含 $PV$ 功，实质是焓 ($H = E + PV$)；Stage 2 压力为零时是纯势能 ($E$)。当前统一叫 `energy` 极易误导使用者直接进行无意义的跨阶段对比。更新优先级：**P1**。
> 任务 2.7：区分 `enthalpy_kj_mol` 和 `energy_kj_mol`。

### 原有实现问题

1. `OptimizableUnitCellBatch.get_potential_energies()` 返回 $E + P \cdot V$（焓），但基类和外层直接作为 `energy` 消费。
2. `Worker` 输出的 `result_data["energy"]` 在加压下为焓、常压下为势能，未给出清晰分离。
3. `results_scheduler.csv` 仅有 `stage1_energy` 与 `stage2_energy` 列，掩盖了 Stage 1 包含 $PV$ 功的本质。

## Core Design & Changes

### 1. `Optimizable` 体系解耦能量分量 — `optimizable.py`

在基类 [`OptimizableBatch`](file:///home/wangleping/codes/ICT-CSP/3rdparty/batchASE/src/batchase/relaxation/optimizable.py) 及派生类 [`OptimizableUnitCellBatch`](file:///home/wangleping/codes/ICT-CSP/3rdparty/batchASE/src/batchase/relaxation/optimizable.py) 和 [`OptimizableFrechetCellBatch`](file:///home/wangleping/codes/ICT-CSP/3rdparty/batchASE/src/batchase/relaxation/optimizable.py) 中定义清晰的标准解耦接口：

```python
def get_internal_energies(self) -> torch.Tensor:
    """Get internal potential energy E for each system in batch (excluding PV term)."""

def get_pv_terms(self) -> torch.Tensor:
    """Get PV work term for each system in batch (zeros when no external pressure)."""

def get_enthalpies(self) -> torch.Tensor:
    """Get enthalpy H = E + PV for each system in batch."""
```

- 在无外压/固定晶胞下：$PV = 0$，$H = E$；
- 在加压晶胞滤波下：严格满足 $H = E + P \cdot V$；
- `get_potential_energies()` 维持作为优化器目标函数接口（加压时为焓，常压时为势能）。

### 2. `Worker._run_stage` 分离输出能量与焓 — `worker.py:480-495`

- 提取 `internal_energies`、`pv_terms` 与 `enthalpies`。
- 在每个结构的 JSON 结果字典中新增：
  - `"energy_raw_ev"`: 兼容字段，保留历史优化目标值（Stage 1 加压时为焓，Stage 2 常压时为势能）
  - `"internal_energy_raw_ev"`: 纯势能 $E$（eV）
  - `"enthalpy_raw_ev"`: 焓 $H = E + PV$（eV）
  - `"pv_raw_ev"`: $PV$ 功（eV）
  - `"energy_kj_mol"`: 纯势能（kJ/mol）
  - `"enthalpy_kj_mol"`: 焓（kJ/mol）
  - `"energy"`: 兼容字段（保留旧字段行为，平滑过渡）

### 3. `Scheduler._write_summary_csv` 增强语义列 — `scheduler.py:280-335`

在 `results_scheduler.csv` 中增补以下列：
- `stage1_energy_kj_mol`（Stage 1 扣除 $PV$ 后的纯势能）
- `stage1_enthalpy_kj_mol`（Stage 1 含外压功的真实焓）
- `stage2_energy_kj_mol`（Stage 2 纯势能）
- `stage2_enthalpy_kj_mol`（Stage 2 焓，常压下等于势能）

## Tests & Verification

### 新增测试 — `tests/test_enthalpy_energy_semantics.py`

| 测试方法 | 覆盖点 |
|---|---|
| `test_optimizable_batch_zero_pressure` | 验证无压固定晶胞下 $PV = 0$，$H = E$ |
| `test_unit_cell_batch_enthalpy_components` | 验证 UnitCellFilter 加压下 $E$、$P \cdot V$ 与 $H = E + PV$ 的精确数值关系 |
| `test_frechet_cell_batch_enthalpy_components` | 验证 FrechetCellFilter 下能量与焓的精确拆分 |
| `test_numpy_energy_components_batch` | 验证 `numpy=True` 的多结构批次也能返回能量、PV 和焓 |
| `test_worker_json_keeps_legacy_objective_and_exposes_internal_energy` | 验证 Worker JSON 同时保留兼容目标值并输出纯内能 |
| `test_scheduler_csv_energy_and_enthalpy_columns` | 验证生成的汇总 CSV 包含完整的能量与焓语义列并正确赋值 |
| `test_dashboard_single_stage_uses_explicit_step_units` | 验证单阶段 dashboard 显示结构级累计步数，不泄露 S2 列 |
| `test_dashboard_stage_averages_use_stage_specific_denominators` | 验证两阶段 dashboard 平均值使用阶段专属分母及收敛率与长尾统计 |

### 回归与代码质量

- 专项测试：8/8 通过
- 历史测试：46/46 通过
- 全套测试：**54/54 通过**
- `git diff --check`：通过

## Dashboard 统计口径修正

### 问题与证据

此前 dashboard 同时使用了两种不同的 `steps` 口径，但都显示为 `S1 Steps` 或 `S2 Steps`：

- `results_scheduler.csv` 中的 `stage*_steps` 是单结构累计步数，来自动态 batch 中该结构实际被计账的 `cur_batch_steps[idx]`。
- `metrics/worker_*.json` 中原有的 `stages.*.steps` 是 worker 对每个 batch burst 的 `nsteps` 求和，即 batch-level iterations，不是结构步数，也不是结构平均步数。
- 动态补槽后，batch 中的活跃结构数会变化，因此两者不应直接相等。

本次修正保留旧 JSON 的 `steps` 字段兼容性，同时在 worker metrics JSON 中新增明确的 `batch_iterations` 字段。该字段仅作为性能探针数据保留，不再放入面向用户的主 dashboard。dashboard 中：

- `Per-Structure Steps` 表示结构级累计步数；
- `Avg S1/S2 per attempted struct` 表示进入对应阶段的结构平均步数；
- worker 表格不再显示容易混淆的 batch-level 计数列；
- worker 表格增加每个阶段的最长结构步数及结构标识。

Stage 2 平均值只除以实际进入 Stage 2 的结构数，不再使用全部输入结构数作为分母。

### 单阶段 dashboard

当没有任何 worker 实际生成 `final` stage metrics 时，dashboard 不再打印 Stage 2：

- 不显示 S2 steps、时间、吞吐和组件列；
- 不显示 `Avg ... S2`；
- worker 表格只保留 Stage 1 的时间和最长结构步数；
- 最终结构摘要明确标注 `Stage 1 only`。

这与 worker 中 `skip_second_stage` 的实际执行行为保持一致；如果只有部分结构进入 Stage 2，则仍显示 Stage 2，但所有 Stage 2 平均值只针对实际尝试过该阶段的结构计算。

### 收敛率、最终受力与 worker 长尾

主 dashboard 增加以下结果质量指标：

- `S1` 收敛率：Stage 1 收敛结构数 / 实际尝试结构数；
- `S2` 收敛率：Stage 2 收敛结构数 / 实际进入 Stage 2 的结构数；
- `Final` 收敛率：最终阶段收敛结构数 / 最终阶段尝试结构数；
- 最终收敛结构的 `fmax`：median、p95、max，以及低于当前阶段收敛阈值的数量。

每个 worker 的 metrics JSON 还保存阶段长尾信息：

- `max_structure_steps`；
- `max_structure_file`；
- `max_structure_status`；
- `max_structure_failed_reason`；
- `max_structure_fmax`。

主日志表格显示为 `S1 Max (steps/id)` 和 `S2 Max (steps/id)`，并保留完整结构标识，不对结构名做截断。这能区分某个 worker 是由难收敛结构拖慢，还是由达到 `max_steps` 或异常失败造成长尾。

典型输出形式为：

```text
 Convergence Rates
  S1                   : 1000/1000 (100.0%)
  S2                   : 997/1000 (99.7%)
  Final                : 997/1000 (99.7%)
  Final fmax [eV/A]    : n=997 median=... p95=... max=... below_target=997/997 (target=...)
 Worker ... S1 Max (steps/id)       S2 Max (steps/id)
```

### 最终结构摘要

dashboard 现在使用已有的汇总记录增加 `Final Structure Summary`，不重新读取 CIF，也不引入 GPU 同步。统计对象为最终阶段收敛结构：

- 两阶段运行使用 Stage 2；
- 单阶段运行使用 Stage 1，并标记为 `Stage 1 only`；
- 失败结构不进入物理量统计，但会单独计入 failed 数量；
- 密度输出 `min / median / mean / max`，单位为 `g/cm^3`；
- 最终受力输出 `median / p95 / max`，单位为 `eV/A`；
- 能量输出 `min / median / mean / max`，单位为归一化的 `kJ/mol`；
- 同时输出相对能量 `ΔE` 的中位数、最大值以及 `ΔE <= 5 kJ/mol` 的结构数。

能量统计遵循任务 2.7 的语义：

- Stage 2 使用 `energy_kj_mol`；
- 仅 Stage 1 且外压不为零时使用 `enthalpy_kj_mol`；
- 只有在能量归一化有效且数值有限时才纳入统计；
- 缺少归一化能量字段时显示 unavailable，不从旧的无单位 `energy` 字段猜测单位。

### 新增回归覆盖

`tests/test_enthalpy_energy_semantics.py` 新增：

- 单阶段 dashboard 不出现 S2，并验证结构步数、收敛率、最终受力和 worker 长尾信息；
- 两阶段 dashboard 验证 Stage 2 平均值使用阶段专属分母，并验证 S1/S2/Final 收敛率、最终受力和能量摘要；
- Worker stage metrics 验证最长结构的步数、文件名和状态被保存。

上述汇总只在任务结束时执行一次，复杂度为 `O(N)`，不会改变弛豫、batch 补槽或 GPU 优化路径。`batch_iterations` 仍保存在 JSON 中，供性能探针使用，但不参与材料结果的主日志展示。

本次扩展后的全量 unittest 为 **54/54 通过**，其中 7 个 CUDA/端到端测试因当前测试环境未提供可用 CUDA runtime 而跳过。
