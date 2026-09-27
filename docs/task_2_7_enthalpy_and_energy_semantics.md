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
  - `"energy_raw_ev"`: 纯势能 $E$（eV）
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
| `test_scheduler_csv_energy_and_enthalpy_columns` | 验证生成的汇总 CSV 包含完整的能量与焓语义列并正确赋值 |

### 回归与代码质量

- 专项测试：4/4 通过
- 历史测试：44/44 通过
- 全套测试：**48/48 通过**
- `git diff --check`：通过
