# Task 2.6 — 校验 `MOLECULE_SINGLE` 与分子数

## Background & Goal

`docs/debug_opt_report.md` (lines 204, 226-234, 310) 指出：
> **D1 `MOLECULE_SINGLE`**：示例值 13/64 不应随意硬编码，因为不同晶体体系每分子原子数不同。
> 推荐修复：
> - 要求 `MOLECULE_SINGLE` 显式提供（未提供时默认为 `None`，不盲目归一化）
> - 校验 `natoms % molecule_single == 0`
> - 输出 `natoms` 与 `num_molecules`
> - 非整数时明确标记不可归一化并发出警告，防止除出非整数分子数产生虚假归一化能量

### 原有实现问题

在 `src/batchase/engine/worker.py:382-387` 中：
```python
natoms = len(optimized_atoms[idx])
num_mol = natoms / self.molecule_single if self.molecule_single > 0 else 1.0
...
energy_per_mol = (e_val / num_mol) * 96.485 if num_mol > 0 else e_val
```
- 若 `natoms = 46`，而 CLI 默认 `molecule_single = 64`，直接浮点除法得到 `num_mol = 0.71875`，导致能量除以 `0.71875` 产生荒谬的每摩尔能量。
- JSON 和 CSV 中从未记录 `natoms` 和 `num_molecules`，下游无法核验归一化分母是否正确。

## Core Design & Changes

### 1. 取消硬编码默认值

- **`scripts/batch_relax.py:50`**：`--molecule_single` 默认值从 `64` 改为 `None`（`type=int, default=None`），提示说明需要显式提供用于每分子能量归一化。
- **`scheduler.py:75, 97`**：`Scheduler.__init__` 参数默认值改为 `molecule_single: Optional[int] = None`，安全转换为正整数或 `None`。
- **`worker.py:50, 82`**：`Worker.__init__` 参数默认值改为 `molecule_single: Optional[int] = None`。
- **`slot_manager.py:48, 54`**：`SlotManager.__init__` 参数默认值改为 `None`。

### 2. 严格整除校验与防护 — `worker.py:381-396`

```python
natoms = len(optimized_atoms[idx])
num_mol = None
energy_per_mol = None
normalization_status = "unnormalized"

if self.molecule_single is not None and self.molecule_single > 0:
    if natoms % self.molecule_single == 0:
        num_mol = natoms // self.molecule_single
        normalization_status = "normalized"
    else:
        normalization_status = "invalid_atom_count"
        logger.warning(
            f"{self.worker_tag} [{stem}] natoms ({natoms}) is not divisible by molecule_single "
            f"({self.molecule_single}). Cannot compute per-molecule energy."
        )

if num_mol is not None and num_mol > 0:
    energy_per_mol = (e_val / num_mol) * 96.485
    energy_out = energy_per_mol
else:
    energy_out = e_val * 96.485
```

- 若整除：计算整数 `num_molecules`，标记 `normalization_status = "normalized"` 并计算每摩尔能量；
- 若不整除：标记 `normalization_status = "invalid_atom_count"`，不执行每分子归一化并输出警告；
- 若未提供：标记 `normalization_status = "unnormalized"`。

### 3. JSON 与 CSV 增强字段输出

- **JSON 输出**：增补 `"natoms"`, `"molecule_single"`, `"num_molecules"`, `"normalization_status"`, `"energy_raw_ev"`, `"energy_per_mol"` 等字段。
- **CSV 输出**（`results_scheduler.csv`）：新增 `"natoms"`, `"num_molecules"`, `"normalization_status"` 列。

## Tests & Verification

### 新增测试 — `tests/test_molecule_single_validation.py`

| 测试方法 | 覆盖点 |
|---|---|
| `test_worker_init_defaults_to_none` | 验证 Worker 与 Scheduler 默认 `molecule_single` 为 `None` |
| `test_normalization_exact_multiple` | 验证当 `natoms % molecule_single == 0` 时，得到正确整数分子数与 `"normalized"` 状态 |
| `test_normalization_not_divisible` | 验证非整除时标记 `"invalid_atom_count"`，跳过每分子能量归一化并保持 `energy_per_mol is None` |
| `test_normalization_when_none` | 验证未传参数时标记 `"unnormalized"` 状态 |
| `test_csv_summary_includes_molecule_fields` | 验证生成的 `results_scheduler.csv` 包含 `natoms`、`num_molecules` 和 `normalization_status` 列 |

### 回归与代码质量

- 专项测试：5/5 通过
- 历史测试：39/39 通过
- 全套测试：**44/44 通过**
- `git diff --check`：通过
