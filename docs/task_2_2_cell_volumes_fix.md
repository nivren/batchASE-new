# Task 2.2 — 修复负体积与退化晶胞处理

## Background & Goal

`OptimizableBatch.get_volumes()` 直接返回 `torch.linalg.det(cells)` 的裸值。
当晶胞基矢行列式为负（左手坐标系或反转轴）时，体积为负，导致：

1. **焓计算错误**：`E + P·V` 中 `V < 0` 使焓值偏低，优化器被引导向错误方向。
2. **维里项符号翻转**：`stress × volume` 的力贡献方向反转。

同时，`Worker._get_density()` 对 NaN、Inf 或接近零的体积未做保护，可能产生
无穷大或 NaN 密度值，污染输出 CSV。

本任务目标：以最小修改修正上述两个问题。

## Core Design & Changes

### 1. `get_volumes()` 取绝对值 — `optimizable.py:511`

```python
def get_volumes(self) -> torch.Tensor:
    cells = self.get_cells()
    return torch.linalg.det(cells).abs()
```

- 一行修改，保证所有下游消费者（焓、维里、密度）获得物理正体积。
- 不修改晶胞本身——反转/退化晶胞仍由 `converged()` 的 `INVALID_CELL`
  状态检测和隔离。

### 2. 密度保护 — `worker.py:102`

```python
vol = atoms.get_volume()
if not np.isfinite(vol) or vol <= 1e-6:
    return 0.0
```

- 在已有 `try/except` 保护内增加一层显式检查。
- 对 NaN、Inf、零或极小体积统一返回 `0.0`，避免除零或溢出。
- 阈值 `1e-6` Å³ 远小于任何物理晶胞，不影响正常结构。

### 设计决策

| 决策 | 理由 |
|------|------|
| 不使用数值钳位（clamp）| 掩盖真实退化，应由状态系统检测 |
| 不加热路径告警 | 避免高频日志噪声，退化晶胞已被 FailReason 记录 |
| 不做 GPU 同步检查 | 体积计算无 GPU 特异性，abs() 原生支持 |

## Tests & Verification

### 新增测试 — `tests/test_cell_volumes.py`

| 测试 | 覆盖点 |
|------|--------|
| `test_volumes_are_positive_while_invalid_cells_still_fail` | 正常、反转、退化晶胞的体积均为正；反转和退化被标记为 `INVALID_CELL` |
| `test_enthalpy_uses_absolute_volume` | `OptimizableUnitCellBatch` 焓 = E + P·\|V\| 验证 |
| `test_density_rejects_degenerate_and_nonfinite_volumes` | 正常密度计算正确；0、1e-9、NaN、Inf 均返回 0.0 |

### 回归结果

- 专项测试：3/3 通过
- 归档测试：23/23 通过
- 合计：**26/26 通过**
- `git diff --check`：通过
