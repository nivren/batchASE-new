# Task 2.3 — FIRE/FIRE2 与 ASE 语义对齐

## Background & Goal

`docs/debug_opt_report.md` 指出当前 FIRE 混入了 FIRE2 的下限规则。经深入对比 ASE 原生实现 (`ase.optimize.fire` 与 `ase.optimize.fire2`)，确认了两处核心差异与隐患：

1. **FIRE 1.0 错误引入 `dtmin` 截断**：
   - ASE 原生 FIRE 1.0 在负功率重置时直接执行 `dt *= fdec`，没有 `dtmin` 下限限制；
   - batchASE 原实现强制套用了 FIRE2 的 `torch.maximum(..., dt_min_t)`（默认 `2e-3`），导致在振荡不稳定的结构中时间步无法充分衰减冻结。
2. **FIRE 路径中的 `torch_scatter` 残留依赖与分支分歧**：
   - `lbfgs.py` 已完全迁移至 PyTorch 原生 `scatter_add_` / `scatter_reduce`；
   - `fire.py` 中仍保留了 `try: from torch_scatter import scatter` 并在 CUDA 下调用外部扩展，引入不必要的第三方二进制扩展和舍入分歧。

本任务目标：以最小范围修正上述两项问题，严格对齐 FIRE/FIRE2 的核心 ASE 语义，并移除 FIRE 模块中的 `torch_scatter` 依赖。其他优化器仍可保留兼容性的可选依赖。

## Core Design & Changes

### 1. FIRE 1.0 负功率移除 `dtmin` 截断 — `fire.py:219`

```python
# ASE native FIRE has no dtmin lower bound; only clamp when
# dtmin > 0 (i.e. when used via FIRE2 flavor).
new_dt = self.dt[neg_mask] * self.fdec
if self.dtmin > 0:
    new_dt = torch.maximum(new_dt, dt_min_t)
self.dt[neg_mask] = new_dt
```

并在 `FIRE.__init__` (line 362) 中将默认 `dtmin` 设置为 `0.0`（FIRE2 保留 `2e-3`），严格复现 ASE native 的无下界时间步衰减。

### 2. 清理 `torch_scatter` 残留 — `fire.py:34`

删除 `torch_scatter` 导入防护块，`_batched_dot_per_system` 统一直接使用 PyTorch 原生 `scatter_add_`：

```python
prod = (x * y).sum(dim=-1)  # [N]
return torch.zeros(
    (num_systems,), device=prod.device, dtype=prod.dtype
).scatter_add_(0, batch_indices, prod)
```

消除外部依赖并保证 CPU / CUDA 跨设备计算逻辑与浮点顺序的一致性。

### 设计与范围决策

| 项目 | 处理方式 | 说明 |
|---|---|---|
| Step clipping (`maxstep`) | 保持现实现 | ASE FIRE/FIRE2 与 batchASE FIRE 均按系统 $3N$ 模长裁切，语义一致。 |
| `downhill_check` / `use_abc` | 暂不引入 | 归入 P3 功能，不属于核心正确性修复。 |
| `force_reeval` 机制 | 保持现实现 | FIRE2 默认 `force_reeval=True` 与 ASE 对齐，并在无回溯发生时保留跳过评估的 GPU 优化。 |

## Tests & Verification

### 新增测试 — `tests/test_fire_semantics.py`

| 测试方法 | 覆盖点 |
|---|---|
| `test_fire_dt_no_lower_bound` | 验证 FIRE 1.0 `dtmin=0.0`，经多次衰减可自由降至任意小 |
| `test_fire2_dt_has_lower_bound` | 验证 FIRE2 保持 `dtmin=2e-3` 下界截断保护 |
| `test_fire_step_respects_no_dtmin` | 验证完整 `step()` 循环在负功率下衰减突破 `2e-3` 限制 |
| `test_no_torch_scatter_in_fire_module` | 源码级断言 `fire.py` 不再引用 `torch_scatter` |
| `test_batched_dot_works_without_scatter` | 原生 `scatter_add_` 点积正确性验证 |
| `test_trajectories_differ` | 验证在非共线速度与受力下，FIRE 与 FIRE2 的不同积分与混速时机产生预期分歧轨迹 |

### 回归与代码质量

- 专项测试：6/6 通过
- 历史测试：26/26 通过
- 全套测试：**32/32 通过**
- `git diff --check`：通过
