# 任务 1.3：修复 CPU BFGS 零步长 NaN 实施文档

## 1. 背景与目标

在原工程中，CPU 线程池版优化器 `_BFGSCpu`（`src/batchase/relaxation/optimizers/bfgs.py`）存在两处稳定性与正确性缺陷：
1. **零步长导致 NaN 污染**：当 Batch 中包含已收敛槽位、未更新槽位或位移极小的槽位时，`prepare_step` 不对这些槽位计算受力位移，导致其步长 `steplengths` 为 0。在 `determine_step` 中，原始计算式 `scale = longest_steps.reciprocal() * torch.min(longest_steps, maxstep)` 会出现 `longest_steps.reciprocal() = inf`，进而在与 `0.0` 相乘时产生 **`inf * 0.0 -> NaN`**。随后 `dpos *= scale` 将槽位位移污染为 `NaN`，并在后续更新中造成整个批次的势能计算失败。
2. **初始化缺失 `f_upper_limit` 属性**：在三态重构后，收敛判定方法统一依赖优化器的 `f_upper_limit` 属性。但 `_BFGSCpu.__init__` 未接收和存储该参数，导致调用 `run()` 时直接抛出 `AttributeError: '_BFGSCpu' object has no attribute 'f_upper_limit'`。

本任务的目标是对齐 GPU `BFGS.determine_step` 的掩码保护逻辑，彻底消除零步长 NaN 隐患，并规范初始化 `f_upper_limit` 属性。

---

## 2. 核心架构设计与修改细节

### 2.1 初始化补齐 `f_upper_limit`

在 `_BFGSCpu.__init__` 中新增入参并在实例上绑定属性：
```python
def __init__(
    self,
    optimizable_batch: OptimizableBatch,
    maxstep: float = 0.2,
    alpha: float = 70.0,
    early_stop: bool = False,
    bfgs_cpu_thread: int = 16,
    f_upper_limit: float = 100.0,
    **kwargs,
) -> None:
    ...
    self.f_upper_limit = f_upper_limit
```
使得无论是直接调用 `_BFGSCpu.run()` 还是通过 Worker 调度，都能正确将大力阈值透传给 `optimizable.converged()` 进行三态隔离判定。

### 2.2 有效步长掩码保护 (`determine_step`)

将 `_BFGSCpu.determine_step` 与 GPU `BFGS.determine_step` 的保护机制完全对齐：
```python
longest_steps = longest_steps[self.optimizable.batch_indices]
maxstep = longest_steps.new_tensor(self.maxstep)
safe_steps = torch.where(
    longest_steps > 1e-12,
    longest_steps,
    torch.ones_like(longest_steps),
)
scale = torch.where(
    longest_steps > 1e-12,
    safe_steps.reciprocal() * torch.min(longest_steps, maxstep),
    torch.zeros_like(longest_steps),
)
dpos *= scale.unsqueeze(1)
return dpos
```
- 对于有效步长（`longest_steps > 1e-12`）：使用 `safe_steps` 正常计算等比缩放因子；
- 对于零步长或非活跃槽位（`longest_steps <= 1e-12`）：`scale` 精确设为 `0.0`，完全杜绝了 `reciprocal()` 产生 `inf` 以及 `inf * 0.0 -> NaN` 的数值退化。

---

## 3. 测试覆盖与验证

1. **单元测试 (`tests/test_cpu_bfgs.py`)**：
   - `test_mixed_inactive_and_active_slots_remain_finite`：精确构造 `update_mask=[False, True]` 的混合批次，验证位移张量有限且零位移槽位无 NaN。
   - `test_all_zero_steps_remain_zero`：验证所有槽位位移全为零时的极限边界。
   - `test_step_limit_preserves_small_and_scales_large_groups`：验证小步长保真和大步长截断的数值正确性。
   - `test_threaded_run_uses_default_force_limit`：验证 `bfgs_cpu_thread=2` 多线程池模式下 5 步真实二次型迭代平稳收敛。
   - `test_cpu_and_gpu_step_limiting_match`：验证 CPU 与 GPU 路径在步长缩放算子上的张量级严格一致性。
   - 结果：**5/5 全部通过 (0.54s)**。

2. **全量测试套件回归**：
   - 包含任务 1.1（三态与隔离）、任务 1.2（LBFGS 原生算子）、任务 1.3（CPU BFGS）及 Worker 端到端管道测试：
   - 运行：`python -m unittest discover -s tests -p "test_*.py"`
   - 结果：**15/15 全部通过，零失败零报错 (3.19s)**。

3. **代码规范与编译自检**：
   - `python -m compileall src/ tests/`：100% 编译通过。
   - `git diff --check`：完全干净，无格式问题。
