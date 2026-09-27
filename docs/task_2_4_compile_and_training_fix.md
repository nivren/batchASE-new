# Task 2.4 — 修复 compile 参数传递及 `training=False`

## Background & Goal

`docs/debug_opt_report.md` (lines 203, 213-225) 记录了与 `compile_mode` 和 `training` 相关的两个关键问题：

1. **`training=self.use_compile` 语义错误**：
   在 `MACEBatchBackend._forward()` 中，前向调用原为 `training=self.use_compile`。若启用编译，此标志会将 MACE 内部的 `training` 设为 `True`，导致底层 `get_outputs` 开启 `create_graph=True` 并保留整个 autograd 计算图，在几何弛豫推断中造成巨大且无用的显存开销与计算冗余。弛豫前向推断必须始终为 `training=False`。
2. **`compile_mode` 参数传递链路断裂**：
   CLI (`batch_relax.py`) 提供了 `--compile_mode` 参数并传递给 `Scheduler`，但 `Scheduler` 未将此参数完整传至 `Worker`，`Worker` 亦未在实例化 `create_backend()` 时传入，导致 MACE 底层 `compile_mode` 无法生效。

本任务目标：将 MACE 前向推断硬编码锁定为 `training=False`，并贯通 `Scheduler` → `Worker` → `create_backend` → `MACEBatchBackend` 的参数透传链路，同时提供编译失败时的安全回退。

## Core Design & Changes

### 1. 彻底锁定 `training=False` — `mace.py:226`

在 `MACEBatchBackend._forward` 中硬编码：

```python
out = self.model(
    inputs,
    compute_stress=compute_stress,
    training=False,
)
```

无论是否启用 `torch.compile`，均以无图（`training=False`）模式运行前向，所有输出张量保持 detached 状态。

### 2. `MACEBatchBackend` 接收 `compile_mode` 与安全回退 — `mace.py:64, 84-110`

- `MACEBatchBackend.__init__` 新增 `compile_mode: Optional[str] = None` 参数。
- `effective_compile_mode` 统一解析 `compile_mode` 与 `use_compile`。
- 在 `calculator = mace_off(..., compile_mode=effective_compile_mode)` 处加入异常捕获机制：若在不支持 `torch.compile` 的环境或驱动下初始化失败，输出清晰警告日志并自动降级为未编译模型（`compile_mode=None`），避免直接崩溃。

### 3. `Scheduler` 与 `Worker` 管道贯通

- **`scheduler.py:77, 98, 198`**：`Scheduler.__init__` 显式接受 `compile_mode` 并存储在 `self.compile_mode`，在 `run()` 中组装 `worker_kwargs` 时将 `"compile_mode": self.compile_mode` 传入每个工作进程。
- **`worker.py:57, 87, 490`**：`Worker.__init__` 显式声明 `compile_mode`，并在 `Worker.run()` 中调用 `create_backend(..., compile_mode=self.compile_mode)`。

## Tests & Verification

### 新增测试 — `tests/test_compile_semantics.py`

| 测试方法 | 覆盖点 |
|---|---|
| `test_forward_always_calls_training_false_and_detaches` | 验证无论 `use_compile` / `compile_mode` 状态，底层模型接收的 `training` 始终为 `False`，且输出能量、力和应力均为 detached（`grad_fn is None`） |
| `test_compile_mode_propagation_scheduler_to_worker` | 验证 `compile_mode` 在 `Scheduler` 与 `Worker` 的初始化与参数传递中完整保留 |
| `test_compile_fallback_on_failure` | 验证当底层的 `mace_off` 针对指定 `compile_mode` 编译失败时，能安全优雅地降级至无编译模式并输出警告日志 |

### 回归与代码质量

- 专项测试：3/3 通过
- 历史测试：32/32 通过
- 全套测试：**35/35 通过**
- `git diff --check`：通过
