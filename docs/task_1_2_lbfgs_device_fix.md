# 任务 1.2：修复 LBFGS 未定义 device 导致运行时异常实施文档

## 1. 背景与目标

在原工程中，批处理 LBFGS 优化器（`src/batchase/relaxation/optimizers/lbfgs.py`）以及 `BFGSFusedLS` 辅助函数中存在严重的硬编码与变量未定义错误：
1. **未定义变量导致的崩溃**：`determine_step()` 与 `_batched_dot()` 中硬编码了 `if device == 'cuda':`，但并未定义 `device` 局部或全局变量，导致调用时直接触发 `NameError: name 'device' is not defined`。
2. **外部库强依赖与环境不一致**：原实现尝试引入外部库 `torch_scatter` 的 `scatter` 函数，导致在未安装 `torch_scatter` 或在 CPU 环境下运行时容易产生分支分叉或不可用。
3. **零步长除零隐患**：原实现中步长计算公式采用 `scale = longest_steps.reciprocal() * torch.min(longest_steps, maxstep)`，在步长为 0（如已收敛结构或零位移步）时会发生 `Inf * 0 -> NaN`。
4. **Worker 传参阻断**：`Worker._run_stage()` 在构造优化器实例时，将特定于 `BFGSFusedLS` 的配置参数（如 `use_profiler`、`device` 等）无差别传递给所有优化器，导致 `Worker(optimizer1="LBFGS")` 因参数不匹配抛出 `TypeError`。

本任务的目标是彻底根除 `device` 相关的 `NameError`，采用 PyTorch 原生算子重构向量化步长控制与内积计算，移除零步长 NaN 隐患，并打通 Worker 对 LBFGS 的调用链。

---

## 2. 核心架构设计与修改细节

### 2.1 原生张量算子替代与设备解耦 (`lbfgs.py`)

移除对 `torch_scatter` 的依赖，全面采用 PyTorch 原生张量算子：
- **最大步长归约与按组截断 (`determine_step`)**：
  ```python
  steplengths = torch.norm(dr, dim=1)
  index = self.optimizable.batch_indices
  longest_steps = torch.full(
      (self.optimizable.batch_size,),
      float("-inf"),
      device=steplengths.device,
      dtype=steplengths.dtype,
  ).scatter_reduce(
      dim=0,
      index=index,
      src=steplengths,
      reduce="amax",
      include_self=True,
  )
  longest_steps = longest_steps[index]
  maxstep = longest_steps.new_tensor(self.maxstep)
  scale = torch.clamp(
      maxstep / torch.clamp(longest_steps, min=1e-12),
      max=1.0,
  )
  dr *= scale.unsqueeze(1)
  return dr * self.damping
  ```
  - 动态使用 `steplengths.device` 与 `steplengths.dtype`，自动兼容 CPU、CUDA 及多卡环境。
  - 通过 `torch.clamp(maxstep / torch.clamp(longest_steps, min=1e-12), max=1.0)` 替代原来的 `reciprocal()` 乘积，完全杜绝了除以零和 `NaN`。

- **分组批量内积计算 (`_batched_dot`)**：
  ```python
  index = self.optimizable.batch_indices
  src = (x * y).sum(dim=-1)
  out = torch.zeros(
      self.optimizable.batch_size,
      device=src.device,
      dtype=src.dtype,
  )
  return out.scatter_add_(dim=0, index=index, src=src)
  ```
  采用原生的 `scatter_add_` 就地累加，避免了复杂的设备判断分支。

### 2.2 清理死代码与潜在陷阱 (`bfgsfusedls.py`)

在 `BFGSFusedLS` 中，移除了包含相同 `device == 'cuda'` 隐患且未被调用的 `_batched_dot_2d` 与 `_batched_dot_1d` 冗余方法及多余的 `scatter` 导入，消除潜在陷阱。

### 2.3 Worker 优化器传参解耦 (`worker.py`)

在 `Worker._run_stage()` 中，对优化器配置字典 `opt_kwargs` 进行了精细化区分：
- 通用参数（`maxstep`、`early_stop`、`f_upper_limit`）传递给所有优化器；
- 专属参数（如 `use_profiler`、`profiler_log_dir`、`device` 等）仅在 `optimizer_key in ("bfgsfusedls", "bfgslinesearch")` 时附带。
由此打通了 Worker 级以 `optimizer1="LBFGS"` 或 `optimizer2="LBFGS"` 进行批量晶体弛豫的执行通路。

---

## 3. 测试覆盖与验证

1. **单元测试 (`tests/test_lbfgs.py`)**：
   - 构造无外部依赖的二次型受敛基准体系 `QuadraticOptimizable`；
   - `test_cpu_helpers_and_zero_step`：验证 CPU 下 `determine_step` 正常限制最大步长、零步长不产生 NaN、`_batched_dot` 结果符合理论值；
   - `test_cpu_run_five_steps`：验证 CPU 下 5 步 LBFGS 迭代收敛平稳，受力持续下降；
   - `test_cuda_helpers_and_zero_step`：验证 CUDA 环境下原生 `scatter_reduce` 与 `scatter_add_` 运行无误；
   - 结果：**3/3 全部通过 (0.43s)**。

2. **GPU 迭代与端到端管道验证**：
   - 验证 GPU 上 5 步实际二次型 LBFGS 迭代，Two-loop 递归与缓冲区正常更新；
   - 验证 `Worker(files=["tests/fixtures/input.cif"], optimizer1="LBFGS", model="mace", device="cuda:0")` 在真实 MACE 势能下端到端运行成功。

3. **回归测试与代码检查**：
   - `test_slot_states.py`：5/5 全部通过；
   - `test_worker_pipeline.py`：2/2 全部通过；
   - `git diff --check`：无任何多余空格或格式问题。
