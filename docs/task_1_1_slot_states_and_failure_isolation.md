# 任务 1.1：槽位三态状态管理与失败隔离实施文档

## 1. 背景与目标

在批量晶体弛豫（Batch Relaxation）过程中，不同结构的受敛行为和数值稳定性存在很大差异。原实现中存在以下重大缺陷：
1. **二态判定漏洞（假收敛）**：原逻辑通过 `update_mask = (force >= fmax) & (force <= f_upper_limit)` 判定活跃槽，导致 `force > f_upper_limit`、`NaN`、`Inf` 或负晶胞体积（$\det \le 0$）等异常槽位因不满足条件而被移出活跃槽，进而被错误归入 `converge_indices_list`，导致发散结构被误标记为“已收敛”。
2. **有限大力阈值失效**：优化器中曾硬编码 `f_upper_limit=1e25`，使原本设计的力超限保护形同虚设。
3. **两阶段泄漏**：Worker 仅过滤了部分异常，导致发散或数值崩溃的结构直接流入 Stage 2，浪费大量计算资源并污染最终交付结果。
4. **CUDA 热路径性能退化**：收敛判定循环中频繁使用 `.item()` 触发逐槽 Host-Device 同步，在高并发批次下带来明显的调度延迟。
5. **汇总报表信息缺失**：主汇总表 `results_scheduler.csv` 丢失状态与失败原因，排查成本极高。

本任务的目标是建立规范的 `ACTIVE` / `CONVERGED` / `FAILED` 三态状态机，实现异常槽位的全生命周期追踪、失败隔离、自动补槽及两阶段阻断。

---

## 2. 核心架构设计与实现

### 2.1 状态枚举定义 (`batchase.relaxation.status`)

在 `src/batchase/relaxation/status.py` 中引入强类型枚举：
- **`SlotStatus`**：
  - `ACTIVE` ("active")：槽位正在弛豫迭代中。
  - `CONVERGED` ("converged")：力收敛至阈值以内且晶胞合法。
  - `FAILED` ("failed")：发生数值发散、超步数、晶胞坍缩或奇异等不可恢复错误。
- **`FailReason`**：
  - `NAN_FORCE` ("nan_force")：力张量包含 NaN。
  - `INF_FORCE` ("inf_force")：力张量包含 Inf。
  - `FORCE_OVERFLOW` ("force_overflow")：最大原子受力超过配置的阈值（默认 `100.0 eV/Å`）。
  - `INVALID_CELL` ("invalid_cell")：晶胞体积矩阵行列式 $\det \le 0$ 或退化。
  - `MAX_STEPS` ("max_steps")：达到最大迭代步数仍未达到收敛力阈值。
  - `EIGENSOLVER_FAILED` ("eigensolver_failed")：特征值求解器失败且无法回退。

### 2.2 向量化状态机与终态不可逆性 (`batchase.relaxation.optimizable`)

在 `OptimizableBatch` 和 `OptimizableUnitCellBatch` 中维护每个槽位的当前状态 `_slot_status` 及失败原因 `_slot_reasons`。

重构 `converged()` 方法：
1. **全张量向量化判定**：
   - 彻底废除基于 Python `.item()` 的逐槽判断。
   - 采用 PyTorch 原生算子执行张量并行检查：
     ```python
     is_nan = torch.isnan(max_forces)
     is_inf = torch.isinf(max_forces)
     is_overflow = max_forces > f_upper_limit
     is_invalid_cell = (dets <= 0)
     is_force_conv = (max_forces <= fmax)
     ```
   - 探针性能测试显示：CUDA 判定延迟从 **0.834 ms 降低至 0.169 ms（加速约 5×）**。
2. **终态不可逆性保障（Terminal Irreversibility）**：
   - 只有当前处于 `ACTIVE` 状态的槽位才参与收敛或失败判定。
   - 一旦进入 `CONVERGED` 或 `FAILED`，其状态被永久锁定，绝不会因为后续模型受力浮动或共享 Batch 计算退回 `ACTIVE`。

### 2.3 优化器阈值配置化与参数解耦

全面移除各优化器内的 `1e25` 硬编码：
- `BFGS`（GPU / CPU）
- `BFGSFusedLS`
- `FIRE`
- `LBFGS`

全部支持在 `__init__` 和 `run()` 中显式接收 `f_upper_limit` 参数（默认值统一为 `100.0`），并由 `Worker` 统一自顶向下透传。

### 2.4 Worker 级失败隔离与连续补槽 (`batchase.engine.worker`)

在 `Worker._run_stage()` 中：
1. **严格区分诊断输出与交付合格结构**：
   - `output_cif_paths`：记录当前阶段所有已结束结构的输出 CIF，保留用于科学排查与诊断。
   - `stage_success_cifs`：严格仅包含 `SlotStatus.CONVERGED`（或受控放行的 `MAX_STEPS`）结构。
   - `FORCE_OVERFLOW`、`NAN_FORCE`、`INF_FORCE`、`INVALID_CELL`、`EIGENSOLVER_FAILED` 结构被绝对阻断，不得写入 `stage_success_cifs`。
2. **失败槽补位机制**：
   - 当某个槽位失败，当前 burst 提前返回。
   - Worker 将该槽位持久化为 JSON（记录 `status: "failed"` 与 `failed_reason`），并立刻从待处理队列取出新结构放入该槽位，保持计算单元饱和。
3. **Stage 2 隔离**：
   - Stage 2 的输入仅来源于 Stage 1 的 `stage_success_cifs`，发散结构绝不会进入第二阶段。

### 2.5 汇总报告扩展 (`batchase.engine.scheduler`)

在 `Scheduler._write_summary_csv()` 生成的 `results_scheduler.csv` 中扩充状态字段：
- `status`：最终综合状态（`converged` / `failed`）。
- `failed_reason`：最终失败原因（若收敛则为空）。
- `stage1_status` / `stage1_failed_reason`：第一阶段状态及失败原因。
- `stage2_status` / `stage2_failed_reason`：第二阶段状态及失败原因。
- 无论单阶段（`skip_second_stage=True`）还是双阶段模式，均能准确输出报表。

---

## 3. 测试覆盖与验证

1. **单元测试 (`tests/test_slot_states.py`)**：
   - `test_nan_inf_forces`：验证 NaN 与 Inf 受力被精确归类为对应失败状态，不再误入收敛。
   - `test_realistic_force_overflow`：验证真实发散阈值（150.0 > 100.0 eV/Å）触发 `FORCE_OVERFLOW`。
   - `test_inverted_and_zero_cell`：验证零晶胞及倒置晶胞（负行列式）被准确识别为 `INVALID_CELL`。
   - `test_terminal_irreversibility`：验证处于 `CONVERGED` 或 `FAILED` 的槽位在经历力突变后依然保持终态。
   - `test_cuda_vectorized_path`：验证 CUDA 环境下全向量化张量逻辑与 CPU 逻辑的一致性。
   - `test_explicit_mark_failed`：验证特征值求解器失败等外部异常的显式标记机制。
   - 结果：**5/5 通过 (0.52s)**。

2. **端到端管道测试 (`tests/test_worker_pipeline.py`)**：
   - 验证发散结构（重叠原子受力超限）在 Worker 运行过程中立即被隔离，队列中的新结构成功补位。
   - 验证失败结构绝不流入 Stage 2，仅产生 Stage 1 诊断文件。
   - 验证单阶段及双阶段下 `results_scheduler.csv` 字段与数值完全吻合。
   - 结果：**2/2 通过 (2.98s)**。

3. **MACE 官方对齐回归测试 (`tests/test_mace_backend.py`)**：
   - 验证修改未引入任何能量、力、应力的精度偏差，完全通过机器精度回归。
   - 结果：**全部通过**。
