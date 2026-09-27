# Task 3.2：细分 BFGSFusedLS Profiler 计时

## 1. 目标与范围

本任务为 `BFGSFusedLS` 增加可选的算法阶段观测，用于定位后续 line-search 优化的瓶颈。Profiler 默认关闭；未开启时不执行 `record_function`、计时器或 CUDA 同步。

现有 Worker 的 `opt_s` 仍是优化器总耗时的权威指标。本任务的五段统计只覆盖明确标注的代码区间，不把模型推理时间重复计入算法区间。

## 2. 五段定义

| 区间 | 统计内容 |
| --- | --- |
| `hessian_update` | `BFGSFusedLS.update()` 的 Hessian 初始化和拟牛顿更新 |
| `direction_calc` | 搜索方向矩阵向量乘法与步长缩放 |
| `host_device_transfer` | 线搜索控制路径中的显式张量搬运；不声称覆盖模型回调内部的全部搬运 |
| `linesearch_state` | More-Thuente 状态机的初始化、步进、状态更新和结果整理；排除 `func()` / `fprime()` 模型回调 |
| `position_filter_update` | 位移拼装和最终 `set_positions()`；模型力和能量计算继续由既有 MLIP 计时统计 |

五段统计可能不覆盖整个优化器 wall time，因此百分比是相对于 `measured_total_s` 的占比，不能与 Worker 的 `opt_s` 直接相加。

Worker 在阶段汇总时会增加一组账本对账字段；这些字段不是第六个算法阶段，也不会新增计时或 CUDA 同步：

- `optimizer_wall_s`：Worker 粗粒度的阶段 `opt_s`；
- `unprofiled_optimizer_s` / `_ms`：`max(optimizer_wall_s - measured_total_s, 0)`；
- `unprofiled_optimizer_pct`：未落入五段区间的比例；
- `profiler_coverage_pct`：`measured_total_s / optimizer_wall_s * 100`。

覆盖率用于解释五段统计与 `opt_s` 的差异，不代表算法质量。由于两套计时包含不同的 CUDA 同步和模型计时口径，覆盖率不应被强行要求为 100%。

## 3. 使用方式

配置文件中开启：

```bash
PROFILE="True"
```

也可以提供 profiler schedule。动态补槽 Burst 通常只有 1～5 步，推荐：

```bash
PROFILE='{"wait":0,"warmup":0,"active":1,"repeat":1}'
```

默认输出目录为 `${OUTPUT_PATH}/log`。Scheduler 和直接构造的 Worker 均使用该目录；也可以通过 `profiler_log_dir` 覆盖。

Trace 文件由 PyTorch 写入：

```text
${OUTPUT_PATH}/log/BFGSLS_<pid>.pt.trace.json
```

可用 TensorBoard 查看：

```bash
tensorboard --logdir ${OUTPUT_PATH}/log
```

## 4. 统计接口与输出

`BFGSFusedLS.get_profiler_breakdown()` 返回 JSON 可序列化字典，包含：

- 各区间的 `_s`、`_ms` 和 `_pct`；
- `calls`：各区间进入次数；
- `measured_total_s` 和 `measured_total_ms`；
- `enabled`：是否开启 profiler。

计时器在同一个优化器实例的多个 Burst 之间累计，适合 Worker 的阶段汇总；`reset_profiler_timings()` 用于显式开始新的统计周期，不会修改优化器状态。

开启 profiler 时，Worker 在 `metrics/worker_<id>.json` 的对应 stage 下写入 `profiler_breakdown` 及上述账本对账字段，并在 `opt.log` 输出各区间百分比、已测时间、未归类时间和覆盖率。已有 `opt_s`、结构 JSON 和 CSV 字段保持不变。

## 5. 性能与限制

Profiler 模式会产生 Trace、额外事件记录，并在 CUDA 计时区间同步设备，因此只建议用于小规模诊断。正式性能对比应关闭 profiler；五段统计不作为生产吞吐指标。

未开启 profiler 时，仍保留极轻量的路径分支，但不会执行 `perf_counter()`、`record_function` 或 `torch.cuda.synchronize()`。

## 6. 验证

专项测试 `tests/test_bfgs_profiler.py` 覆盖：

1. 关闭 profiler 时计时器保持为零；
2. 开启 profiler 时五段均产生非负计时和调用次数；
3. Trace 中出现五个 `bfgs::` 标签；
4. 重置接口清空累计值；
5. profiler 开关不改变 CPU Mock 优化轨迹；
6. Worker 阶段 JSON 能导出 `profiler_breakdown`。
7. Worker 能正确计算未归类优化器时间和五段统计覆盖率。

本次验证结果：专项测试 `test_bfgs_profiler.py` 为 5/5 通过；全量测试为 71 项通过、7 项按环境条件跳过；`compileall` 和 `git diff --check` 均通过。
