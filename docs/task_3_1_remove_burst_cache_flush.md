# Task 3.1：移除 Burst 级缓存清理

## 1. 背景与证据

`BFGSFusedLS.run()` 和 `LBFGS.run()` 会在每个动态补槽 Burst 结束时调用 CUDA 缓存清理。由于 Worker 在结构完成后会频繁重新进入 `run()`，这些调用会反复清空可复用的缓存块，并额外触发 Python 垃圾回收。

历史探针使用 `2 GPU / 10 workers / batch=25 / 500 structures`，并明确触发补槽：

| 版本 | 墙钟 | Stage 1 单位批步 | 吞吐 |
| --- | ---: | ---: | ---: |
| 基线平均值 | 169.68 s | 29.63 ms | 176.8 structs/min |
| 移除 Burst 清理 | 164.65 s | 26.86 ms | 182.2 structs/min |

对应改善约为：Stage 1 优化器单位批步 9%，总墙钟 3%，吞吐 3%。峰值显存仍约 1.4 GB，未观察到显存持续增长。

证据日志位于 Git 工作区之外的 `/tmp`：

- `/tmp/batchase_probe_2g10w25_baseline/opt.log`
- `/tmp/batchase_probe_2g10w25_baseline_repeat/opt.log`
- `/tmp/batchase_probe_2g10w25_nogc/opt.log`

## 2. 实施范围

### 2.1 优化器层

删除以下常规 Burst 清理：

- `BFGSFusedLS.run()` 中的 `torch.cuda.empty_cache()` 和 `gc.collect()`；
- `LBFGS.run()` 中的 `torch.cuda.empty_cache()`。

不修改优化器状态、步长、线搜索和收敛逻辑，也不增加 `clear_cache_per_burst` 公共参数。

### 2.2 Worker 阶段边界

Worker 增加一次性 `_clear_stage_memory()` hook：

- 在 `_run_stage()` 返回之后调用，而不是在其返回语句之前调用；
- Stage 1 和 Stage 2 各最多调用一次；
- 先执行 `gc.collect()`，CUDA 设备再执行 `torch.cuda.empty_cache()`；
- CPU 路径不调用 CUDA 缓存清理；
- 通过 `try/finally` 覆盖阶段异常路径。

阶段清理只处理阶段结束后已经不可达的 Python 对象和未使用缓存块，不代表释放模型或仍被引用的活动张量。

### 2.3 不纳入本任务的内容

本任务不实现 OOM 自动重试或动态缩小 batch。OOM 发生在 Burst 中途时，优化器状态可能已经部分更新，直接复用同一优化器重试并不安全；这需要另一个任务负责批次重建、失败记录和 Scheduler 退出码处理。

## 3. 测试

`tests/test_burst_cache_behavior.py` 使用 monkeypatch 验证：

1. 正常 `LBFGS.run()` 和 `BFGSFusedLS.run()` 不再执行 Burst 级 `empty_cache()` / `gc.collect()`；
2. Worker 阶段清理在 CUDA 设备上调用一次，在 CPU 设备上不调用 `empty_cache()`。

显存稳定性和性能不在单元测试中固定断言，使用外部探针按固定参数重复测量墙钟、批步耗时、吞吐、峰值显存、收敛率和失败率。

## 4. 本次验证结果

- 专项测试：`test_burst_cache_behavior.py`，2 项通过；
- 全量回归：61 项通过，7 项按现有环境条件跳过；
- `compileall`：`src`、`tests`、`scripts` 编译通过；
- `git diff --check`：通过，新增文件也无空白错误；
- 测试使用项目已有虚拟环境，探针和测试产物均写入 `/tmp` 或临时目录，未加入 Git 工作区。
