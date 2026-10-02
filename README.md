# batchASE (Batched ASE Relaxation Engine)

`batchASE` 是面向分子晶体结构预测（CSP）及材料模拟的高吞吐量批量结构松弛优化引擎。

## 特性亮点

- **硬件解耦架构**：统一硬件加速算子抽象，支持 NVIDIA CUDA、海光 DCU (ROCm/HIP)、Triton、TileLang 与 CPU 备用执行。
- **无锁零同步优化**：消除每步优化的 CPU-GPU 状态同步停顿（`_dirty` 状态缓存），提升 GPU 利用率。
- **完整批次动态调度**：提前按均匀结构数或原子预算分批，worker 从公共队列领取 batch，完成两个阶段后领取下一批；批内仅移除完成结构，不跨批补位。
- **优雅的线性代数策略**：支持小体系在 CPU MKL 与 GPU 间灵活切换 `eigh` 特征值分解，无需复制代码。
- **标准包结构**：采用 PEP 517/621 `src/` 布局，支持独立 wheel 打包发布与 editable 开发。

## 快速安装

```bash
pip install -e .
```

## 分批与排序

CLI 的 `--batch_mode` 支持两种互斥规则：

- `bsize`（默认）：根据 `--batch_size` 计算 `ceil(N/batch_size)` 个批次，均匀分配所有输入，各批结构数差别不超过 1。忽略 `--max_batch_atoms`。
- `atoms`：按 `--max_batch_atoms` 的正整数预算进行首次适配装箱，忽略 `--batch_size`。使用 ASE 实际读取的原子数；单结构超过预算则在 worker 启动前报错。

两种规则共用 `--num_structures N` 和 `--structure_order rand|syst|atom`。`syst` 按文件名顺序取前 N 个；`rand` 用 seed 随机抽取 N 个并保留抽样顺序；`atom` 抽取同一随机样本后按实际原子数降序，同原子数按文件名排序。只解析所选 CIF，不为抽样读取整个目录的结构。N 为 0 或不少于文件数时使用全部：`syst` 按文件名排列，`rand` 打乱，`atom` 按原子数降序。

`--structure_order_seed` 默认 42，控制 `rand` 的抽样和排列、`atom` 的抽样；`syst` 和使用全部结构的 `atom` 忽略 seed。装箱允许后续结构回填先前 batch，因此装箱后的文件展开顺序不一定等于所选文件的排列顺序。

```bash
python scripts/batch_relax.py --target_folder /path/to/cifs \
    --batch_mode bsize --batch_size 25 \
    --structure_order rand --structure_order_seed 42

python scripts/batch_relax.py --target_folder /path/to/cifs \
    --batch_mode atoms --max_batch_atoms 2000 \
    --num_structures 10000 --structure_order atom --structure_order_seed 42
```

各 worker 复用一个模型，每批、每阶段创建新的优化器。完整分组写入 `batch_plan.json`，实际执行及累计指标分别写入 `metrics/batch_<id>.json` 和 `metrics/worker_<id>.json`。worker 崩溃或缺少批次完成记录时，运行以非零状态退出。

CLI 默认通过 `--batch_plan_cache_dir .cache/batch_plans` 缓存完整内部计划（相对于当前工作目录）。传入空字符串或 `none` 可关闭缓存，删除该目录可强制重建。缓存在只处理文件名的抽样之后检查所选文件的绝对路径、大小、纳秒修改时间、所选数量/顺序、有效分批参数及规划/ASE 版本；命中后跳过 CIF 解析、原子数排序和装箱，每次重新执行弛豫并生成新的运行 ID。GPU/worker、模型、优化器和输出目录不参与缓存键。缓存损坏或不可写会回退到正常规划，写入通过原子替换完成。输入内容改变但保留大小和修改时间时，需手动清空缓存。新规划版本不会误用旧抽样规则的缓存。

Python `Scheduler(..., batch_plan_cache_dir="/path/to/cache")` 可显式启用缓存，API 默认关闭。输出 `batch_plan.json` 的 `planning_cache` 记录缓存状态与规划耗时。固定外部计划不使用此缓存。

不同输入、模式、排序或 seed 的计划独立保存，切换回来可以复用。`--batch_plan_cache_limit`（Python 参数 `batch_plan_cache_limit`）默认 20，必须为正整数；超出目录上限时删除最久未使用的计划，缓存命中刷新使用时间。原生 Unix 进程通过目录锁协调写入和清理。

外部工程可继续使用 `--fixed_batch_plan` 提供每个 worker 恰好一批的 JSON。固定计划不执行内部排序或装箱；旧的 `--prebatch`、`--use_ordered_files`、`--structure_select`、`--random_seed` 参数已移除。
