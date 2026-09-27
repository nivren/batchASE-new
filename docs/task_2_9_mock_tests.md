# Task 2.9: 增加无 MACE/模型权重依赖的 Mock 测试与纯 CPU 回归套件

## 1. 任务背景与目标

在之前的测试架构中，端到端管道测试（如 [`tests/test_worker_pipeline.py`](../tests/test_worker_pipeline.py)）依赖两项重度前置条件：
1. 本地可用的 NVIDIA CUDA GPU 运行时。
2. 磁盘上预下载的 MACE-OFF23 模型权重文件（`~/.cache/mace/MACE-OFF23_small.model`，~100MB）。

当在无 GPU 的轻量 CI 环境（例如 GitHub Actions 标准 CPU Runner）、无卡开发机或个人笔记本上运行时，端到端测试会被直接跳过（`self.skipTest`），导致两阶段弛豫流程、动态补槽机制、错误结构隔离与汇总 CSV 输出等关键生产逻辑无法自动回归。

**本任务目标**：
- 实现一个完全自包含、无外部模型权重依赖、可纯 CPU 运行的一等公民 `MockBatchBackend`。
- 为原子 LJ 力提供解析保守性，并提供对称的合成晶胞恢复应力，使 FIRE、FIRE2、BFGS、LBFGS 能够稳定运行。
- 支持明确的故障注入（如原子重叠触发 `FailReason.FORCE_OVERFLOW`），以检验动态补槽与两阶段故障隔离。
- 提供纯 CPU 运行的独立管道测试套件 [`tests/test_mock_pipeline.py`](../tests/test_mock_pipeline.py)。

`MockBatchBackend` 是测试专用后端，不用于生成材料科学结果。它不实现周期镜像和距离截断，也不保证晶胞应力与返回能量之间的能量守恒一致性。

---

## 2. 核心架构与物理模型

### 2.1 `MockBatchBackend` 协议实现 ([`src/batchase/potentials/mock.py`](../src/batchase/potentials/mock.py))

`MockBatchBackend` 严格遵循 [`BatchPotential`](../src/batchase/potentials/base.py) 接口协议：
- `kind = "mock"`
- `device: torch.device`（默认 `"cpu"`，兼容 `"cuda"`）
- `dtype: torch.dtype`（默认 `torch.float64`）
- `predict(gbatch, compute_stress: bool = False)`
- `predict_from_atoms(atoms_list, compute_stress: bool = False)`
- 性能耗时接口：`mace_time`、`graph_time`、`forward_calls`，与 `Worker` 性能分析面板完全兼容。

### 2.2 测试用势场建模

为了避免原子在弛豫过程中因无排斥力而虚假聚集坍缩，同时支持晶胞全自由度形变与外压平衡，`MockBatchBackend` 采用以下物理模型：

1. **原子间相互作用（Lennard-Jones 势）**：
   对同一晶胞内的任意原子对 $(i, j)$，定义：
   $$V(r) = 4\epsilon \left[ \left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^6 \right]$$
   解析原子力：
   $$\mathbf{F}_{ij} = \frac{24\epsilon}{r^2} \left[ 2\left(\frac{\sigma}{r}\right)^{12} - \left(\frac{\sigma}{r}\right)^6 \right] (\mathbf{r}_i - \mathbf{r}_j)$$
   在未触发重叠故障注入时，满足牛顿第三定律与负梯度保守性：$\nabla_{\mathbf{r}_i} V = -\mathbf{F}_i$。

2. **晶胞体积弹性与形变应力恢复**：
   引入体积模量 $B$ 与剪切模量 $G$：
   $$\boldsymbol{\sigma}_{\text{vol}} = B \frac{V - V_0}{V_0} \mathbf{I}$$
   $$\boldsymbol{\sigma}_{\text{dev}} = \frac{G}{a_0} (\mathbf{cell} - a_0\mathbf{I})$$
   $$\boldsymbol{\sigma} = \boldsymbol{\sigma}_{\text{vol}} + \boldsymbol{\sigma}_{\text{dev}} + \frac{1}{V} \boldsymbol{\Xi}_{\text{virial}}$$
   其中 $\boldsymbol{\Xi}_{\text{virial}} = \frac{1}{2} \sum_{i,j} \mathbf{F}_{ij} \otimes \mathbf{r}_{ij}$ 为原子 Virial 应力项。代码对最终应力显式对称化，但该晶胞恢复项是合成测试项，并非返回能量的严格导数；外部压力下的体积行为只用于稳定管道测试，不代表材料的物理平衡体积。

3. **异常注入（Fault Injection）**：
   当检测到原子间距 $r_{ij} < 0.2\text{ Å}$（原子重叠）时，显式注入超大测试力（$> 10^4\text{ eV/Å}$），触发 Task 2.8 / 2.3 的 `fmax > 100`，由优化器与 Worker 自动标记为 `FailReason.FORCE_OVERFLOW`。该分支是故障注入，不是保守势的一部分。

---

## 3. 注册与 CLI 集成

1. **后端工厂注册** ([`src/batchase/potentials/__init__.py`](../src/batchase/potentials/__init__.py))：
   - `SUPPORTED_BACKENDS = ("mace", "mock")`
   - `create_backend("mock", ...)` 实例化并返回 `MockBatchBackend`。
2. **顶层导出** ([`src/batchase/__init__.py`](../src/batchase/__init__.py))：
   - 导出 `MockBatchBackend`。
3. **CLI 选项扩充** ([`scripts/batch_relax.py`](../scripts/batch_relax.py))：
   - `--model` 参数 choices 增加 `"mock"`。
   - `--device cpu` 覆盖 `--n_gpus`，可直接启动纯 CPU 流程；不指定时保持原有 CUDA 设备选择逻辑。

---

## 4. 专项测试套件 ([`tests/test_mock_pipeline.py`](../tests/test_mock_pipeline.py))

包含 5 项专项单元测试，覆盖纯 CPU 场景下的全流程：

| 测试函数 | 验证目标 |
| :--- | :--- |
| `test_mock_backend_contract_and_derivatives` | 验证 `BatchPotential` 协议、输出张量形状、Cauchy 应力对称性，以及原子位置有限元差分与解析力的一致性（$\Delta E \approx -\mathbf{F} \cdot \Delta \mathbf{r}$）。 |
| `test_mock_cpu_optimizers_convergence` | 验证四大主流优化器（FIRE、FIRE2、BFGS、LBFGS）在纯 CPU + UnitCellFilter 下均能在测试设置的最大步数内达到 $f_{\max} < 0.01\text{ eV/Å}$。 |
| `test_mock_worker_two_stage_pipeline_cpu` | 端到端执行 2 阶段松弛（Stage 1 press + Stage 2 final），验证 CIF 生成、JSON 写入及 Task 2.8 指标（`fmax_atom`, `fmax_stress`, `fmax_stress_gpa`）的有效输出。 |
| `test_mock_worker_replenishment_and_isolation_cpu` | 混合正常结构与重叠结构，验证故障隔离、补槽机制、第二阶段准入拦截与汇总 CSV 中的状态记录。 |
| `test_mock_scheduler_e2e_cpu` | 启动多 Worker Scheduler（`num_workers=2`, `device="cpu"`），验证多进程并行调度与 `results_scheduler.csv` 生成。 |

---

## 5. 验证结果

- **专项测试**：`tests/test_mock_pipeline.py` 全部 5 项测试通过。
- **全量测试**：`tests/test_*.py` 共 64 项测试通过，其中 7 项因环境条件跳过（0 failures, 0 errors）。
- **执行效率**：当前环境纯 CPU 下全量套件运行约 7.5 秒，实际耗时随机器和进程调度变化。
