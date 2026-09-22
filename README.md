# batchASE (Batched ASE Relaxation Engine)

`batchASE` 是面向分子晶体结构预测（CSP）及材料模拟的高吞吐量批量结构松弛优化引擎。

## 特性亮点

- **硬件解耦架构**：统一硬件加速算子抽象，支持 NVIDIA CUDA、海光 DCU (ROCm/HIP)、Triton、TileLang 与 CPU 备用执行。
- **无锁零同步优化**：消除每步优化的 CPU-GPU 状态同步停顿（`_dirty` 状态缓存），提升 GPU 利用率。
- **正交动态批处理**：独立的 `SlotManager` 负责长尾晶体动态替换，与无状态（FIRE2）及有状态（BFGS）优化算法干净解耦。
- **优雅的线性代数策略**：支持小体系在 CPU MKL 与 GPU 间灵活切换 `eigh` 特征值分解，无需复制代码。
- **标准包结构**：采用 PEP 517/621 `src/` 布局，支持独立 wheel 打包发布与 editable 开发。

## 快速安装

```bash
pip install -e .
```
