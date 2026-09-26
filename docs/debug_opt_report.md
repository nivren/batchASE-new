## 总体结论

原计划中多数源码问题真实存在，但若严格按“当前常用运行路径、结果正确性、实测收益”排序，需要做几处明显调整：

- **真正的 P0** 应集中在崩溃、NaN、cuSOLVER 错误结果和“失败被误判为收敛”。
- **B4 已实测有效**，应作为首个性能修复，但不属于正确性 P0。
- **B1 收益被严重高估**，不应优先实施。
- **B2 的瓶颈判断基本正确**，但不能简单把现有 Python 状态机搬到 GPU。
- A6、Stage 1 Active-Slicing、pytest fixture 等部分判断不成立。
- A7、D1、D6 等问题存在，但原计划对影响范围或修复方式有所夸大。

---

# 一、逐项审阅结论

## A 类：正确性与稳定性

| 项目 | 核验结论 | 更新优先级 | 建议 |
|---|---|---:|---|
| **A1 `det` 未取绝对值** | 问题真实，但当前 1000 个实际结果中未发现负体积。单纯 `abs(det)` 还可能掩盖晶胞翻转。 | **P1** | 返回正体积，同时对 `det<=0` 或接近零的晶胞明确告警/失败。位置：`3rdparty/batchASE/src/batchase/relaxation/optimizable.py:465` |
| **A2 `constant_volume` reshape** | 确凿代码错误，`[B]` 不能 `.view(3,-1)`；但该选项当前未启用。 | **P1** | 改为 `.view(-1,1)`，增加 `B=1/2/4` 单测。位置：`3rdparty/batchASE/src/batchase/relaxation/optimizable.py:787` |
| **A3 LBFGS 未定义 `device`** | LBFGS 活跃路径确实会 `NameError`。但 BFGSFusedLS 中相同问题位于未调用辅助函数。 | **P0** | LBFGS 使用相关张量的 `.is_cuda` 或 `.device.type`；不能直接改成 `self.device`，因为当前 LBFGS 没有完整定义该属性。 |
| **A4 cuSOLVER `info` 未检查** | 确凿正确性隐患。API 调用成功不代表所有矩阵都收敛。 | **P0** | 检查失败切片；仅对失败矩阵回退 `torch.linalg.eigh` 或重置 Hessian，并记录原因。位置：`cusolver_batched.py:195` |
| **A5 CPU BFGS 除零** | 已复现：零步长槽位出现 `inf*0 -> NaN`。 | **P0** | 对齐 GPU 路径的有效步长掩码。当前常用配置 `BFGS_CPU_THREADS=0` 不走该路径，但代码必须修。 |
| **A6 mask 跨 burst 重置** | 原审阅判断不成立。重建 batch 前已移除收敛槽，存活和新补入槽本来就应继续计算。 | **不处理** | 不建议继承旧 `_update_mask`，避免将已变化槽位错误冻结。 |
| **A7 输出未 wrap** | 不是当前 P0。ASE CIF writer 会自动折叠分数坐标，实际 1000 个输出也均位于合法范围。 | **P3** | 若下游要求完整分子不跨边界，再基于分子连通关系做“质心整体 wrap”，不要逐原子 wrap。 |
| **FIRE `dtmin`** | 判断正确：当前 FIRE 混入了 FIRE2 的下限规则。 | **P1** | FIRE 移除负功率时的 `dtmin` 截断；FIRE2 保留，并增加 ASE 对照测试。 |

## 新发现的关键问题

### 失败结构被误判为收敛

当前逻辑大致为：

```python
update_mask = (max_force >= fmax) & (max_force <= f_upper_limit)
```

随后“不活跃槽”会进入 `converge_indices_list`。这意味着：

- `force > f_upper_limit`
- `NaN`
- `Inf`

都有可能被当成“收敛完成”，而不是失败。

这是比单纯设置 `f_upper_limit=100` 更严重的问题。

**更新优先级：P0。**

应引入至少三态：

1. `active`
2. `converged`
3. `failed`

失败原因至少区分：

- `nan_force`
- `inf_force`
- `force_overflow`
- `invalid_cell`
- `max_steps`
- `eigensolver_failed`

在完成三态改造前，不能直接降低 `f_upper_limit`，否则会增加假收敛。

---

# 二、性能项目实测结论

测试参数：

- 500 个常用 92 原子结构
- 2 张 H100
- 10 workers
- batch size 25
- 每 worker 50 个结构，明确触发补槽
- FastEq、MPS、BFGSFusedLS → BFGS

| 版本 | 墙钟 | Stage 1 优化器耗时/批步 | 吞吐 |
|---|---:|---:|---:|
| 原始基线 1 | 168.49 s | 29.80 ms | 178.0 structs/min |
| 原始基线 2 | 170.87 s | 29.45 ms | 175.6 structs/min |
| B1 紧凑 Rank-2 | 167.95 s | 29.49 ms | 178.6 structs/min |
| B4 移除回收 | **164.65 s** | **26.86 ms** | **182.2 structs/min** |

证据：

- `/tmp/batchase_probe_2g10w25_baseline/opt.log:1203`
- `/tmp/batchase_probe_2g10w25_baseline_repeat/opt.log:1205`
- `/tmp/batchase_probe_2g10w25_compact/opt.log:1205`
- `/tmp/batchase_probe_2g10w25_nogc/opt.log:1200`

## B1：紧凑 Rank-2 更新

数学变换正确，GPU 数值误差约 `1e-15`。但性能结论与原计划不符：

- H100、`batch=25,D=285`：
  - 原展开核：`0.2184 ms`
  - 紧凑核：`0.1415 ms`
  - 更新核自身仅快 `1.54×`
- 整轮 Stage 1 单位优化器耗时没有可辨识改善。
- 总墙钟差异落在两次基线自身波动范围内。
- 峰值显存也没有可见变化。

原因是 BFGS 更新核只占优化器步骤的一小部分；主要时间在线搜索和其他 Python/张量控制逻辑。

**更新优先级：从 P0 降为 P3。**

仅在以下场景重新考虑：

- 单结构自由度显著超过当前 `D=285`
- 使用算力较弱、FP64 GEMM 较慢的 GPU
- profiler 证明 Hessian 更新占比明显升高

位置：`3rdparty/batchASE/src/batchase/relaxation/optimizers/bfgsfusedls.py:238`

## B4：每个 burst 回收缓存

实测有效：

- Stage 1 优化器单位批步耗时降低约 **9–10%**
- 整体墙钟降低约 **3%**
- 吞吐提高约 **3%**
- 峰值显存仍约 `1.4 GB`，未观察到风险

**更新优先级：P1，但应作为第一个性能修改实施。**

建议：

- 删除每个 burst 末尾的 `gc.collect()`
- 删除常规 `torch.cuda.empty_cache()`
- 仅在阶段结束、捕获 OOM 后或显式调试模式执行

位置：`3rdparty/batchASE/src/batchase/relaxation/optimizers/bfgsfusedls.py:407`

## B2：CPU line search

瓶颈方向基本正确，但原修复方案过于简单。

相同 `batch=25,D=285` 的状态机探针：

- 当前 CPU 路径：`24.95 ms`
- 将现有实现直接放到 GPU：`111.15 ms`

直接 GPU 化慢约 `4.5×`，原因是大量：

- Python 逐槽循环
- 动态布尔掩码
- 小张量 GPU kernel
- `.item()` 隐式同步
- CPU/GPU 状态交叉

位置：

- CPU 初始化：`3rdparty/batchASE/src/batchase/relaxation/optimizers/bfgsfusedls.py:63`
- 逐槽循环：`3rdparty/batchASE/src/batchase/relaxation/optimizers/bfgsfusedls.py:959`

**更新优先级：P1 性能研发项。**

推荐顺序：

1. 对 production line search 分段计时。
2. 将槽位状态改为连续张量。
3. 预计算结构切片，消除反复 `i == batch_indices`。
4. 先尝试 CPU 向量化。
5. 只有在消除 Python 控制流后，再比较 GPU 实现。

## B3：Stage 2 与 Active-Slicing

需要拆成两个不同问题：

- **Stage 1 Active-Slicing：不适用。** Stage 1 没有批量 `eigh`，且 Hessian 更新已经按活跃索引处理。
- **Stage 2 改逆 Hessian：属于算法变更。** 可以消除 `eigh`，但会改变当前“直接 Hessian + 特征值绝对值/截断”的步长语义。

**更新优先级：P2 实验项。**

需比较：

- 收敛率
- 最终能量分布
- 最大残余力
- 迭代步数
- 病态结构稳定性

## B5/B7

- `FixSymmetry` 会改变允许搜索的势能面，不是透明的性能优化。
- 分别输出原子残余力和残余应力非常有价值。

调整为：

- **FixSymmetry：P3，可选科学模式**
- **残余力/应力输出：P1 可观测性**

---

# 三、工程和结果语义

| 项目 | 审阅结论 | 更新优先级 |
|---|---|---:|
| **D6 `training=self.use_compile`** | 语义错误，但当前 CLI 的 `compile_mode` 又没有真正传入 backend，所以常规运行尚未触发。 | **P1** |
| **D1 `MOLECULE_SINGLE`** | 示例值 13 不应简单统一为 46，因为不同体系每分子原子数不同。 | **P1** |
| **D2 `examples/csp.sh`** | 确实是旧入口，但不影响常用 `relaxation_auto.sh`。 | **P2** |
| **D3 死代码** | `SlotManager`、`robust_eigh` 未使用，但不应为了消除死代码而强行接入。 | **P3** |
| **D4 FrechetCellFilter** | SciPy CPU 循环性能差，但逻辑大体源自 ASE，不能直接认定为错误。 | **P2** |
| **D5 占位后端** | 配置暴露不可用后端会造成运行时失败。 | **P1** |
| **C1 能量语义** | Stage 1 含 `pV`，实质是焓；Stage 2 压力为零时是势能。当前统一叫 `energy` 易误用。 | **P1** |
| **C2 步数统计** | 不完全是“空转步”；主要问题是 line-search 轮次、力评估次数和接受步混在一起。 | **P2** |
| **测试签名** | 带默认值的 `device="cuda:0"` 不会被 pytest 当成 fixture；原判断错误。 | **修正测试策略** |

## D6 推荐修复

同时处理两个问题：

1. MACE 调用固定 `training=False`
2. Scheduler → Worker → Backend 正确传递 `compile_mode`

然后测试：

- compile 关闭与开启时能量/力一致
- 不保留无用 autograd graph
- 编译失败时能清晰回退或报错

## D1 推荐修复

不要把所有配置固定成 46，而是：

- 要求 `MOLECULE_SINGLE` 显式提供
- 校验 `natoms % molecule_single == 0`
- 输出 `num_molecules`
- 非整数时直接失败或明确标记不可归一化

## C2 推荐指标

建议新增：

- `optimizer_iterations`
- `accepted_steps`
- `force_evaluations`
- `line_search_evaluations`
- `replenishment_count`
- `active_slot_steps`
- `failed_reason`

---

# 四、可重复性问题

两次完全相同的原始基线出现明显轨迹分叉：

- Stage 1 步数只有约 2% 的结构完全相同
- 最终能量差中位数较小，但少数结构进入不同局部极小值
- 最终能量差 P95 约 `9.8 kJ/mol`
- 最大差异约 `66.5 kJ/mol`

这不一定是 batchASE 的确定性 Bug，可能来自：

- MPS 并发调度
- FastEq/CUDA 原子操作顺序
- 浮点舍入导致 line search 分支变化
- 局部极小值敏感性

但它会直接影响性能 A/B 和算法正确性比较。

**建议新增 P1：基准与可重复性规范。**

- 性能测试至少重复 3 次。
- 使用单位 batch-step 和结构吞吐，而不是单次墙钟。
- 正确性单测使用单 worker、无 MPS 的可重复模式。
- 大规模结果比较使用能量/密度/残余力分布，而不是要求轨迹逐步一致。
- 对高能量差异结构单独输出名单。

---

# 五、更新后的实施计划

## 第一阶段：P0 正确性与失败隔离

1. **实现 active/converged/failed 三态**
   - NaN、Inf、大力、非法晶胞不能进入收敛结果。
   - 输出结构级失败原因。

2. **修复 LBFGS 未定义 `device`**
   - 增加 CPU/CUDA 最小运行测试。

3. **修复 CPU BFGS 零步长 NaN**
   - 覆盖活跃槽和空槽混合场景。

4. **检查 cuSOLVER `info`**
   - 失败矩阵单独回退。
   - 记录失败索引和状态码。
   - 避免整批静默使用错误特征向量。

验收标准：

- NaN/Inf 永远不会被记为 `converged=True`
- LBFGS 能完成 CPU/CUDA mock 优化
- 零步长槽不产生 NaN
- 人工制造的 eigensolver 失败能正确回退

## 第二阶段：P1 低风险正确性与接口修复

1. 修复 `constant_volume` reshape。
2. 规范负体积和退化晶胞处理。
3. FIRE/FIRE2 与 ASE 语义对齐。
4. 修复 compile 参数传递及 `training=False`。
5. 隐藏或拒绝未实现后端。
6. 校验 `MOLECULE_SINGLE` 与分子数。
7. 区分 `enthalpy_kj_mol` 和 `energy_kj_mol`。
8. 输出原子残余力、残余应力。
9. 增加无 MACE/GPU 依赖的 mock 测试。

## 第三阶段：P1 性能优化

1. **先移除 burst 内 `empty_cache()/gc.collect()`**
   - 已有约 3% 整体收益证据。
   - 保留 OOM 回退清理机制。

2. **细分 BFGSFusedLS profiler**
   - Hessian update
   - direction calculation
   - line-search state
   - position/filter conversion
   - host/device transfer

3. **重构 line search**
   - 先消除重复 mask 和 Python 对象状态。
   - 再做 CPU 张量化。
   - 最后才测试完整 GPU 张量化。

4. **固定性能验收参数**
   - `2 GPU / 10 workers / batch=25`
   - 至少 500 个结构以触发补槽
   - 每个版本运行至少 3 次
   - 报告墙钟、batch-step、structures/min、峰值显存和失败率

建议性能合入门槛：

- 墙钟改善至少 3%
- 或目标组件单位耗时改善至少 10%
- 收敛率不得下降
- 失败结构不得增加
- 最终能量分布无系统性偏移

## 第四阶段：P2 可观测性与架构整理

1. 增加 accepted step、force evaluation、line-search evaluation。
2. 修复或移除旧 `examples/csp.sh`。
3. 将 FrechetCellFilter 标记为 experimental。
4. 为非确定性测试建立固定单 worker 模式。
5. 清理无路线图的 `SlotManager`、`robust_eigh` 和旧辅助函数。

## 第五阶段：P3 实验项

1. Stage 2 逆 Hessian算法实验。
2. 特定大自由度体系重新评估紧凑 Rank-2。
3. 基于分子连通关系的整体 wrap。
4. 可选 `FixSymmetry` 科学约束模式。

---

## 最终优先级摘要

- **P0**：失败三态、LBFGS、CPU BFGS NaN、cuSOLVER `info`
- **P1 正确性**：A1、A2、FIRE、compile 接线、配置校验、后端能力、能量字段、测试
- **P1 性能**：B4，然后是 line-search 分析与向量化
- **P2**：指标完善、旧入口、Frechet、Stage 2 算法实验
- **P3/暂缓**：B1 紧凑 Rank-2、FixSymmetry、分子整体 wrap、死代码整理
- **不实施**：A6 mask 继承、Stage 1 Active-Slicing、把 pytest 默认参数误判为 fixture

所有性能探针和测试结果均位于 `/tmp`，工程及 batchASE 子模块 Git 工作区没有被修改。
