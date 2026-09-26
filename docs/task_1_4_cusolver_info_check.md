# 任务 1.4：检查 cuSOLVER info 状态码与切片异常回退实施文档

## 1. 背景与目标

在原工程中，批处理对称特征值求解器 `cusolver_syevj_batched()`（`src/batchase/relaxation/cusolver_batched.py`）存在严重的未收敛静默污染漏洞：
1. **忽略矩阵级收敛状态**：仅检查了 C API 的整体调用返回值 `status == CUSOLVER_STATUS_SUCCESS`，未检查针对各个矩阵独立的 `info` 数组。
2. **静默使用错误特征向量**：按照 NVIDIA 官方 cuSOLVER 规范，当某个矩阵的 Jacobi 迭代未能在 `max_sweeps` 步内收敛时，API 状态返回 0（成功），但对应矩阵的 `info[i] > 0`。此时 cuSOLVER 在输出中残留未收敛的垃圾特征值和特征向量（实测相对代数残差高达 3.18，正常值应 $< 10^{-12}$）。优化器若静默使用该特征向量，将产生严重错误或发散的搜索方向。
3. **缺少数值非有限保护**：未对特征分解输出中可能存在的非有限值（NaN/Inf）进行前置兜底。

本任务的目标是建立完善的 cuSOLVER 状态校验机制，对失败切片进行独立捕获与 `torch.linalg.eigh` 自动回退，并在不可恢复时输出上下文异常，彻底杜绝错误特征向量的静默使用。

---

## 2. 核心架构设计与修改细节

### 2.1 状态码与数值非有限联合检查 (`_recover_failed_slices`)

在 `src/batchase/relaxation/cusolver_batched.py` 中新增 `_recover_failed_slices` 恢复算子：
```python
def _recover_failed_slices(
    A: torch.Tensor,
    eigenvalues: torch.Tensor,
    eigenvectors: torch.Tensor,
    info: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    finite_values = torch.isfinite(eigenvalues).all(dim=-1)
    finite_vectors = torch.isfinite(eigenvectors).all(dim=-1).all(dim=-1)
    nonfinite_mask = ~(finite_values & finite_vectors)
    failed_mask = info.ne(0) | nonfinite_mask

    if not failed_mask.any().item():
        return eigenvalues, eigenvectors
```
- **双重检查**：同时检查 `info.ne(0)`（算法收敛失败/参数非法）与 `nonfinite_mask`（输出含 NaN/Inf）。
- **快路径零冗余**：正常情况下（99.99%+）`not failed_mask.any().item()` 直接原位返回，额外开销极低（实测仅约 0.12 ms/次）。

### 2.2 失败切片精准提取与 PyTorch eigh 批量回填

一旦检测到失败切片：
1. **聚合日志输出**：记录总批次大小、失败切片索引列表、`info` 错误码及非有限索引。
2. **提取失败切片**：使用 `A.index_select(0, failed_indices)` 仅取出异常矩阵。
3. **回退计算与校验**：
   ```python
   fallback_values, fallback_vectors = torch.linalg.eigh(failed_matrices)
   fallback_is_finite = (
       torch.isfinite(fallback_values).all()
       and torch.isfinite(fallback_vectors).all()
   )
   if not fallback_is_finite:
       raise RuntimeError("torch.linalg.eigh returned non-finite values")
   ```
4. **批量原位回填**：
   使用 `eigenvalues.index_copy_(0, failed_indices, fallback_values)` 和 `eigenvectors.index_copy_(0, failed_indices, fallback_vectors)` 原位更新，未发生异常的矩阵保留 cuSOLVER 的极速解算结果。

### 2.3 异常语义明确，杜绝伪造假数据

如果回退切片在 `torch.linalg.eigh` 中依然抛出异常或产生非有限值，向外抛出明确的 `RuntimeError`，包含失败索引和 info 状态码上下文信息：
```text
cusolver_syevj_batched fallback failed for indices=[...], info=[...]: ...
```
绝不静默填充伪造数据（如单位阵），确保科学计算的严密性与错误追溯性。

---

## 3. 测试覆盖与验证

1. **单元测试 (`tests/test_cusolver_info.py`)**：
   - `test_success_path_does_not_call_fallback`：验证正常切片快路径不触发回退函数。
   - `test_info_code_recovers_only_failed_slice`：验证仅特定 `info != 0` 的切片被提取并回退，其余切片不受影响，警告日志包含准确索引。
   - `test_nonfinite_output_recovers_even_with_zero_info`：验证非有限输出在 info 为 0 时亦能被准确捕获并恢复。
   - `test_fallback_failure_raises_with_slice_context`：验证回退失败时抛出包含切片上下文的明确 `RuntimeError`。
   - `test_native_cusolver_normal_batch_residual`：在真实 CUDA 环境下验证 cuSOLVER 特征分解相对残差 $\|AV - V\Lambda\| / \|A\| < 10^{-11}$。
   - `test_native_nonfinite_input_raises_clear_error`：验证 GPU 下输入 NaN 矩阵能被严格拦截并报错。
   - 结果：**6/6 全部通过 (0.61s)**。

2. **全量测试套件历史回归**：
   - 包含任务 1.1（三态状态机与失败隔离）、任务 1.2（LBFGS 原生算子）、任务 1.3（CPU BFGS 零步长掩码）、任务 1.4（cuSOLVER info 检查与回退）及 Worker 端到端测试：
   - 运行：`python -m unittest discover -s tests -p "test_*.py"`
   - 结果：**21/21 全部通过，零失败零报错 (3.25s)**。

3. **静态检查与编译自检**：
   - `python -m compileall src/ tests/`：100% 编译通过。
   - `git diff --check`：输出干净，0 格式违规。
