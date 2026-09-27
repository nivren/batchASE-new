# Task 2.5 — 隐藏或拒绝未实现后端

## Background & Goal

`docs/debug_opt_report.md` (lines 208, 309) 指出：
> **D5 占位后端**：配置暴露不可用后端会造成运行时失败。更新优先级：**P1**。
> 任务 2.5：隐藏或拒绝未实现后端。

### 原有问题分析

1. **静默实例化与延迟崩溃**：
   在 `src/batchase/potentials/__init__.py` 中，`create_backend("sevennet")`、`create_backend("chgnet")` 与 `create_backend("matris")` 均能正常被实例化。但在进入几何弛豫循环，首次调用 `predict()` 或 `build_inputs()` 时，才突然抛出未实现异常，造成昂贵的前期计算与流程中断。
2. **CLI 误导暴露未完成选项**：
   在 `scripts/batch_relax.py` 中，`--model` 参数文档写有 `help="MLIP model backend (mace, sevennet, chgnet, matris)"`，让使用者误以为这些 MLIP 模型已可在生产环境运行。

本任务目标：在工厂函数入口处实现快速失败（fail-fast）拦截与清晰引导，收敛 CLI 参数暴露，防止静默实例化与运行时延迟崩溃。

## Core Design & Changes

### 1. `create_backend` 快速失败与明确指引 — `src/batchase/potentials/__init__.py`

```python
SUPPORTED_BACKENDS = ("mace",)
UNIMPLEMENTED_BACKENDS = ("sevennet", "chgnet", "matris", "matgl")


def create_backend(backend: str = "mace", **kwargs) -> BatchPotential:
    b_name = backend.lower()
    if b_name in SUPPORTED_BACKENDS:
        if b_name == "mace":
            return MACEBatchBackend(**kwargs)
    elif b_name in UNIMPLEMENTED_BACKENDS:
        raise NotImplementedError(
            f"Backend '{backend}' is currently a placeholder/under active development "
            f"and not yet functional in batchASE. Currently supported backends: {list(SUPPORTED_BACKENDS)}"
        )
    else:
        raise ValueError(
            f"Unknown potential backend: '{backend}'. Supported: {list(SUPPORTED_BACKENDS)}"
        )
```

- 若请求已实现的有效后端（`mace`），正常实例化并返回。
- 若请求开发中占位存根（`sevennet`、`chgnet`、`matris`、`matgl`），立刻抛出 `NotImplementedError`，明确提示目前处于开发中并列出当前可用后端列表。
- 若请求未知模型，抛出包含有效列表的 `ValueError`。

### 2. 占位类直接实例化保护

在 `SevenNetBatchBackend`、`CHGNetBatchBackend`、`MatRISBatchBackend` 的 `__init__` 中直接抛出 `NotImplementedError`，避免绕过工厂函数产生虚假可用对象。

### 3. CLI 选项收敛 — `scripts/batch_relax.py:52`

将 `--model` 的 `choices` 严格限定为 `["mace"]`，并将帮助信息修正为 `(currently supported: mace)`。

## Tests & Verification

### 新增测试 — `tests/test_backend_registry.py`

| 测试方法 | 覆盖点 |
|---|---|
| `test_supported_backend_creation` | 验证 `create_backend('mace')` 正常创建可用实例且 `kind='mace'` |
| `test_unimplemented_backends_fail_fast` | 验证对 `sevennet`、`chgnet`、`matris`、`matgl` 调用立即抛出包含可用列表的 `NotImplementedError` |
| `test_unknown_backend_raises_value_error` | 验证未知模型名称抛出 `ValueError` |
| `test_stub_classes_direct_instantiation_raises` | 验证直接调用未完成类的构造函数时立即抛出 `NotImplementedError` |

### 回归与代码质量

- 专项测试：4/4 通过
- 历史测试：35/35 通过
- 全套测试：**39/39 通过**
- `git diff --check`：通过
