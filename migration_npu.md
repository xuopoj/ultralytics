# Ascend NPU 迁移记录

## 概述

本文档记录将 Ultralytics YOLO 训练迁移到华为昇腾 NPU（Ascend 910B4）的全部改动，包括两次提交的内容。

---

## 环境

- 硬件：Ascend 910B4 × 2
- CANN：8.3.RC1
- torch_npu：2.5.1
- Python：3.10
- 平台：ModelArts（华为云）

---

## 提交一：全面 NPU 基础支持

**提交**：`7e61f72e` — comprehensive NPU/Ascend support across training and inference

### 1. DDP 后端 — 使用 HCCL 替代 NCCL

**文件**：`ultralytics/engine/trainer.py`

NPU 使用华为自研的 HCCL（Huawei Collective Communication Library）通信后端，而非 NCCL。

```python
if self.device.type == "npu":
    torch.npu.set_device(RANK)
    self.device = torch.device("npu", RANK)
    backend = "hccl"
else:
    torch.cuda.set_device(RANK)
    self.device = torch.device("cuda", RANK)
    backend = "nccl" if dist.is_nccl_available() else "gloo"
dist.init_process_group(backend=backend, ...)
```

### 2. GradScaler — NPU 不支持，回退到 CPU

**文件**：`ultralytics/engine/trainer.py`

NPU 不支持 `GradScaler`，将 device_type 设为 `"cpu"` 以禁用 AMP scaler：

```python
device_type = self.device.type if self.device.type != "npu" else "cpu"
self.scaler = torch.amp.GradScaler(device_type, enabled=self.amp)
```

### 3. workers 和 world_size — 单 NPU 设为 0

**文件**：`ultralytics/engine/trainer.py`

```python
if self.device.type in {"cpu", "mps", "npu"}:
    self.args.workers = 0
...
elif self.args.device in {"cpu", "mps", "npu"}:
    world_size = 0
```

### 4. 内存管理 — NPU empty_cache / memory_reserved

**文件**：`ultralytics/engine/trainer.py`、`ultralytics/utils/torch_utils.py`、`ultralytics/utils/autobatch.py`

在所有 `torch.cuda.empty_cache()` 和 `torch.cuda.memory_reserved()` 调用处增加 NPU 分支：

```python
if hasattr(torch, "npu") and torch.npu.is_available():
    torch.npu.empty_cache()
else:
    torch.cuda.empty_cache()
```

### 5. 随机种子 — NPU RNG 初始化

**文件**：`ultralytics/utils/torch_utils.py`

```python
if hasattr(torch, "npu"):
    torch.npu.manual_seed(seed)
    torch.npu.manual_seed_all(seed)
```

### 6. 时间同步 — NPU synchronize

**文件**：`ultralytics/utils/torch_utils.py`

```python
elif hasattr(torch, "npu") and torch.npu.is_available():
    torch.npu.synchronize()
```

### 7. torch_distributed_zero_first — 支持 HCCL

**文件**：`ultralytics/utils/torch_utils.py`

HCCL 与 NCCL 一样需要 `device_ids` 参数：

```python
use_ids = initialized and dist.get_backend() in {"nccl", "hccl"}
```

### 8. autocast & attempt_compile — NPU 支持

**文件**：`ultralytics/utils/torch_utils.py`

```python
if use_autocast and device.type in {"cuda", "npu"}:
    dummy = dummy.half()
if use_autocast and device.type in {"cuda", "mps", "npu"}:
    with torch.autocast(device.type):
```

### 9. non_blocking — 兼容 NPU 的 `.to()` 调用

**文件**：多个 `train.py` / `val.py`

在各任务（detect、classify、world 等）的 `.to(device)` 调用中增加 NPU 的 `non_blocking` 支持。

---

## 提交二：多 NPU DDP 支持修复

**提交**：`03eba8df` — Fix Ascend NPU multi-NPU DDP training support

### 1. `device` 被覆盖导致 DDP 未触发

**文件**：`ultralytics/engine/trainer.py`

**问题**：`select_device('npu:0,npu:1')` 返回单设备 `torch.device('npu:0')`，第 130 行随即将 `self.args.device` 覆盖为 `'npu:0'`，后续 `world_size` 计算结果为 1，DDP 始终不触发。

**修复**：在 `select_device` 调用前保存原始字符串，覆盖后恢复：

```python
_device_arg = self.args.device
self.device = select_device(self.args.device)
self.args.device = os.getenv("CUDA_VISIBLE_DEVICES") if "cuda" in str(self.device) else str(self.device)
if isinstance(_device_arg, str) and "," in _device_arg:
    self.args.device = _device_arg
```

同时修复 `world_size` 计算，将单设备字符串判断内移：

```python
if isinstance(self.args.device, str) and len(self.args.device):
    _d = self.args.device.lower()
    if _d in {"cpu", "mps", "npu"}:
        world_size = 0
    else:
        world_size = len(_d.split(","))
```

### 2. 最终验证崩溃 — 硬编码 CUDA 设备

**文件**：`ultralytics/engine/validator.py`

**问题**：训练完成后的 `final_eval` 中，DDP 子进程走 `torch.device("cuda", RANK)` 分支，NPU 机器上崩溃。

**修复**：

```python
device=select_device(self.args.device) if RANK == -1 else torch.device(
    "npu" if hasattr(torch, "npu") and torch.npu.is_available() else "cuda", RANK
)
```

### 3. `select_device` 崩溃 — tuple 输入 & 多 NPU 格式

**文件**：`ultralytics/utils/torch_utils.py`

**修复**：加 `isinstance(str)` 守卫；支持 `'npu:0,npu:1'` 格式，多 NPU 时返回 `npu:0`，由 `_setup_ddp` 按 rank 重新分配：

```python
npu_devices = [d.strip() for d in device.split(",")]
if len(npu_devices) > 1:
    idx = 0  # _setup_ddp 会按 rank 重新分配
```

---

## 正确的多 NPU 启动方式

ultralytics 内部自动调用 `torchrun`，**禁止**手动调用：

```bash
# ✅ 正确
yolo train cfg=train_detect_template.yaml data=my_voc2012.yaml device=npu:0,npu:1

# ❌ 错误（process group 未初始化）
torchrun --nproc_per_node=2 $(which yolo) train ... device=npu
```

---

## 训练配置要点

- `batch: 8`（单卡），双卡等效 batch=16
- `amp`：NPU 上自动禁用（GradScaler 不支持）
- `workers`：NPU 上自动设为 0
- 详见 `train_detect_template.yaml` 和 `my_voc2012.yaml`

---

## TensorBoard

```bash
# 首次启用（只需一次）
yolo settings tensorboard=True

# 训练时自动记录，查看方式：
tensorboard --logdir runs/detect --port 6006
```

---

## 部署路径

```
训练          best.pt
    ↓ yolo export format=onnx opset=12 simplify=True
ONNX          best.onnx        （通用中间格式，可用 Netron 查看）
    ↓ atc --model=best.onnx --framework=5 --soc_version=Ascend910B4
OM            best.om          （昇腾原生格式，性能最优）
```

推理服务：FastAPI + `ais_bench.InferSession`（OM）或直接 `torch_npu`（PT）。
