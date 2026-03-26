# VOC 目标检测任务

基于 VOC2012 数据集的 YOLO 目标检测，支持昇腾 NPU。

## 文件说明

- `voc_detect.yaml` — 训练配置（模型、优化器、数据增强等）
- `voc_data.yaml` — 数据集配置（路径、类别名称、标注转换脚本）

## 准备工作

在 `voc_data.yaml` 中将 `path` 设为数据集根目录：

```yaml
path: /path/to/your/datasets
```

目录结构如下：

```
<path>/
  VOC2012/
    Annotations/       ← XML 标注文件
    ImageSets/Main/    ← train.txt、val.txt
    JPEGImages/        ← .jpg 图片
  images/
    train2012/         ← 首次运行后自动填充
    val2012/
  labels/
    train2012/         ← 首次运行后自动生成 YOLO .txt 标注
    val2012/
```

`voc_data.yaml` 中的 `download:` 脚本会在首次运行时自动将 VOC XML 标注转换为 YOLO 格式，后续运行跳过已转换的文件。


## 昇腾 NPU 训练

需要安装 `torch_npu` 和 CANN 工具包。

```bash
# 单 NPU
yolo train cfg=tasks/voc/voc_detect.yaml data=tasks/voc/voc_data.yaml device=npu:0

# 多 NPU DDP（hccl 后端，ultralytics 内部自动调用 torchrun）
yolo train cfg=tasks/voc/voc_detect.yaml data=tasks/voc/voc_data.yaml device=npu:0,npu:1
```

注意事项：
- NPU 上 AMP 自动禁用（不支持 GradScaler）
- NPU 上 `workers` 自动设为 0
- DDP 使用 HCCL 后端（非 NCCL）

## 验证

```bash
yolo val model=runs/detect/exp/weights/best.pt data=tasks/voc/voc_data.yaml

# 在 NPU 上验证
yolo val model=runs/detect/exp/weights/best.pt data=tasks/voc/voc_data.yaml device=npu:0
```

## 导出与模型转换

### 第一步：导出为 ONNX

```bash
yolo export model=runs/detect/exp/weights/best.pt format=onnx opset=12 simplify=True
# 输出：runs/detect/exp/weights/best.onnx
```

### 第二步：ONNX → 昇腾 OM（使用 atc）f

ATC 包含在 CANN Toolkit 中，路径通常为 `/usr/local/Ascend/ascend-toolkit/latest/bin/atc`。

根据目标硬件选择 `--soc_version`：

```bash
# Ascend 910B4（训练卡，ModelArts 默认）
atc --model=best.onnx \
    --framework=5 \
    --output=best_910b4 \
    --soc_version=Ascend910B4 \
    --input_shape="images:1,3,640,640" \
    --log=error

# Ascend 910B3
atc --model=best.onnx \
    --framework=5 \
    --output=best_910b3 \
    --soc_version=Ascend910B3 \
    --input_shape="images:1,3,640,640" \
    --log=error

# Ascend 310P（推理卡，Atlas 300I/500）
atc --model=best.onnx \
    --framework=5 \
    --output=best_310p \
    --soc_version=Ascend310P3 \
    --input_shape="images:1,3,640,640" \
    --log=error
```

输出为 `best_<device>.om`，可用 `ais_bench` 或 `acl` 进行推理。

> `--input_shape` 中的名称 `images` 需与 ONNX 模型的输入节点名一致，可用 Netron 查看确认。

## TensorBoard

```bash
# 启用一次即可
yolo settings tensorboard=True

# 训练中或训练后查看
tensorboard --logdir runs/detect --port 6006
```

完整 NPU 迁移说明见 [`migration_npu.md`](../../migration_npu.md)。
