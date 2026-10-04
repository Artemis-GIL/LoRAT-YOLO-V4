# B+D 运行、恢复与迁移

## 冻结来源

源代码来自正式 B+D Test420 目录的 915 个冻结快照，并按冻结哈希还原原目录层次。
`provenance/SOURCE_MANIFEST.json` 记录原路径、归档路径、内容 SHA、存储 SHA 及压缩格式。

当前机器上的工作区可能已有后续实验改动，不能用新工作区替代这些快照来声称复现相同模型。归档上传不修改原目录，也不覆盖原 V4、B、D 权重或正式结果。

## 环境与依赖

正式环境见 `results/test420/ENVIRONMENT.json`：LoRAT 为 Python 3.10.21 / Torch 2.7.1+cu128；YOLO 为独立环境 Python 3.10.20 / Torch 2.11.0+cu128、Ultralytics 8.4.100；GPU RTX 5090 D，TF32 开启。

原 Python 路径：
- `/home/xu/miniconda3/envs/mvlm_cvpr26_cuda128/bin/python`
- `/home/xu/miniconda3/envs/yolo26_rtx5090/bin/python`

冻结代码依赖原机 send 下的 LoRAT 源码和权重、YOLO 适配器、MMOT，以及数据清单。上游 `requirements.txt` 不等于这套完整环境锁；本次没有重建一个全新环境。

## 校验和恢复

```bash
python3 tools/verify_snapshot.py
python3 tools/restore_sources.py
```

第二条默认仅预览；`--apply` 也只创建缺失文件，已有内容不同则拒绝，绝不自动覆盖。可用 `--prefix /原目录前缀` 限定恢复范围。所有恢复在写入之前先完成冲突检查。不建议将所有研究元数据直接恢复到仍用于开发的工作区。

从私有 Release 获取权重和预测，运行 `sha256sum -c SHA256SUMS.txt`。`provenance/WEIGHTS.json` 指定附件名与原路径的对应关系；DINOv2-L 仍需原官方骨干，校验值已记录。

## 最小模型调用（需要已存在原机环境、依赖与权重）

```python
from pathlib import Path
from research.de_r1.tracker_de import build

# rgb0/rgb_t 必须为当前帧 uint8 HxWx3 RGB；只有首帧提供目标框。
tracker = build('D', Path('NEW_OUTPUT/model_audit'),
                checkpoint='/home/xu/zyx/send/LORAT_B_DE_R1_20261004/D/final.safetensors')
try:
    result = tracker.initialize(rgb0, [x, y, w, h])
    for rgb_t in remaining_frames_in_time_order:
        result = tracker.track(rgb_t)
        box = result['bbox']
finally:
    tracker.close()
```

这段是 API 调用示例，不是已执行的新推理。LoRAT 构造过程先加载官方 LoRAT checkpoint，再加载 B 的 UAV 微调参数，最后由 D 参与候选决策。

原 `research.de_test420_r1.worker` 会验证原训练/验证流水线和已完成标记，且绑定原输出目录；它不是通用新实验 CLI。正式全量入口和冻结逻辑保留供复核，迁移到新机器或新的实验输出目录需独立适配路径、生成对应输入与冻结清单，不能删除校验断言或绕过完成记录来重复运行。

## 数据与评测

原运行用规范 JPEG 清单，**不能用原 MP4 重新解码后直接替换**。序列 000316 原 MP4 与规范图像/GT 长度不同。归档的输入清单以 `.json.gz` 无损保存；内含首帧框与每帧路径，不含实际图像或完整 GT。

完整 Test420 使用 420 序列 / 476221 帧；主指标是官方序列宏平均，NP AUC 与 NP@0.20 分列；四坐标框不自动获得缺席奖励。预测包包含原预测文件和逐序列完成记录，可按 MANIFEST 复核帧数与 SHA。

模型训练、候选缓存、后续开发实验和外域评测均未在本次上传中重新执行。CPU 检查详细结果见 `provenance/CPU_CHECKS.json`。
