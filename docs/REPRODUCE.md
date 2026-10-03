# 原版 V4 的运行与恢复边界

## 原机环境

- LoRAT Python：`/home/xu/miniconda3/envs/mvlm_cvpr26_cuda128/bin/python`。
- YOLO Python：`/home/xu/miniconda3/envs/yolo26_rtx5090/bin/python`。
- 正式环境快照见 `results/test420/ENVIRONMENT.json`；未在本次上传中重建环境或重新训练模型。
- 冻结核心入口依赖原机 `/home/xu/zyx/send/` 下的适配器、LoRAT 源码与权重，以及原有 MMOT 路径。`provenance/SOURCE_MANIFEST.json` 记录逐文件原路径。

## 恢复同样的原机目录

`python3 tools/restore_sources.py` 默认仅预览。加 `--apply` 时仅创建缺失文件，已存在且哈希一致则跳过；任何不同内容均报错，绝不自动覆盖。该脚本只恢复清单中的源码和辅助文件，不安装环境、不下载数据、不恢复权重。

从私有版本附件获取 LoRA 和 YOLO 权重后，核对 `provenance/WEIGHTS.json`，分别放回清单中的原位置。YOLO 附件名 `yolo26_uav_best.pt` 对应原文件名 `best.pt`。DINOv2-L 骨干约 1.22 GB，未再次上传，原下载文件的校验值已经记录。

迁移到其他用户名或其他目录时，应在新的适配分支处理路径，不要直接改写本标签下的冻结算法。修改路径后的运行一致性尚未在本次上传中验证。

## 执行

单序列：`bash tracking/run_tracker_v4.sh --video VIDEO --init-box X Y W H --output NEW_OUTPUT`。

全量：`bash tracking/run_test420_v4.sh --output NEW_OUTPUT --dataset-root TEST_ROOT --frames-root FRAMES_ROOT`。

这会真实逐帧推理，不是离线预测回放；首次运行前必须满足原环境和依赖。全量入口有 `--dry-run`、`--evaluate-only` 与逐序列续跑功能。不要将原 V4 正式目录作为新实验输出。

模型只获得首帧指定框和当前图像，后续 GT 在评测阶段读取。测试图像和 GT 不随仓库分发。预测附件仅含算法输出，无输入图片或 GT。

## 可复核内容

`tools/verify_snapshot.py` 核对仓库源码和辅助文件 SHA-256。预测附件里的 `MANIFEST.json` 可核对 420 条和 476221 帧。版本附件校验值记录在 `provenance/RELEASE_ASSETS.json`。

本次上传完成的是版本保存、文件校验和代码级检查，不代表在新环境重新运行过 Test420。
