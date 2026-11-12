# OSDTrack D0 身份与指标口径

## 当前要迁移的算法

本 Windows 迁移分支固定的 OSDTrack 是 **B + 静态 D0**，不是 GitHub `b-plus-d` 标签中已发布的时序 GRU-D。

发布分支为 `codex/windows-native`，标签为 `osdtrack-d0-windows-rc1-20261112`。该 Release 自包含 LoRAT、B、静态 D0、YOLO 四个可分发权重及 `SHA256SUMS.txt`；DINOv2-L/14 仍从 Meta 官方地址下载并按下表校验。由于尚未完成目标 Windows/GPU 实机验收，本版本明确标为 RC1。

- B：UAV 数据微调后的 LoRAT-L/378，带同步当前帧 YOLO 候选和确认机制。
- D0：`ActionValue(temporal=False)` 的静态候选价值头；模型前向会屏蔽早于当前步的历史。
- 检测器仅接收当前 RGB 像素，并在独立 YOLO Python 环境中常驻运行。
- 首帧只输入一次 `xywh` 初始框；后续不读取 GT、absent、语言或未来帧。
- 首帧模板不更新。追踪状态与当帧输出框分离，未确认的候选不能写回权威锚点。
- 算法不是物理缺席分类器，`physical_presence` 保持 `unknown`；本仓库也不输出飞控指令。

## 五个必需权重

| 默认文件名 | 用途 | 字节数 | SHA-256 |
|---|---|---:|---|
| `dinov2_vitl14_pretrain.pth` | DINOv2-L 骨干 | 1,217,586,395 | `d5383ea8f4877b2472eb973e0fd72d557c7da5d3611bd527ceeb1d7162cbf428` |
| `LoRAT_large-378.bin` | LoRAT 构造器权重 | 130,101,948 | `601529735439da5d7ad0136b63e894ee02923d4694ce17d3568e333c298a06b0` |
| `LoRAT_B_UAV_final.safetensors` | B 微调权重 | 130,101,948 | `e62112dd6582552d1668130b58b0a0c3382775670ccc6c714e9b00e94dbd58bc` |
| `D0_static_final.safetensors` | 静态 D0 权重 | 1,002,996 | `0a27352804839714f7e691f60ae288ef06dc0b358209e379de156831d24871bd` |
| `yolo26_uav_best.pt` | 当前帧 YOLO | 60,285,155 | `efa76597c9912fbda8ff150208996a88597544fb5ba2e52a6ad02e4dc964b433` |

启动前必须同时核对文件大小和 SHA-256。仓库的默认目录是 `weights/`，也可以通过对应的 `OSDTRACK_*_WEIGHT` 环境变量指定绝对路径。权重本身不提交到 Git。

## 已审计的 Test420 结果

420 序列、476,221 帧的完整闭环 D0 推理汇总：

| 指标 | 序列宏平均 |
|---|---:|
| AUC | 49.6938% |
| Precision20 | 67.8241% |
| nPrecision AUC | 58.1167% |
| cAUC | 49.4311% |
| NP@0.20 | 63.7121% |
| mACC | 50.4621% |

论文表格中的 49.8050 / 67.9783 / 58.2478 来自同一批 D0 预测的共同 419 序列离线重汇总：只排除与对比方长度不一致的 `UAV-Anti-UAV_Test_000316` 后重新计算算术平均，没有重新训练或推理。

| 指标 | 共同 419 序列 |
|---|---:|
| AUC | 49.8050% |
| Precision20 | 67.9783% |
| nPrecision AUC | 58.2478% |

完整 420 汇总文件 SHA-256 为 `970ed87095fe04af2c3b12b362b5589d3635e37a204c3c1847d60d49ae91bcc9`；共同 419 离线重汇总文件 SHA-256 为 `d6275fb548da15ee01d09b4cf93426d73bbf5c4a6627035376e070a5f3f4b86d`。

完整 420 汇总随本分支保存在 `results/osdtrack_d0/test420/SUMMARY.json`。共同 419 文件记录论文横向比较口径，但本次 Windows 迁移 Release 不把完整 D0 Test420 预测包冒充为部署依赖；Release 的分发范围以 `provenance/OSDTRACK_D0_RELEASE_ASSETS.json` 为准。

## 与旧 GitHub 附件的差异

`bd-test420-20261004` 版本附件中的 `GRU_D_final.safetensors` SHA-256 为 `998a0b111232fae8ad8b6b2cdfa5ed906101c25969abf670850b7f0f01ccdd51`，它是时序 D，不能代替上表的静态 D0。Windows 迁移验收时应以 D0 权重和本文的不变量为准。
