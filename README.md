# LoRAT + YOLO + GRU：B+D / Test420 冻结版本

本分支保存 **B+D（GRU）**：UAV 微调 LoRAT-L/378、同步当前帧 YOLO、保留基础动作的短历史候选比较、连续确认，以及输出与可信状态提交分离。

- 分支：`b-plus-d`；版本标签：`bd-test420-20261004`。
- 对应正式运行：`/home/xu/zyx/send/LORAT_BEST_DE_TEST420_20261004/test420`。
- 反向验证关闭；固定首帧模板；不使用语言分支；不是 Mamba、E、裁剪修订或其他开发中的分支。
- 历史 `main` 和 `v4-test420-20261003` 仍保存原版 V4，本次没有改动它们。

## 完整 Test420 成绩

420 条序列、476221 帧。以下均为原始正式汇总的序列宏平均；包含首帧初始化和缺失帧。

| 指标 | 结果 |
|---|---:|
| AUC | 49.4759% |
| Precision20 / Pre | 67.5602% |
| cAUC | 49.2137% |
| nPrecision AUC | 57.8833% |
| NP@0.20 | 63.4514% |
| mACC | 50.2395% |
| 在线速度（项目累计计时） | 35.5037 FPS |
| 含读图及写入的序列循环速度 | 33.6733 FPS |

原始汇总：[SUMMARY.json](results/test420/metrics/SUMMARY.json)。评分定义及既有 CPU 独立复算：[官方评分审计](results/test420/official_metric_audit/OFFICIAL_METRIC_AUDIT.md)。
速度不是 MMOT report() 的 FPS 聚合；相对 V4 还存在 TF32 与控制路径差异。版本间增益不能全部归到 GRU，具体见[完整对照](results/test420/final_audit/COMPARISON_V4.zh.md)。

## 目录

| 路径 | 内容 |
|---|---|
| `research/bc_r1/` | B 的 UAV 微调、候选特征与基础框架；含历史 C 对照所需代码 |
| `research/de_r1/` | D 的相对质量网络、因果短历史、训练和确认替换；E 为同阶段对照，不是当前模型 |
| `research/de_test420_r1/` | 原选优、冻结输入、Test420 执行与测试代码 |
| `tracking/` | 同步 YOLO、LoRAT 封装、局部轨迹和继承的恢复控制 |
| `dependencies/` | 正式运行冻结的 LoRAT、适配器、策略及必要协议/输入元数据 |
| `third_party/` | 已使用的 MMOT 评测源码与许可 |
| `results/` | Test420、训练记录及开发集对照；不混为相同评测范围 |
| `provenance/` | 逐文件来源、SHA-256、权重、预测及附件清单 |
| `tools/` | 只读校验及默认仅预览的原路径恢复工具 |

## 权重与全部预测

私有版本附件：
[bd-test420-20261004](https://github.com/Artemis-GIL/LoRAT-YOLO-V4/releases/tag/bd-test420-20261004)

| 附件 | 用途 |
|---|---|
| `LoRAT_B_UAV_final.safetensors` | 正式 B 的 UAV 微调权重 |
| `GRU_D_final.safetensors` | 由验证集选定的 GRU-D 权重 |
| `LoRAT_large-378.bin` | 原封装构造模型时首先加载的 LoRAT 权重，之后再加载 B；不能只下载 D |
| `yolo26_uav_best.pt` | 同步全图检测器权重 |
| `BD_TEST420_PREDICTIONS.zip` | 420 份预测、逐序列完成记录、帧数及 SHA 清单 |
| `SHA256SUMS.txt` | 版本附件校验值 |

DINOv2-L 原骨干没有重复上传；原路径、大小和 SHA 已保存于 [WEIGHTS.json](provenance/WEIGHTS.json)。
不包含视频、图像、完整逐帧 GT/absent 标注、训练特征缓存、模型优化器状态或逐帧调试 traces。**协议和输入清单保留了首帧初始化框、文件路径和哈希**，不是零标注元数据包。

## 使用与限制

这是忠实保存正式实验的研究版本，不是已经在新机器验证过的一键安装包。源码原有的绝对路径与防篡改断言保持不变；大输入清单采用无损 gzip 存储，恢复时字节内容与原运行一致。

先执行 `python3 tools/verify_snapshot.py`，再阅读 [运行与恢复说明](docs/REPRODUCE.md)。原始 `worker.py` 是已有训练/验证流水线的管理入口，**不要把它直接当成可任意重跑 Test 的通用 CLI**。原正式实验目录不能用于新推理输出。

已有完整 MMOT 评分复核不等于本次重新运行过模型；本次仅上传、文件检查和 CPU 代码级检查。候选相对质量监督是当前帧 IoU，不保证实例身份或未来长期收益。当前没有物理缺失分类器。

第三方技术与代码署名、许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 和上游 LICENSE；本归档不重新许可第三方权重或数据。
