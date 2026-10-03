# LoRAT + YOLO V4 — UAV-Anti-UAV 跟踪

本仓库保存 **2026-10-03 完成 Test420 的原版 V4**，并非后来的关闭反向验证或其他消融版本。
核心源码由该次正式运行的 `SOURCE_SNAPSHOT.json` 重建，**866 个冻结文件逐个 SHA-256 校验一致**；没有为上传而修改定位、验证、候选关联或状态逻辑。

## 算法框架

**LoRAT 局部跟踪 → 同一当前帧 YOLO 全图候选 → 候选独立定位 → 首帧反向参考检查 → 空间消歧与连续确认 → 分别决定输出和可信位置提交。**

- 首帧图像与指定框建立固定模板；后续 GT 不参与推理。
- LoRAT-L/378：DINOv2 ViT-L/14 + LoRA（rank 64、alpha 64），模板 196，搜索 378，搜索因子 5。
- 每帧同步运行 YOLO；没有预先检测整个视频。原始候选下限 0.001，活动候选过滤 0.1，最多 12 框。
- 局部和全局候选共用正向响应、反向响应与首帧参考框重叠检查；不确定后的恢复要求唯一候选连续 3 帧支持。
- 未确认但局部坐标有效时仍输出暂定框；可信位置不提交，暂定位置可继续推进。
- 没有额外 Mamba、MVLM 语言分支或独立 DINO 身份分类器。

## 已完成的 Test420 结果

范围：**420 条完整序列，476,221 帧**。以下直接引用正式运行 `results/test420/metrics/SUMMARY.json`，不是本次上传重新推理得到。

| 指标 | 结果 |
|---|---:|
| AUC | **48.8293%** |
| Precision20 | **64.2552%** |
| cAUC | **48.4851%** |
| nPrecision 曲线 AUC | 54.5007% |
| nPrecision @ 0.2 | 59.4813% |
| mACC | **49.6193%** |
| 在线跟踪速度 | **23.9166 FPS** |

主指标为既有 MMOT 序列等权统计，包含 GT 初始化首帧。可见帧诊断排除首帧。nPrecision 两种定义分别保留，不混用。速度包含当前帧检测通信、候选定位、反向检查与状态处理，不含读图和模型启动；硬件和环境见 `results/test420/ENVIRONMENT.json`。

## 目录

| 路径 | 内容 |
|---|---|
| `tracking/` | V4 跟踪器、状态机、验证器、入口及继承的 V3/通用依赖 |
| `dependencies/LOCAL_TRACKER_SELECTION_R1_20261002/` | 冻结 LoRAT 适配器与 LoRAT 源码 |
| `dependencies/MVLM_V2_GLOBAL_PROPOSAL_BACKBONE_R1_20261001/` | 现用 YOLO26 适配器；目录名是历史名称，不代表运行 MVLM |
| `dependencies/LORAT_YOLO_GEOMETRY_R2_20261002/` | 入口导入所需的历史几何辅助依赖 |
| `third_party/MMOT_Evaluation_Toolkit/` | 本次评测使用的冻结工具包源码 |
| `results/test420/` | 正式汇总、逐序列指标、环境、配置、预测文件清单 |
| `provenance/` | 原路径映射、源码/权重/版本附件校验值、上游版本 |
| `docs/` | 架构与复现说明、历史架构原文 |
| `tools/` | 不加载模型的校验与原路径恢复辅助脚本 |

## 权重与预测结果

[版本附件：v4-test420-20261003](https://github.com/Artemis-GIL/LoRAT-YOLO-V4/releases/tag/v4-test420-20261003)

附件单独保存 `LoRAT_large-378.bin`、`yolo26_uav_best.pt`、`V4_TEST420_PREDICTIONS.zip`，不混入 Git 源码历史。420 份预测逐文件核对了原运行哈希和总帧数。

**DINOv2-L 骨干权重未上传**。请使用原本已下载的官方权重；原路径、大小和 SHA-256 见 `provenance/WEIGHTS.json`。数据集、视频、图像、GT、逐帧调试日志、缓存、登录信息不包含在本仓库。

## 运行与迁移

此仓库是**精确保真的原机环境快照，不是已经验证过的跨机器一键安装包**。冻结代码中仍有原机绝对路径；保留这些路径是为了不把上传整理与算法修改混在一起。

```bash
# 不需要加载模型，校验本仓库快照
python3 tools/verify_snapshot.py

# 在原机原依赖仍存在的环境中，可运行仓库里的单序列入口
bash tracking/run_tracker_v4.sh \
  --frames /path/to/frames --init-box 100 100 20 20 \
  --output /path/to/new_v4_output
```

环境、迁移步骤和全量评测入口见 [docs/REPRODUCE.md](docs/REPRODUCE.md)。不要覆盖正式结果目录，也不要在没有准备权重和数据的机器上把源码归档误认为完整运行环境。

## 已知边界

反向参考检查是定位一致性检查，不是可靠的实例身份认证。相似干扰物可能通过，外观变化后的正确目标可能被拒绝。暂定输出可能错误；`SEARCHING` 不表示目标已经离场。当前没有可靠的物理存在/离场判断。

此快照保留原版已知的候选连续性和暂定轨迹生命周期局限，没有混入修复。后续 Train32 消融尚未证明完整 V4 比合理的简化输出基线有稳定增量优势；本仓库不宣称所有组件都已获得独立收益证明，也不宣称 SOTA。Test420 已参与过项目归因和开发，应如实说明开发过程。

冻结 JSON 中的 `status=experimental_structural_revision_not_accuracy_validated` 是运行前历史字段；它未被改写。是否完成以本仓库保存的 `COMPLETE.json` 和正式结果为准。

## 上游与许可

LoRAT 来源于 [LitingLin/LoRAT](https://github.com/LitingLin/LoRAT)，MMOT 评测来源于 [Awesome-Multimodal-Object-Tracking](https://github.com/983632847/Awesome-Multimodal-Object-Tracking)。各自许可随代码保留；详见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。本次上传为私有研究备份，没有代替作者为整个项目选择新的公开开源许可证。
