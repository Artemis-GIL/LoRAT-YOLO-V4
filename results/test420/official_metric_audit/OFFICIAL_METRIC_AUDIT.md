# D Test420 官方函数评分复核

状态：**PASS**。仅 CPU 离线读取已完成预测和 GT，对 420 条、476221 帧重新调用本地安装的官方 MMOT 函数。6313 项逐序列及汇总标量比对，容差 1e-12，最大绝对差 **0.0**。未重新模型推理，原指标和预测 SHA 复核未变。

本地 MMOT 路径：`/home/xu/zyx/GOT_REMOTE/_external/Awesome-Multimodal-Object-Tracking/MMOT_Evaluation_Toolkit`；仓库 `https://github.com/983632847/Awesome-Multimodal-Object-Tracking.git`，提交 `bdd579fd3363e2bea51c89dd3b337f0427d291e4`。`uavantiuav_overall.py`、`got10k/utils/metrics.py`、`accuracy/accuracy.py` 均与该提交及本次冻结清单一致。本审计没有访问 GitHub HEAD，不能声称是 GitHub 当前最新版。

| 指标 | 重算值 |
|---|---:|
| AUC | 0.494759132028955 |
| Precision20 | 0.675601705833576 |
| cAUC | 0.492137165519218 |
| nPrecision_at_0_2 | 0.634513943828808 |
| nPrecision_AUC | 0.578832632681368 |
| mACC | 0.502395076093191 |
| visible_mean_iou | 0.566453205002489 |
| visible_hit50 | 0.695211550526983 |

AUC 是每序列 IoU 严格大于 0、0.05、…、1 共21个阈值的成功率均值，再对420序列等权平均；P20 是中心误差不超过20像素的比例。NP AUC 是按GT宽高分别归一化中心误差后，在0到0.50的51阈值精度均值，区别于NP@0.20。cAUC 使用工具包原CIoU实现。

主曲线与 mACC 均包含首帧GT初始化及缺失帧；可见诊断排除首帧，要求 absent.txt=0 且 GT 宽高为正。官方 report() 会把首框替换为GT；本次保存预测本来即等于GT，评分前做了显式核验。

mACC 使用官方 accuracy.eval：GT四值任一非零视为存在，存在时计IoU，缺失时仅 `[]` 或 `[0]` 可获缺失判定奖励。本算法一直导出四坐标，所以 `[0,0,0,0]` 也不会获奖励。GT全零帧 80930，absent.txt非零帧 80930，两种存在性标记不一致 0 帧；本次无null输出。

在线FPS和完整循环FPS采用本项目记录时间求总帧数/总时间，**不是** MMOT report() 的速度聚合方式；不应把速度定义混为一谈。可见命中率的序列宏平均与全帧合并比例也不同。

逐项证据和全部预测SHA见同目录 `OFFICIAL_METRIC_AUDIT.json`。
