"""Paired development-set comparisons after every complete validation run."""
from .common import *

def main():
    root=RUN/'validation';variants=['A','B','B_POOL','C'];summaries={};rows={};digests={}
    for v in variants:
        folder=root/v;done=read(folder/'COMPLETE.json')
        assert done['sequences']==132 and done['frames']==47200
        assert sha(folder/'metrics/SUMMARY.json')==done['summary_sha256']
        assert sha(folder/'metrics/SEQUENCES.json')==done['sequences_sha256']
        summaries[v]=read(folder/'metrics/SUMMARY.json')
        rows[v]={r['sequence']:r for r in read(folder/'metrics/SEQUENCES.json')}
        digests[v]=done
    assert read(root/'B_POOL/PARITY_B.json')['exact_prediction_equality']
    names=sorted(rows['A']);assert all(sorted(rows[v])==names for v in variants)
    rng=np.random.default_rng(20261003);indices=rng.integers(0,len(names),(10000,len(names)))
    comparisons={}
    for new,base in [('B','A'),('C','B_POOL'),('C','A')]:
        metrics={}
        for key in ['AUC','Precision20','cAUC','nPrecision_AUC','mACC','visible_hit50']:
            d=np.array([rows[new][n][key]-rows[base][n][key] for n in names])
            assert np.isfinite(d).all()
            ci=np.quantile(d[indices].mean(1),[.025,.975])
            metrics[key]=dict(mean_delta_pp=float(d.mean()*100),paired_sequence_bootstrap_95ci_pp=(ci*100).tolist(),
                better=int((d>1e-12).sum()),worse=int((d < -1e-12).sum()),tied=int((np.abs(d)<=1e-12).sum()),
                median_delta_pp=float(np.median(d)*100),leave_one_sequence_out_range_pp=[float(((d.sum()-d)/(len(d)-1)).min()*100),float(((d.sum()-d)/(len(d)-1)).max()*100)])
        comparisons[f'{new}_minus_{base}']=metrics
    result=dict(scope='TRAIN_DEVELOPMENT_VAL132',summaries=summaries,comparisons=comparisons,input_digests=digests,
        B_POOL_exactly_matches_B=True,no_test_inference=True,automatic_promotion=False,
        interpretation='Candidate-pool construction and learned selection are separated by the B_POOL control. C changes its trajectory, so actual numerical candidates need not remain identical to B_POOL.',
        limits=['One seed and fixed training budgets, no blind Test result.',
                '132 validation sequences were already used for YOLO model selection.',
                'Sequence bootstrap assumes independent sequences; source-level dependence is not resolved.',
                'C also includes unmatched handling and confirmed anchor replacement; its delta measures the integrated C mechanism.'])
    out=RUN/'comparison';save(out/'COMPARISON.json',result)
    lines=['# LoRAT B C 开发验证结果','',
        '所有结果来自 Train 内固定划分的 132 条完整开发验证序列，共 47200 帧。该划分此前用于 YOLO 选权重；本轮没有运行 Test420。以下是单种子、固定训练预算的首轮结果，不能视为稳定收益或盲测结论。','',
        '| 版本 | AUC | P20 | NP AUC | mACC | 在线 FPS |','| --- | ---: | ---: | ---: | ---: | ---: |']
    for v,s in summaries.items():
        m=s['sequence_macro'];lines.append(f"| {v} | {m['AUC']*100:.3f} | {m['Precision20']*100:.3f} | {m['nPrecision_AUC']*100:.3f} | {m['mACC']*100:.3f} | {s['online_tracker_fps']:.2f} |")
    lines+=['','A 为原始权重的 V3 输出分离基线；B 为 UAV 微调；B_POOL 增加与 C 相同构造方式的候选池及特征提取，但保留 B 的选择与状态；C 使用学习比较、未匹配输出和连续确认后的搜索位置替换。B_POOL 与 B 的全部预测文件已通过逐文件一致性检查。C 改变跟踪轨迹，因此两者后续帧的实际候选数值可能不同。','',
        '## 配对变化','', '| 比较 | AUC 变化百分点 | 按序列 Bootstrap 95% 区间 | 更好 更差 持平 |','| --- | ---: | --- | --- |']
    for name,metrics in comparisons.items():
        a=metrics['AUC'];lo,hi=a['paired_sequence_bootstrap_95ci_pp']
        lines.append(f"| {name} | {a['mean_delta_pp']:+.3f} | [{lo:+.3f}, {hi:+.3f}] | {a['better']} / {a['worse']} / {a['tied']} |")
    b=comparisons['B_minus_A']['AUC'];c=comparisons['C_minus_B_POOL']['AUC']
    lines+=['',f"B 的开发验证 AUC 变化为 {b['mean_delta_pp']:+.3f} 个百分点；C 相对 B_POOL 的变化为 {c['mean_delta_pp']:+.3f} 个百分点。区间跨零表示当前序列样本未提供明确的平均收益证据。区间只反映按序列重采样的不确定性，未覆盖训练种子和视频共同来源的影响。",'',
        '## 诊断与复查','',
        '各版本 SUMMARY.json 同时提供微小目标命中、严重偏移、重现后连续三帧命中、候选池覆盖、存在正确候选时仍未匹配或选错、全局提交命中等诊断。它们用于区分表征、候选覆盖和选择失败；不能把几何关联标签当作人工身份标注。', '',
        f"完整数值：[COMPARISON.json]({out/'COMPARISON.json'})。逐序列指标与完整逐帧轨迹位于各版本 validation 目录。原 V4 权重与结果未被替换，当前脚本不会自动晋升新模型。",'',
        '指标调用已安装的官方 MMOT 函数，主指标含第一帧 GT 初始化；可见帧诊断排除第一帧。在线 FPS 包含逐帧同步 YOLO、候选探测与比较头，不含图像读取和模型启动。', '']
    (out/'REPORT.zh.md').write_text('\n'.join(lines))
    save(out/'COMPLETE.json',dict(comparison_sha256=sha(out/'COMPARISON.json'),report_sha256=sha(out/'REPORT.zh.md')))
    print('BC_COMPARISON_COMPLETE',out,flush=True)
if __name__=='__main__':main()
