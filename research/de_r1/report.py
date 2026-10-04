"""Paired Val132 comparisons for static, temporal and spatial B extensions."""
from .common import *
from research.bc_r1.evaluate import check_done

METRICS=('AUC','Precision20','cAUC','nPrecision_AUC','mACC','visible_hit50')
COMPARISONS=(('D0','B'),('D','D0'),('D','B'),('E','B'))


def aggregate_diagnostics(rows,summary):
    keys=sorted({k for row in rows for k in row if k.startswith('diag_')})
    totals={k[5:]:int(sum(row.get(k,0) for row in rows)) for k in keys}
    assert totals==summary['diagnostics'],'per-sequence diagnostics disagree with completed summary'
    size={}
    for tag in ('tiny_le8','small_8_16','larger_gt16'):
        f='diag_'+tag+'_frames';h='diag_'+tag+'_hit50_frames'
        subset=[r for r in rows if r.get(f,0)>0]
        frames=sum(r.get(f,0) for r in rows);hits=sum(r.get(h,0) for r in rows)
        size[tag]=dict(sequences=len(subset),frames=int(frames),hit50_frames=int(hits),
            frame_hit50=hits/frames if frames else None,
            sequence_macro_hit50=float(np.mean([r.get(h,0)/r[f] for r in subset])) if subset else None)
    events=totals.get('reappearance_events',0);failures=totals.get('reappearance_no_three_hits',0)
    return dict(totals=totals,size_strata=size,reappearance=dict(events=events,no_three_consecutive_hits=failures,
        recovery_fraction=(events-failures)/events if events else None),
        severe_offset=dict(frames=totals.get('severe_offset_frames',0),definition='Visible IoU < 0.1, not an identity-error annotation'),
        global_commit=dict(frames=totals.get('global_commits',0),hit50_frames=totals.get('global_commit_hit50',0)))


def paired_change(new,base,names,indices):
    out={}
    for key in METRICS:
        delta=np.asarray([new[n][key]-base[n][key] for n in names],float)
        assert np.isfinite(delta).all(),key
        ci=np.quantile(delta[indices].mean(1),[.025,.975]);leave=(delta.sum()-delta)/(len(delta)-1)
        out[key]=dict(mean_delta_pp=float(delta.mean()*100),paired_sequence_bootstrap_95ci_pp=(ci*100).tolist(),
            better=int((delta>1e-12).sum()),worse=int((delta < -1e-12).sum()),tied=int((np.abs(delta)<=1e-12).sum()),
            median_delta_pp=float(np.median(delta)*100),leave_one_sequence_out_range_pp=[float(leave.min()*100),float(leave.max()*100)])
    return out


def main():
    verify_baseline();cases=[c for c in read(OLD/'PROTOCOL.json')['cases'] if c['split']=='val']
    assert len(cases)==132 and sum(c['frames'] for c in cases)==47200
    summaries={};rows={};completed={};diagnostics={}
    folders={'B':OLD/'validation/B',**{v:RUN/'validation'/v for v in ('D_ZERO','D0','D','E_ZERO','E')}}
    for variant,folder in folders.items():
        done=read(folder/'COMPLETE.json');assert done['sequences']==132 and done['frames']==47200
        assert sha(folder/'metrics/SUMMARY.json')==done['summary_sha256']
        assert sha(folder/'metrics/SEQUENCES.json')==done['sequences_sha256']
        records=read(folder/'metrics/SEQUENCES.json');assert len(records)==132
        rows[variant]={r['sequence']:r for r in records};assert len(rows[variant])==132
        for case in cases:assert check_done(folder,case,None)
        summaries[variant]=read(folder/'metrics/SUMMARY.json');completed[variant]=done
        assert summaries[variant]['sequences']==132 and summaries[variant]['frames']==47200
        diagnostics[variant]=aggregate_diagnostics(records,summaries[variant])
    names=sorted(rows['B']);assert all(sorted(value)==names for value in rows.values())
    zero_parity={}
    for variant in ('D_ZERO','E_ZERO'):
        folder=folders[variant];parity=read(folder/'PARITY_B.json')
        assert sha(folder/'PARITY_B.json')==completed[variant]['parity_sha256']
        assert parity['status']=='PASS' and parity['exact_prediction_equality'] and parity['sequences']==132 and parity['frames']==47200
        assert parity['exact_state_and_anchor_equality'] and parity['exact_RGB_equality']
        for name in names:assert sha(folder/name/'predictions.txt')==sha(folders['B']/name/'predictions.txt'),(variant,name)
        for metric in METRICS:assert abs(summaries[variant]['sequence_macro'][metric]-summaries['B']['sequence_macro'][metric])<=1e-12
        zero_parity[variant]=parity
    rng=np.random.default_rng(20261004);indices=rng.integers(0,132,(10000,132))
    comparisons={f'{new}_minus_{base}':paired_change(rows[new],rows[base],names,indices) for new,base in COMPARISONS}
    speed={f'{new}_minus_{base}':dict(new_online_fps=summaries[new]['online_tracker_fps'],base_online_fps=summaries[base]['online_tracker_fps'],
        delta_fps=summaries[new]['online_tracker_fps']-summaries[base]['online_tracker_fps'],
        relative_percent=(summaries[new]['online_tracker_fps']/summaries[base]['online_tracker_fps']-1)*100) for new,base in COMPARISONS}
    result=dict(scope='TRAIN_DEVELOPMENT_VAL132',sequences=132,frames=47200,summaries=summaries,comparisons=comparisons,
        speed_comparisons=speed,diagnostics=diagnostics,zero_controls=zero_parity,input_completion_digests=completed,
        no_test_inference=True,automatic_promotion=False,
        interpretation='D0 vs B measures static scoring changes; D vs D0 estimates added temporal mechanism within matched loss/control design; D vs B is integrated temporal arm. E vs B is the independent spatial arm. Zero controls verify engineering parity, not learned benefit.',
        limits=['This development validation split was used for detector model selection and prior research attribution.',
            'One training seed and fixed budgets; bootstrap intervals cover sequence resampling, not training variability.',
            'Sequences may share source videos; capture-level independence is not verified.',
            'Trajectory-dependent proposals can differ after intervention; these are integrated causal rollouts, not a common cached candidate replay.',
            'Small-target pooled rates can be dominated by long sequences; per-sequence size averages are provided separately.',
            'IoU and geometric labels do not establish target identity; unmatched is not physical absence.',
            'Runtime comparisons include synchronous current-frame YOLO but are separate wall-clock runs and not isolated hardware benchmarks.'])
    out=RUN/'comparison';save(out/'COMPARISON.json',result)
    lines=['# B 基线上的 D / E 开发验证','',
        '固定 Train 开发验证集共 132 条完整序列、47200 帧。B 使用已经完成的冻结微调权重；D0 为同训练目标的静态对照，D 为时序候选关联，E 从 B 独立进行局部空间细化。未运行 Test420，不自动晋升任何新模型。','',
        '| 方案 | AUC | NP AUC | P20 | 在线 FPS |','| --- | ---: | ---: | ---: | ---: |']
    for v in ('B','D0','D','E'):
        s=summaries[v];m=s['sequence_macro'];lines.append(f"| {v} | {100*m['AUC']:.3f}% | {100*m['nPrecision_AUC']:.3f}% | {100*m['Precision20']:.3f}% | {s['online_tracker_fps']:.2f} |")
    lines+=['','| 比较 | AUC 变化 / 95% 区间（百分点） | NP AUC 变化 / 95% 区间（百分点） | AUC 改善 / 退化 / 持平 |','| --- | --- | --- | --- |']
    for label,metrics in comparisons.items():
        a=metrics['AUC'];n=metrics['nPrecision_AUC'];alo,ahi=a['paired_sequence_bootstrap_95ci_pp'];nlo,nhi=n['paired_sequence_bootstrap_95ci_pp']
        lines.append(f"| {label} | {a['mean_delta_pp']:+.3f} [{alo:+.3f}, {ahi:+.3f}] | {n['mean_delta_pp']:+.3f} [{nlo:+.3f}, {nhi:+.3f}] | {a['better']} / {a['worse']} / {a['tied']} |")
    lines+=['','D 相对 B 的变化包含静态打分、状态与时序机制；D 相对 D0 才是检验新增时序机制的主要对照。D0 采用相同训练损失，因此不能把 D 相对 B 的全部变化归因于时序。E 为独立空间分支，未与 D 组合。区间跨零表示本次序列样本没有给出明确平均变化证据。','',
        '| 方案 | ≤8 像素命中 / 帧数 | ≤8 像素逐序列宏平均 | 重现后未连续命中三帧 / 事件数 | IoU<0.1 帧 | 错误全局提交 / 总提交 |','| --- | --- | ---: | --- | ---: | --- |']
    for v in ('B','D0','D','E'):
        d=diagnostics[v];t=d['size_strata']['tiny_le8'];r=d['reappearance'];g=d['global_commit'];macro='—' if t['sequence_macro_hit50'] is None else f"{t['sequence_macro_hit50']*100:.2f}%"
        lines.append(f"| {v} | {t['hit50_frames']} / {t['frames']} | {macro} | {r['no_three_consecutive_hits']} / {r['events']} | {d['severe_offset']['frames']} | {g['frames']-g['hit50_frames']} / {g['frames']} |")
    lines+=['','完整 JSON 同时提供 8–16 像素和 >16 像素分层、帧汇总与逐序列宏平均，以及全部可见帧、候选覆盖、未匹配、全局提交和重现诊断。IoU<0.1 帧和提交时 IoU<0.5 均为几何诊断，不是人工身份错误标签。', '',
        'D 的 comparison.selected 记录比较头建议的候选。确认不足且 d_action=KEEP_B_WHILE_ALTERNATIVE_CONFIRMS 时，实际输出仍采用 B；因此 selected_wrong_with_hit50_candidate 统计建议选错，不能直接解读为已经接管或实际输出错误。', '',
        'D_ZERO、E_ZERO 均完成 132 条序列的预测文件 SHA256 与旧 B 逐文件一致性核验，并逐帧核对控制状态、提交、搜索位置和 RGB 摘要完全一致；时延、探测数与新增日志不参与一致性判定。它们检验新增代码在零作用时保持 B 行为；不计为训练收益。在线速度分别为 '+', '.join(f"{v} {summaries[v]['online_tracker_fps']:.2f} FPS" for v in ('D_ZERO','E_ZERO'))+'。','',
        '指标复用已安装的官方 MMOT 函数，主指标含第一帧 GT 初始化，可见帧诊断排除第一帧；NP 主指标为归一化精度曲线 AUC。每帧同步调用当前帧 YOLO，在线 FPS 包含探测、关联及头部计算，排除图像读取和模型启动。','',
        '这是开发验证：该划分此前用于检测器选权重和算法归因，不能当作 Test420 成绩。单种子固定预算与按序列 Bootstrap 不覆盖训练种子和共同视频来源的不确定性；小目标样本规模有限。','',
        f"完整结果：[COMPARISON.json]({out/'COMPARISON.json'})。",'']
    (out/'REPORT.zh.md').write_text('\n'.join(lines))
    save(out/'COMPLETE.json',dict(status='PASS',comparison_sha256=sha(out/'COMPARISON.json'),report_sha256=sha(out/'REPORT.zh.md'),
        sequences=132,frames_per_variant=47200,zero_controls_exact=True,automatic_promotion=False))
    print('DE_COMPARISON_COMPLETE',out,flush=True)


if __name__=='__main__':main()
