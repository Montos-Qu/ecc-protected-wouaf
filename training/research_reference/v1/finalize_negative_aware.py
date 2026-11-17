"""Human-readable delivery and integrity manifest; standard library only."""
import hashlib
import html
import json
from pathlib import Path
import sys

root=Path(sys.argv[1])
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def save(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')

assert json.loads((root/'pipeline_status.json').read_text())['state']=='complete'
audit=json.loads((root/'independent_delivery_audit.json').read_text());assert audit['complete']
assert json.loads((root/'additional_source_integrity.json').read_text())['complete']
assert sha(root/'audit_negative_aware.py')==json.loads((root/'audit_protocol_lock.json').read_text())['audit_script_sha256']
plan=json.loads((root/'plan.json').read_text());selection=json.loads((root/'selection_locked.json').read_text())
results=json.loads((root/'summary.json').read_text());primary=next(r for r in results if r['primary'])
paired=json.loads((root/'paired_correct_changes.json').read_text())
bad=json.loads((root/'remaining_bad_cases.json').read_text())
old=primary['models']['candidate300'];new=primary['models']['negative_aware']
passed=primary['meets_predeclared_candidate_criteria']
if passed:verdict='这轮出现值得继续验证的新候选：正确恢复提高，观察到的错误接受减少到0。'
elif new['correct']>old['correct'] and new['bad']>0:verdict='恢复能力有提升，但错误归属尚未解决，未达到本轮预定采用条件。'
else:verdict='本轮未达到预定采用条件；完整保留训练和评估结果，现用配置不替换。'
def table(k):
    data=next(r for r in results if r['k']==k)
    text=f'<h3>BCH(31,{k})＋整体奇偶校验</h3><table><tr><th>提取器</th><th>PNG 正确/1024</th><th>JPEG444 正确/1024</th><th>JPEG420 正确/1024</th><th>坏接受记录</th></tr>'
    names={'old200':'原+200','candidate300':'原+300候选','negative_aware':'本轮联合训练'}
    for model in ['old200','candidate300','negative_aware']:
        row=data['models'][model];items=[names[model]]+[row['correct_by_condition'][c] for c in ['png8_control','jpeg_q90_444','jpeg_q90_420']]+[row['bad']]
        text+='<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in items)+'</tr>'
    return text+'</table>'
detail_table='<table><tr><th>主方案与原+300同图比较</th><th>新恢复成功</th><th>原成功、本轮丢失</th><th>净变化</th></tr>'
for row in paired:
    if row['k']==11 and row['reference']=='candidate300':
        detail_table+='<tr>'+''.join('<td>'+html.escape(str(row[n]))+'</td>' for n in ['condition','newly_correct','lost_correct','net_correct_difference'])+'</tr>'
detail_table+='</table>'
note=(f'实际完成2400步训练，按固定开发集排序选定第{selection["selected_step"]}步。前瞻评估主方案正确恢复记录：'
      f'{old["correct"]}→{new["correct"]}；坏接受记录：{old["bad"]}→{new["bad"]}。')
boundaries=[
 '单条件每类1024源图。坏接受包含无水印生成图/自然照片被认成用户，以及带水印图认错用户；总数含同图三种压缩版本，不能解释为独立总体FPR。观察到0错也不是零风险保证。',
 '最后4096张源图是在模型与门槛锁定后生成或下载的，按同一图像、同一真指纹对照三个模型；真用户排除本轮4456图开发集合的用户。latent2420..3443曾进入早期baseline5000评估，因此这是针对本轮适配的前瞻对照，不是完全未使用过的外部测试。',
 '这是整套联合训练与原+200/+300的比较，尚未通过同训练量的正样本对照拆分各训练目标的贡献，不能把全部收益归因于单独的存在性输出。相关训练条件与权重已预先锁定。',
 '全部2400个训练批次只来自fit0/1并使用同内容配对；六个检查点只在selection2排序；calibration3随后锁门槛。最后验收未用于改参数或换检查点。完整54组计数及55296条归属决策核验通过，所选网络96个真实图片视图重放通过。',
 '生成器和映射网络仍是baseline3000；软件包没有安装或升级。当前工作提取器/门槛未自动替换。RESUME_REVIEW.ipynb调用原冻结环境归档并加载本轮所选权重，不会重新训练或生成图片。',
 '逐样本NPZ保留item_id、sample_id、condition、logits、probabilities、pred_bits、presence、候选用户与最终决策；gt_bits_valid只在带水印样本为True，无水印样本的零填充GT不是实际标签，其bit_errors为-1。',
 ('下一步：优先在真正未使用的新内容上复核锁定候选，并做同训练量正样本对照以确认收益来源。' if passed else '下一步：先检查按来源分组的剩余错误和正确保留，不在本次验收集上继续调门槛，也不自动替换工作配置。')
]
if (root/'generator_backend_diagnostic.json').exists():
    diagnostic=json.loads((root/'generator_backend_diagnostic.json').read_text())
    assert sum(not r['cudnn_deterministic'] and r['pixel_exact'] for r in diagnostic)==3
    boundaries.insert(-1,'执行修复记录：最初将训练用cudnn.deterministic=True带入生成器，导致参考图少量1灰度级差异，流程在验收图片生成前停止。诊断后恢复原生成器的False设置，12张参考图逐像素一致，再复用原锁定用户/图片和已缓存照片继续。训练模型、门槛、软件版本均未改变；原失败日志和修复脚本一并保留。')
css='<style>body{font-family:Arial,"Microsoft YaHei",sans-serif;max-width:1320px;margin:24px auto;padding:20px;color:#17394e;background:#f3f8fc}table{width:100%;border-collapse:collapse}td,th{border:1px solid #b0c7d6;padding:9px}th{background:#dcebf5}p{line-height:1.7}</style>'
original=(root/'RESULTS_zh.html').read_text()
if not (root/'METHOD_AND_ORIGINAL_RESULTS_zh.html').exists():
    (root/'METHOD_AND_ORIGINAL_RESULTS_zh.html').write_text(original)
page='<!doctype html><html lang="zh"><meta charset="utf-8"><title>负样本感知训练结果</title>'+css+'<h1>'+verdict+'</h1><p>'+note+'</p>'+table(11)+table(16)+detail_table+''.join('<p>'+html.escape(x)+'</p>' for x in boundaries)+'</html>'
(root/'RESULTS_zh.html').write_text(page)
brief='<div class="negativebrief"><style>.negativebrief{background:#f3f8fc;padding:18px}.negativebrief *{color:#17394e!important}.negativebrief table{width:100%;border-collapse:collapse}.negativebrief td,.negativebrief th{border:1px solid #b0c7d6;padding:8px}.negativebrief th{background:#dcebf5}</style><h2>'+verdict+'</h2><p>'+note+'</p>'+table(11)+table(16)+'<p>每个条件、每类1024张源图；坏接受计数含同图压缩版本。最后批次未用于调参，但latent曾进入最早baseline评估，观察到0错不等于总体零风险。</p><p>54组计数、55296条归属决策、96个图片视图重放通过。权重、输出、源图与恢复入口已保存Drive；原工作配置保留。</p></div>'
(root/'BRIEF_zh.html').write_text(brief)
resume='# 负样本感知联合训练：从这里继续\n\n'+verdict+'\n\n'+note+'\n\n'+'\n\n'.join(boundaries)
resume+='\n\n恢复：打开本目录RESUME_REVIEW.ipynb，连接原A100/driver580.82.07并挂载Drive，运行唯一恢复单元格。它校验FINAL_SHA256SUMS、调用原冻结环境恢复入口，严格加载selected_multitask.pth。无新软件依赖。历史大notebook不要Run all。\n'
resume+='\n核心文件：plan.json、selection_locked.json、locked_presence_policy.json、summary.json、test_cell_results.csv、independent_delivery_audit.json、paired_correct_changes.json、remaining_bad_cases.json。last_resumable_checkpoint.pth保存最后优化器/RNG状态；不自动接着训练。\n'
(root/'CONTINUE_HERE_zh.md').write_text(resume)
(root/'IN_PROGRESS_zh.md').write_text('本轮实验和独立核验已完成。最新结论见RESULTS_zh.html、CONTINUE_HERE_zh.md；恢复入口RESUME_REVIEW.ipynb。不应重复启动训练。\n')
files=sorted(p for p in root.iterdir() if p.is_file() and p.name not in ['FINAL_SHA256SUMS.txt','delivery_final.json','delivery_status.json'])
delivery={'complete':True,'root':str(root),'plan_sha256':sha(root/'plan.json'),'selected_step':selection['selected_step'],
          'selected_weight_sha256':sha(root/'selected_multitask.pth'),'locked_policy_sha256':sha(root/'locked_presence_policy.json'),
          'primary_meets_predeclared_candidate_criteria':passed,'primary_reference_correct':old['correct'],'primary_new_correct':new['correct'],
          'primary_reference_bad':old['bad'],'primary_new_bad':new['bad'],'environment_packages_unchanged':True,
          'original_working_configuration_unchanged':True,'package_file_count':len(files)+1,'verdict':verdict,
          'restoration_entry':str(root/'restore_negative_aware_review.py'),'restoration_notebook':str(root/'RESUME_REVIEW.ipynb'),
          'audit':audit,'test_source_limitations_disclosed':True}
if (root/'restore_entry_roundtrip.json').exists():
    restored=json.loads((root/'restore_entry_roundtrip.json').read_text())
    assert restored['complete']
    delivery['review_restore_entry_verified_in_existing_session']=True
    delivery['new_empty_environment_restore_not_repeated_this_run']=True
save(root/'delivery_final.json',delivery)
files=sorted(p for p in root.iterdir() if p.is_file() and p.name not in ['FINAL_SHA256SUMS.txt','delivery_status.json'])
(root/'FINAL_SHA256SUMS.txt').write_text('\n'.join(sha(p)+'  '+p.name for p in files)+'\n')
for line in (root/'FINAL_SHA256SUMS.txt').read_text().splitlines():
    expected,name=line.split('  ',1);assert sha(root/name)==expected,name
delivery['manifest_sha256']=sha(root/'FINAL_SHA256SUMS.txt')
save(root.parent/'LATEST_NEGATIVE_AWARE_TRAINING.json',delivery)
print('NEGATIVE_AWARE_DELIVERY_VERIFIED',json.dumps(delivery,ensure_ascii=False))
