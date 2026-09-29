"""Train on train/val only; explicitly lock a run before one-time final Test."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import csv
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler
import segmentation_models_pytorch as smp
from data import sha, save_json, verify_prepared, check_leakage
from engine import Samples, checkpoint_state, evaluate, loss_value, MEAN, STD

DEFAULTS = dict(size=512, epochs=20, patience=5, batch_size=2, lr=3e-5,
                threshold=0.5, normal_fp_pixels=0, seed=20260928, device='cuda',
                selection_domain='real', alpha=None, tversky_reduction='batch', steps_per_epoch=None)


def setup(config):
    c = {**DEFAULTS, **config}
    if c['task'] not in ('crack', 'corrosion') or c['selection_domain'] not in ('real', 'synthetic'):
        raise ValueError('Invalid task or selection_domain')
    if c['size'] < 64 or c['size'] % 32 or min(c['epochs'], c['patience'], c['batch_size']) < 1:
        raise ValueError('Invalid size/epoch/batch configuration')
    if not 0 < c['threshold'] < 1 or not np.isfinite(c['lr']) or c['lr'] <= 0:
        raise ValueError('Invalid threshold or learning rate')
    if not isinstance(c['normal_fp_pixels'], int) or c['normal_fp_pixels'] < 0:
        raise ValueError('normal_fp_pixels must be a nonnegative integer')
    if c['alpha'] is not None and not 0 < c['alpha'] < 1:
        raise ValueError('alpha must be between 0 and 1')
    if c['tversky_reduction'] not in ('batch', 'image'):
        raise ValueError('Invalid Tversky reduction')
    if c['steps_per_epoch'] is not None and (type(c['steps_per_epoch']) is not int or c['steps_per_epoch'] < 1):
        raise ValueError('steps_per_epoch must be a positive integer or None')
    if c['device'] not in ('cuda', 'cpu'):
        raise ValueError('device must be cuda or cpu')
    if c['device'] == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('Colab GPU is not connected; no silent CPU fallback')
    random.seed(c['seed']); np.random.seed(c['seed']); torch.manual_seed(c['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(c['seed'])
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    return c


def new_model(weights, device):
    model = smp.Unet(encoder_name='efficientnet-b0', encoder_weights=None, in_channels=3, classes=1)
    model.load_state_dict(checkpoint_state(weights), strict=True)
    return model.to(device)


def comparison(baseline, tuned):
    result = {}
    for kind in ('real', 'synthetic'):
        b, t = baseline['groups'][kind], tuned['groups'][kind]
        delta = {key: t['positive_macro'][key]-b['positive_macro'][key]
                 if t['positive_macro'][key] is not None and b['positive_macro'][key] is not None else None
                 for key in ('iou', 'dice', 'precision', 'recall')}
        result[kind] = dict(baseline=b, finetuned=t, delta=delta)
    return result


def write_tables(out, task, split, baseline, tuned):
    out = Path(out)
    fields = ['task', 'split', 'data_type', 'model', 'n_images', 'source_groups', 'positive_n', 'normal_n',
              'iou', 'dice', 'precision', 'recall', 'normal_false_positive_images',
              'normal_false_positive_image_rate', 'normal_mean_false_positive_area']
    tables = []
    for name, result in [('baseline', baseline), ('finetuned', tuned)]:
        for kind, group in result['groups'].items():
            row = dict(task=task, split=split, data_type=kind, model=name,
                       source_groups=len({r['source_group'] for r in result['per_image'] if r['data_type']==kind}),
                       **{k: group[k] for k in fields if k in group}, **group['positive_macro'])
            tables.append(row)
    with (out/'summary.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields); writer.writeheader(); writer.writerows(tables)
    with (out/'per_image.csv').open('w', encoding='utf-8-sig', newline='') as f:
        per = [dict(model=name, **row) for name, result in [('baseline', baseline), ('finetuned', tuned)] for row in result['per_image']]
        writer = csv.DictWriter(f, fieldnames=list(per[0])); writer.writeheader(); writer.writerows(per)
    def fmt(value):
        return '해당 없음' if value is None else f'{value:.4f}' if isinstance(value, float) else str(value)
    title = 'Validation 결과' if split=='val' else '최종 Test 결과'
    lines = [f'# {task} {title}', '', '실제 사진과 합성 사진 분리 집계. IoU 등은 해당 결함이 있는 사진의 평균값임.', '',
             '|자료|모델|사진 수|원본 그룹 수|IoU|Precision|Recall|정상 오탐 비율|정상 오탐 면적 비율|',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in tables:
        lines.append('|'+'|'.join(fmt(row[k]) for k in ('data_type','model','n_images','source_groups','iou','precision','recall','normal_false_positive_image_rate','normal_mean_false_positive_area'))+'|')
    lines += ['', '빈 평가 집단의 지표는 해당 없음으로 표시함. 위 비율의 범위는 0~1임.',
              '단일 실행 결과이며 안전등급 또는 현장 성능 보장 근거가 아님.',
              'Validation은 모델 선택에 사용한 자료이며 독립 Test 성능과 구분함.' if split=='val' else '이 결과 확인 후 같은 Test에 맞춘 모델 또는 기준 조정 금지.']
    (out/'결과표.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def train(config, out):
    c = setup(config)
    root, out = Path(c['data']).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError('Use a new run directory')
    all_rows = verify_prepared(root, {'train', 'val'})
    rows = [r for r in all_rows if c['task'] in r['complete_targets']]
    splits = {s: [r for r in rows if r['split']==s] for s in ('train','val')}
    for name, members in splits.items():
        if not any(r['positive_pixels'][c['task']] for r in members) or not any(r['normal_confirmed'] for r in members):
            raise ValueError(f'{name}: positive and confirmed normal images required for this task')
    selected = [r for r in splits['val'] if r['data_type']==c['selection_domain']]
    if not any(r['positive_pixels'][c['task']] for r in selected) or not any(r['normal_confirmed'] for r in selected):
        raise ValueError('selection_domain validation needs positive and normal images')
    weights = Path(c['weights']).resolve()
    model = new_model(weights, c['device'])
    train_ds = Samples(root, splits['train'], c['task'], c['size'], True)
    val_ds = Samples(root, splits['val'], c['task'], c['size'])
    encoder_before = {k:v.cpu().clone() for k,v in model.encoder.state_dict().items()}
    for param in model.encoder.parameters(): param.requires_grad_(False)
    out.mkdir(parents=True)
    c.update(data=str(root), weights=str(weights), model='Unet/efficientnet-b0',
             encoder_frozen=True, encoder_batchnorm_frozen=True, mean=MEAN.tolist(), std=STD.tolist(),
             dataset_sha256=sha(root/'dataset.json'), baseline_sha256=sha(weights),
             torch=str(torch.__version__), smp=smp.__version__, python=platform.python_version(),
             code_hashes={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')})
    save_json(out/'config.json', c)
    # Keep the complete train/val provenance, including classes not selected for this model.
    save_json(out/'trainval_provenance.json', dict(items=all_rows))
    ev = lambda dest=None: evaluate(model, val_ds, c['device'], c['threshold'], dest, c['normal_fp_pixels'])
    baseline = ev(out/'baseline')
    save_json(out/'baseline_val.json', baseline)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=c['lr'], weight_decay=1e-4)
    sampler = (RandomSampler(train_ds, replacement=True,
                num_samples=c['steps_per_epoch'] * c['batch_size'],
                generator=torch.Generator().manual_seed(c['seed']))
               if c['steps_per_epoch'] is not None else None)
    loader = DataLoader(train_ds, batch_size=c['batch_size'], shuffle=sampler is None,
                        sampler=sampler, num_workers=0)
    best_score, best_epoch, stale, history = -1., 0, 0, []
    start = time.monotonic()
    for epoch in range(1, c['epochs']+1):
        model.train(); model.encoder.eval()
        losses = []
        for x,y,v in loader:
            x,y,v = [a.to(c['device']) for a in (x,y,v)]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_value(model(x), y, v, c['task'], c['alpha'], c['tversky_reduction'])
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss')
            loss.backward()
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
            optimizer.step(); losses.append(float(loss.detach()))
        val = ev()
        score = val['groups'][c['selection_domain']]['positive_macro']['iou']
        history.append(dict(epoch=epoch, optimizer_steps=len(losses), loss=float(np.mean(losses)), selection_iou=score, groups=val['groups']))
        save_json(out/'history.json', history)
        print(f"{c['task']} epoch={epoch} val_iou={score:.6f}", flush=True)
        if score > best_score + 1e-8:
            best_score, best_epoch, stale = score, epoch, 0
            state = {k:v.detach().cpu() for k,v in model.state_dict().items()}
            if not all(torch.isfinite(v).all() for v in state.values()): raise FloatingPointError('Nonfinite state')
            torch.save(state, out/'best.tmp'); (out/'best.tmp').replace(out/'best.pt')
        else: stale += 1
        if stale >= c['patience']: break
    for k,v in model.encoder.state_dict().items():
        if not torch.equal(v.cpu(), encoder_before[k]): raise RuntimeError('Frozen encoder changed')
    if sha(weights) != c['baseline_sha256']: raise RuntimeError('Baseline checkpoint changed')
    model.load_state_dict(checkpoint_state(out/'best.pt'), strict=True)
    tuned = ev(out/'finetuned')
    save_json(out/'finetuned_val.json', tuned)
    save_json(out/'result.json', dict(status='completed', split='val', task=c['task'], comparison=comparison(baseline,tuned),
              best_epoch=best_epoch, epochs_run=len(history), elapsed_seconds=time.monotonic()-start,
              best_sha256=sha(out/'best.pt'), config_sha256=sha(out/'config.json'),
              provenance_sha256=sha(out/'trainval_provenance.json'), test_evaluated=False, deployment_replaced=False))
    write_tables(out, c['task'], 'val', baseline, tuned)


def validate_run(run):
    run = Path(run).resolve()
    c = json.loads((run/'config.json').read_text())
    r = json.loads((run/'result.json').read_text())
    if r['status'] != 'completed': raise ValueError('Run has not completed')
    for name, expected in [('best.pt',r['best_sha256']),('config.json',r['config_sha256']),('trainval_provenance.json',r['provenance_sha256'])]:
        if sha(run/name) != expected: raise ValueError('Run artifact changed: '+name)
    if sha(c['weights']) != c['baseline_sha256']: raise ValueError('Baseline checkpoint changed')
    for name, expected in c['code_hashes'].items():
        if sha(Path(__file__).parent/name) != expected: raise ValueError('Evaluation code changed since training: '+name)
    return c, r


def lock(run, note):
    run = Path(run).resolve()
    validate_run(run)
    if not note.strip(): raise ValueError('Validation-based model-selection note required')
    path = run/'evaluation_lock.json'
    payload = dict(result_sha256=sha(run/'result.json'), note=note, locked_at=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    with path.open('x', encoding='utf-8') as f: json.dump(payload, f, ensure_ascii=False, indent=2)


def final_test(run, test_data, out, device=None):
    run, root, out = Path(run).resolve(), Path(test_data).resolve(), Path(out).resolve()
    c, result = validate_run(run)
    locked = json.loads((run/'evaluation_lock.json').read_text())
    if sha(run/'result.json') != locked['result_sha256']: raise ValueError('Locked run changed')
    if out.exists(): raise FileExistsError('Use a new final evaluation directory')
    test_rows = verify_prepared(root, {'test'})
    for row in test_rows:
        if row.get('independence_reviewed') is not True or row.get('used_for_training') is not False or row.get('used_for_model_selection') is not False:
            raise ValueError('Test independence evidence missing')
    known = json.loads((run/'trainval_provenance.json').read_text())['items']
    leakage = check_leakage(known+test_rows)
    if {r['id'] for r in known} & {r['id'] for r in test_rows}: leakage.append('Image IDs overlap')
    if leakage: raise ValueError('\n'.join(leakage))
    task_rows = [r for r in test_rows if c['task'] in r['complete_targets']]
    if not any(r['positive_pixels'][c['task']] for r in task_rows) or not any(r['normal_confirmed'] for r in task_rows):
        raise ValueError('Test requires target-positive and confirmed normal images')
    c = setup({**c, 'device':device or c['device']})
    dataset = Samples(root, task_rows, c['task'], c['size'])
    model = new_model(c['weights'], c['device'])
    # Exclusive marker is deliberately created before predictions; failed runs stay recorded.
    marker = run/'test_evaluation_started.json'
    with marker.open('x', encoding='utf-8') as f:
        json.dump(dict(test_manifest_sha256=sha(root/'dataset.json'), out=str(out), status='started'), f, indent=2)
    out.mkdir(parents=True)
    baseline = evaluate(model, dataset, c['device'], c['threshold'], out/'baseline', c['normal_fp_pixels'])
    model.load_state_dict(checkpoint_state(run/'best.pt'), strict=True)
    tuned = evaluate(model, dataset, c['device'], c['threshold'], out/'finetuned', c['normal_fp_pixels'])
    save_json(out/'baseline_test.json', baseline); save_json(out/'finetuned_test.json', tuned)
    save_json(out/'result.json', dict(status='completed', split='test', task=c['task'], comparison=comparison(baseline,tuned),
              run_result_sha256=sha(run/'result.json'), evaluation_lock_sha256=sha(run/'evaluation_lock.json'),
              test_manifest_sha256=sha(root/'dataset.json'), threshold=c['threshold'], size=c['size'],
              normal_fp_pixels=c['normal_fp_pixels'], deployment_replaced=False))
    write_tables(out, c['task'], 'test', baseline, tuned)
    save_json(run/'test_evaluation_completed.json', dict(result=str(out/'result.json'), result_sha256=sha(out/'result.json')))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest='command', required=True)
    t = sub.add_parser('train'); t.add_argument('--config', type=Path, required=True); t.add_argument('--out', type=Path, required=True)
    l = sub.add_parser('lock'); l.add_argument('--run', type=Path, required=True); l.add_argument('--note', required=True)
    e = sub.add_parser('test'); e.add_argument('--run', type=Path, required=True); e.add_argument('--data', type=Path, required=True); e.add_argument('--out', type=Path, required=True); e.add_argument('--device', choices=['cpu','cuda'])
    a = p.parse_args()
    if a.command == 'train': train(json.loads(a.config.read_text()), a.out)
    elif a.command == 'lock': lock(a.run, a.note)
    else: final_test(a.run, a.data, a.out, a.device)
