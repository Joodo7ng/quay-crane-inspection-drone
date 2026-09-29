"""Five conditions: two pretrained baselines and three independent fine-tuning runs.
Input must have passed data.prepare; never auto-approve or reassign raw submissions.
"""
import argparse
import json
import math
import shutil
from collections import Counter
from pathlib import Path
from data import verify_prepared, save_json, sha
from run import train, setup, new_model, final_test, lock
from engine import Samples, evaluate

RUNS = {'crack_real': ('crack', False), 'crack_real_synthetic': ('crack', True),
        'corrosion_real': ('corrosion', False)}


def select_conditions(rows):
    result = {}
    for name, (task, mixed) in RUNS.items():
        selected = [r for r in rows if task in r['complete_targets'] and
                    ((r['split'] == 'val' and r['data_type'] == 'real') or
                     (r['split'] == 'train' and (mixed or r['data_type'] == 'real')))]
        for split in ('train', 'val'):
            part = [r for r in selected if r['split'] == split]
            if not any(r['positive_pixels'][task] for r in part) or not any(r['normal_confirmed'] for r in part):
                raise ValueError(f'{name}/{split}: target-positive and normal images required')
        result[name] = selected
    if not any(r['split'] == 'train' and r['data_type'] == 'synthetic' and r['positive_pixels']['crack']
               for r in result['crack_real_synthetic']):
        raise ValueError('Synthetic crack training data required for comparison')
    return result


def condition_data(source, destination, selected, all_rows):
    destination.mkdir(parents=True, exist_ok=False)
    for row in selected:
        for name in row['file_hashes']:
            target = destination / name
            if not target.resolve().is_relative_to(destination.resolve()):
                raise ValueError('Unsafe prepared path')
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source / name, target)
    save_json(destination/'dataset.json', dict(schema_version=1, kind='trainval', items=selected))
    save_json(destination/'all_input_provenance.json', dict(items=all_rows))


def execute(data, weights, out, epochs=20, seed=20260929, device='cuda', size=512, batch_size=2):
    root, out = Path(data).resolve(), Path(out).resolve()
    if out.exists(): raise FileExistsError('Use a new experiment directory')
    rows = verify_prepared(root, {'train','val'})
    conditions = select_conditions(rows)
    weights = {k: str(Path(v).resolve()) for k,v in weights.items()}
    # Check both original checkpoints and all image/mask inputs before starting any run.
    for task in ('crack','corrosion'):
        cfg = setup(dict(task=task, device=device, size=size, batch_size=batch_size, epochs=epochs, seed=seed))
        model = new_model(weights[task], device)
        del model
    for name, selected in conditions.items():
        Samples(root, selected, RUNS[name][0], size)
    steps = max(1, math.ceil(sum(r['split']=='train' for r in conditions['crack_real']) / batch_size))
    out.mkdir(parents=True)
    plan = dict(status='started', test_evaluated=False,
                conditions=['crack_baseline','crack_real','crack_real_synthetic','corrosion_baseline','corrosion_real'],
                input_sha256=sha(root/'dataset.json'), weights={k:sha(v) for k,v in weights.items()},
                seed=seed, rounds=epochs, crack_steps_per_round=steps,
                crack_total_updates=steps*epochs,
                note='Matched crack update budgets; sampling with replacement; best real Validation IoU selected. No Test used.',
                counts={name:dict(Counter(f"{r['split']}/{r['data_type']}" for r in rs)) for name,rs in conditions.items()})
    save_json(out/'experiment_plan.json', plan)
    save_json(out/'all_trainval_provenance.json', dict(items=rows))
    for name, selected in conditions.items():
        task, mixed = RUNS[name]
        dest = out/'condition_data'/name
        condition_data(root, dest, selected, rows)
        config = dict(task=task, data=str(dest), weights=weights[task], device=device, size=size,
                      batch_size=batch_size, epochs=epochs, seed=seed, selection_domain='real')
        if task == 'crack':
            # A round has the same update count in real-only and mixed conditions.
            config.update(steps_per_epoch=steps, patience=epochs+1)
        train(config, out/name)
    result = {}
    for name in RUNS:
        result[name] = json.loads((out/name/'finetuned_val.json').read_text())
    for task in ('crack','corrosion'):
        result[task+'_baseline'] = json.loads((out/(task+'_real')/'baseline_val.json').read_text())
    # Both crack runs must have the same baseline predictions and Validation IDs.
    other = json.loads((out/'crack_real_synthetic'/'baseline_val.json').read_text())
    if result['crack_baseline'] != other:
        raise RuntimeError('Crack baseline evaluations differ across conditions')
    save_json(out/'five_conditions_validation.json', result)
    plan['status']='completed'; save_json(out/'experiment_plan.json', plan)
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',required=True,type=Path)
    p.add_argument('--crack-weights',required=True,type=Path)
    p.add_argument('--corrosion-weights',required=True,type=Path)
    p.add_argument('--out',required=True,type=Path)
    p.add_argument('--epochs',default=20,type=int)
    p.add_argument('--seed',default=20260929,type=int)
    a=p.parse_args()
    execute(a.data,dict(crack=a.crack_weights,corrosion=a.corrosion_weights),a.out,a.epochs,a.seed)
