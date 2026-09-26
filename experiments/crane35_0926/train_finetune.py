"""Fine-tune the existing binary U-Net models; never silently initialize a new model."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from prepare_data import sha, save_json

MEAN = np.array([.485, .456, .406], dtype=np.float32)
STD = np.array([.229, .224, .225], dtype=np.float32)


def metrics(pred, gt, valid):
    pred, gt, valid = [np.asarray(a, bool) for a in (pred, gt, valid)]
    tp = int((pred & gt & valid).sum())
    fp = int((pred & ~gt & valid).sum())
    fn = int((~pred & gt & valid).sum())
    tn = int((~pred & ~gt & valid).sum())
    if not valid.any():
        raise ValueError('No valid pixels')
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, target_pixels=tp+fn, valid_pixels=int(valid.sum()),
                iou=tp/(tp+fp+fn) if tp+fp+fn else 1.,
                dice=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 1.,
                recall=tp/(tp+fn) if tp+fn else None,
                precision=tp/(tp+fp) if tp+fp else (0. if tp+fn else None))


def summarize(rows):
    def avg(rs, key):
        values = [r[key] for r in rs if r[key] is not None]
        return float(np.mean(values)) if values else None
    positives = [r for r in rows if r['target_pixels'] > 0]
    normals = [r for r in rows if r['normal_confirmed']]
    return dict(n_images=len(rows), positive_n=len(positives), normal_n=len(normals),
                positive_macro={k: avg(positives, k) for k in ('iou', 'dice', 'recall', 'precision')},
                all_macro={k: avg(rows, k) for k in ('iou', 'dice', 'recall', 'precision')},
                normal_false_positive_images=sum(r['fp'] > 0 for r in normals),
                normal_mean_false_positive_area=float(np.mean([r['fp']/r['valid_pixels'] for r in normals])) if normals else None,
                per_image=rows)


def read_sample(root, row, task, size):
    sid = row['id']
    with Image.open(root / row['image']) as im:
        rgb = im.convert('RGB').resize((size, size), Image.Resampling.BILINEAR)
    arrays = []
    for name in (task, 'valid'):
        with Image.open(root / name / f'{sid}.png') as im:
            arrays.append(np.array(im.resize((size, size), Image.Resampling.NEAREST)) > 0)
    mask, valid = arrays
    if not valid.any() or (row['positive_pixels'][task] > 0 and not (mask & valid).any()):
        raise ValueError(f'{sid}: valid area or thin defect lost after resize; revise input size/patches')
    return np.asarray(rgb), mask, valid


def tensor(rgb):
    return torch.from_numpy(np.ascontiguousarray(((rgb.astype(np.float32)/255 - MEAN)/STD).transpose(2, 0, 1)))


class Samples(Dataset):
    def __init__(self, root, rows, task, size, augment=False):
        self.rows = rows
        self.samples = [read_sample(root, r, task, size) for r in rows]
        self.augment = augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        rgb, mask, valid = [a.copy() for a in self.samples[index]]
        if self.augment:
            if random.random() < .5:
                rgb, mask, valid = [np.flip(a, 1) for a in (rgb, mask, valid)]
            if random.random() < .2:
                rgb, mask, valid = [np.flip(a, 0) for a in (rgb, mask, valid)]
            rgb = np.asarray(ImageEnhance.Brightness(Image.fromarray(rgb)).enhance(random.uniform(.9, 1.1)))
        return (tensor(rgb), *[torch.from_numpy(np.ascontiguousarray(a[None], dtype=np.float32)) for a in (mask, valid)])


def loss_value(logits, y, valid, task):
    bce = (nn.functional.binary_cross_entropy_with_logits(logits, y, reduction='none') * valid).sum()/valid.sum()
    p = logits.sigmoid()
    tp = (p*y*valid).sum((1,2,3))
    fp = (p*(1-y)*valid).sum((1,2,3))
    fn = ((1-p)*y*valid).sum((1,2,3))
    alpha, beta = (.3, .7) if task == 'crack' else (.5, .5)
    return .5*bce + .5*(1 - (tp+1)/(tp+alpha*fp+beta*fn+1)).mean()


def checkpoint_state(path):
    obj = torch.load(path, map_location='cpu', weights_only=True)
    if isinstance(obj, dict):
        for key in ('state_dict', 'model_state_dict'):
            if key in obj:
                obj = obj[key]
                break
    if not isinstance(obj, dict) or not obj or not all(isinstance(v, torch.Tensor) for v in obj.values()):
        raise ValueError('Expected tensor state_dict or state_dict/model_state_dict wrapper')
    if all(k.startswith('module.') for k in obj):
        obj = {k[7:]: v for k, v in obj.items()}
    if not all(torch.isfinite(v).all() for v in obj.values()):
        raise ValueError('Nonfinite checkpoint')
    return obj


@torch.no_grad()
def evaluate(model, dataset, device, threshold, destination=None):
    model.eval()
    rows = []
    if destination:
        destination.mkdir(parents=True, exist_ok=True)
    for i, record in enumerate(dataset.rows):
        x, y, valid = dataset[i]
        pred = model(x[None].to(device)).sigmoid()[0,0].cpu().numpy() > threshold
        gt, v = y[0].numpy() > 0, valid[0].numpy() > 0
        rows.append(dict(id=record['id'], normal_confirmed=record.get('normal_confirmed', False), **metrics(pred, gt, v)))
        if destination:
            Image.fromarray(pred.astype('uint8')*255).save(destination / f"{record['id']}_prediction.png")
            rgb = dataset.samples[i][0].copy()
            overlay = rgb.copy()
            overlay[pred & gt & v] = [0, 210, 0]
            overlay[~pred & gt & v] = [255, 40, 40]
            overlay[pred & ~gt & v] = [255, 210, 0]
            overlay[~v] = [128, 128, 128]
            panel = np.concatenate([rgb, np.repeat((gt*255).astype('uint8')[...,None], 3, 2), overlay], axis=1)
            Image.fromarray(panel).save(destination / f"{record['id']}_comparison.jpg")
    return summarize(rows)


def train(args):
    if args.out.exists():
        raise FileExistsError('Use a new output directory to preserve existing runs')
    if args.size < 64 or args.size % 32 or args.epochs < 1 or args.batch_size < 1 or args.patience < 1 or args.lr <= 0:
        raise ValueError('Invalid training parameters; size must be a multiple of 32 and >=64')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU not connected; no silent CPU fallback')
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    root = args.data.resolve()
    data = json.loads((root/'dataset.json').read_text())
    records = data['items']
    for r in records:
        if r['split'] not in ('train', 'val'):
            raise ValueError('Unexpected split')
        for file, expected in r['file_hashes'].items():
            if sha(root/file) != expected:
                raise ValueError(f'Dataset changed: {file}')
    for field in ('source_group', 'original_sha256'):
        a = {r[field] for r in records if r['split']=='train' and r.get(field)}
        b = {r[field] for r in records if r['split']=='val' and r.get(field)}
        if a & b:
            raise ValueError(f'{field} leakage across train/val')
    split_by_id = {r['id']: r['split'] for r in records}
    for group in data.get('group_constraints', []):
        ids = group['ids'] if isinstance(group, dict) else group
        if len({split_by_id[x] for x in ids if x in split_by_id}) > 1:
            raise ValueError('Explicit grouping constraint violated')
    selected = [r for r in records if args.task in r['complete_targets']]
    splits = {s: [r for r in selected if r['split']==s] for s in ('train','val')}
    for s, rows in splits.items():
        if not any(r['positive_pixels'][args.task] > 0 for r in rows) or not any(r.get('normal_confirmed') is True for r in rows):
            raise ValueError(f'{s} needs both positive images and confirmed normals')
    train_ds = Samples(root, splits['train'], args.task, args.size, True)
    val_ds = Samples(root, splits['val'], args.task, args.size)
    model = smp.Unet(encoder_name='efficientnet-b0', encoder_weights=None, in_channels=3, classes=1)
    model.load_state_dict(checkpoint_state(args.weights), strict=True)
    model.to(args.device)
    original_hash = sha(args.weights)
    encoder_before = {k:v.detach().cpu().clone() for k,v in model.encoder.state_dict().items()}
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    args.out.mkdir(parents=True)
    config = {k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}
    config.update(model='Unet/efficientnet-b0', encoder_frozen=True, encoder_batchnorm_frozen=True,
                  mean=MEAN.tolist(), std=STD.tolist(), weights_sha256=original_hash, dataset_sha256=sha(root/'dataset.json'),
                  script_sha256=sha(__file__), python=platform.python_version(), torch=torch.__version__, smp=smp.__version__,
                  gpu=torch.cuda.get_device_name(0) if args.device=='cuda' else None,
                  train_ids=[r['id'] for r in splits['train']], val_ids=[r['id'] for r in splits['val']],
                  limitation='Pilot validation also used for early stopping/model selection; not independent test performance')
    save_json(args.out/'config.json', config)
    baseline = evaluate(model, val_ds, args.device, args.threshold, args.out/'baseline')
    save_json(args.out/'baseline.json', baseline)
    print('BASELINE', json.dumps(baseline['positive_macro']), flush=True)
    if args.check_only:
        save_json(args.out/'check_only.json', dict(status='passed', training_executed=False))
        return
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    best_iou, best_epoch, stale, history = -1., 0, 0, []
    start = time.monotonic()
    best = args.out/f'best_{args.task}_finetuned.pt'
    for epoch in range(1, args.epochs+1):
        model.train()
        model.encoder.eval()
        losses = []
        for x,y,v in loader:
            x,y,v = [a.to(args.device) for a in (x,y,v)]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_value(model(x), y, v, args.task)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite loss')
            loss.backward()
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1., error_if_nonfinite=True)
            optimizer.step()
            losses.append(float(loss.detach()))
        val = evaluate(model, val_ds, args.device, args.threshold)
        row = dict(epoch=epoch, loss=float(np.mean(losses)), positive_macro=val['positive_macro'],
                   normal_mean_false_positive_area=val['normal_mean_false_positive_area'], elapsed_seconds=round(time.monotonic()-start,2))
        history.append(row)
        save_json(args.out/'history.json', history)
        print('EPOCH', json.dumps(row), flush=True)
        score = val['positive_macro']['iou']
        if score > best_iou + 1e-8:
            best_iou, best_epoch, stale = score, epoch, 0
            state = {k:v.detach().cpu() for k,v in model.state_dict().items()}
            if not all(torch.isfinite(v).all() for v in state.values()):
                raise FloatingPointError('Nonfinite model state')
            tmp = best.with_suffix('.tmp')
            torch.save(state, tmp)
            tmp.replace(best)
        else:
            stale += 1
        if stale >= args.patience:
            break
    for k,v in model.encoder.state_dict().items():
        if not torch.equal(v.cpu(), encoder_before[k]):
            raise RuntimeError(f'Frozen encoder changed: {k}')
    model.load_state_dict(checkpoint_state(best), strict=True)
    tuned = evaluate(model, val_ds, args.device, args.threshold, args.out/'finetuned')
    if sha(args.weights) != original_hash:
        raise RuntimeError('Original checkpoint changed')
    delta = {k:tuned['positive_macro'][k]-baseline['positive_macro'][k] for k in ('iou','dice','recall','precision')}
    candidate = delta['iou'] > 0 and delta['recall'] >= 0 and tuned['normal_mean_false_positive_area'] <= baseline['normal_mean_false_positive_area']
    result = dict(status='completed', task=args.task, baseline=baseline, finetuned=tuned, delta=delta,
                  best_epoch=best_epoch, epochs_run=len(history), elapsed_seconds=time.monotonic()-start,
                  checkpoint=str(best), checkpoint_sha256=sha(best), encoder_unchanged=True, original_checkpoint_unchanged=True,
                  review_candidate=candidate, deployment_replaced=False,
                  limitations=[config['limitation'], 'Small sample and few source groups', '512px resizing can lose thin-crack detail', 'Scores depend on reviewed label quality; no safety-grade validation'])
    save_json(args.out/'results.json', result)
    print('COMPLETED', args.task, json.dumps(delta), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--task', choices=['crack','corrosion'], required=True)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--device', choices=['cuda','cpu'], default='cuda')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--patience', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--size', type=int, default=512)
    p.add_argument('--lr', type=float, default=3e-5)
    p.add_argument('--threshold', type=float, default=.5)
    p.add_argument('--seed', type=int, default=20260926)
    p.add_argument('--check-only', action='store_true')
    args = p.parse_args()
    if not 0 < args.threshold < 1:
        p.error('threshold must be between 0 and 1')
    train(args)
