"""Binary U-Net core adapted from Subin's verified 2026-09-26 experiment.
The encoder, including its BatchNorm running state, is frozen by the runner.
"""
import random
from pathlib import Path
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from metrics import metrics, grouped_summary

MEAN = np.array([.485, .456, .406], dtype=np.float32)
STD = np.array([.229, .224, .225], dtype=np.float32)


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
            rotations = random.randrange(4)
            rgb, mask, valid = [np.rot90(a, rotations) for a in (rgb, mask, valid)]
            rgb = np.asarray(ImageEnhance.Brightness(Image.fromarray(rgb)).enhance(random.uniform(.9, 1.1)))
            if random.random() < .2:
                rgb = np.asarray(Image.fromarray(rgb).filter(ImageFilter.GaussianBlur(random.uniform(.2, .8))))
        return (tensor(rgb), *[torch.from_numpy(np.ascontiguousarray(a[None], dtype=np.float32)) for a in (mask, valid)])


def loss_value(logits, y, valid, task, alpha=None, reduction='batch'):
    bce = (nn.functional.binary_cross_entropy_with_logits(logits, y, reduction='none') * valid).sum()/valid.sum()
    p = logits.sigmoid()
    if reduction not in ('batch', 'image'):
        raise ValueError('Unknown Tversky reduction')
    axes = (0,1,2,3) if reduction == 'batch' else (1,2,3)
    tp = (p*y*valid).sum(axes)
    fp = (p*(1-y)*valid).sum(axes)
    fn = ((1-p)*y*valid).sum(axes)
    alpha = (.3 if task == 'crack' else .5) if alpha is None else alpha
    beta = 1 - alpha
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
def evaluate(model, dataset, device, threshold, destination=None, normal_fp_pixels=0):
    model.eval()
    rows = []
    if destination:
        destination.mkdir(parents=True, exist_ok=True)
    for i, record in enumerate(dataset.rows):
        x, y, valid = dataset[i]
        probability = model(x[None].to(device)).sigmoid()[0,0].cpu().numpy()
        if not np.isfinite(probability).all():
            raise FloatingPointError('Nonfinite prediction')
        pred = probability > threshold
        gt, v = y[0].numpy() > 0, valid[0].numpy() > 0
        rows.append(dict(id=record['id'], source_group=record['source_group'], data_type=record['data_type'], normal_confirmed=record.get('normal_confirmed', False), **metrics(pred, gt, v)))
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
    return dict(groups=grouped_summary(rows, normal_fp_pixels), per_image=rows)
