"""Merge reviewed team manifests without silently splitting or approving samples."""
import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from labelme_masks import TASKS, rasterize

BOOL_FIELDS = ('image_reviewed', 'annotation_reviewed', 'normal_confirmed',
               'used_for_training', 'used_for_model_selection', 'independence_reviewed', 'train_only',
               'low_resolution_exception')
LIST_FIELDS = ('complete_targets', 'donor_groups', 'background_groups')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def read_manifests(paths):
    rows = []
    for path in paths:
        path = Path(path).resolve()
        if path.suffix.lower() == '.csv':
            with path.open(encoding='utf-8-sig', newline='') as f:
                items = list(csv.DictReader(f))
            for row in items:
                for key in BOOL_FIELDS:
                    if row.get(key, '').strip().lower() in ('true', 'false'):
                        row[key] = row[key].strip().lower() == 'true'
                for key in LIST_FIELDS:
                    row[key] = [s.strip() for s in row.get(key, '').split('|') if s.strip()]
        else:
            obj = json.loads(path.read_text(encoding='utf-8'))
            items = obj['items'] if isinstance(obj, dict) else obj
        for item in items:
            row = dict(item, manifest=str(path))
            for key in ('image', 'labelme'):
                row[key] = str((path.parent / row[key]).resolve()) if row.get(key) else ''
            rows.append(row)
    return rows


def lineage(row):
    """Global keys, never participant-prefixed: donors can also appear as real images."""
    keys = set()
    for key in ('source_group', 'original_group', 'source_url', 'source_image_url'):
        if row.get(key):
            keys.add('source:' + str(row[key]).strip())
    for key in ('donor_groups', 'background_groups'):
        for value in row.get(key, []):
            keys.add('source:' + value.strip())
    for key in ('original_sha256', 'image_sha256', 'pixel_sha256'):
        if row.get(key):
            keys.add('hash:' + row[key])
    return keys


def check_leakage(rows):
    issues = []
    groups = defaultdict(list)
    for row in rows:
        for token in lineage(row):
            groups[token].append(row)
    for token, members in groups.items():
        splits = {r.get('split') for r in members}
        if len(splits) > 1:
            issues.append(f"Split leakage {token}: " + ', '.join(r['id'] for r in members))
        if 'test' in splits and any(r.get('used_for_training') is True or
                                    r.get('used_for_model_selection') is True for r in members):
            issues.append(f"Test lineage has prior training/model-selection exposure: {token}")
    return issues


def audit(rows, min_side=512, allow_reviewed_low_resolution=False):
    errors, warnings, checked = [], [], []
    seen_ids, seen_pixels = set(), {}
    for item in rows:
        row = dict(item)
        sid = row.get('id', '?')
        try:
            if not isinstance(sid, str) or not re.fullmatch(r'(SB|ES|HM)_[A-Za-z0-9_-]+', sid) or sid in seen_ids:
                raise ValueError('unique ID with SB_, ES_ or HM_ prefix required')
            seen_ids.add(sid)
            if row.get('split') not in ('train', 'val', 'test'):
                raise ValueError('explicit split train, val or test required; no automatic reassignment')
            if row.get('data_type') not in ('real', 'synthetic'):
                raise ValueError('data_type must be real or synthetic')
            for key in ('source_group', 'source_url', 'review_evidence'):
                if not isinstance(row.get(key), str) or not row[key].strip():
                    raise ValueError(f'{key} required')
            for key in ('image_reviewed', 'annotation_reviewed'):
                if row.get(key) is not True:
                    raise ValueError(f'{key} must be true; drafts are not approved')
            for key in ('used_for_training', 'used_for_model_selection'):
                if not isinstance(row.get(key), bool):
                    raise ValueError(f'{key} must explicitly be true or false')
            if row.get('train_only') and row['split'] != 'train':
                raise ValueError('train_only sample cannot enter val/test')
            if row['split'] == 'test':
                if row['used_for_training'] or row['used_for_model_selection']:
                    raise ValueError('Test sample was previously used for training or model selection')
                if row.get('independence_reviewed') is not True or not row.get('independence_evidence'):
                    raise ValueError('Test independence review and evidence required, including donor/background history')
            for key in LIST_FIELDS:
                if not isinstance(row.get(key, []), list) or any(not isinstance(v, str) or not v.strip() for v in row.get(key, [])):
                    raise ValueError(f'{key} must be a list of nonempty strings')
            if row['data_type'] == 'synthetic':
                if not row.get('background_groups'):
                    raise ValueError('synthetic background_groups required')
                if row.get('normal_confirmed') is not True and not row.get('donor_groups'):
                    raise ValueError('synthetic defect donor_groups required')
            for key in ('image', 'labelme'):
                row[key + '_sha256'] = sha(row[key])
                if row.get('reviewed_' + key + '_sha256') != row[key + '_sha256']:
                    raise ValueError(f'{key} review hash missing or file changed since approval')
            with Image.open(row['image']) as im:
                rgb = im.convert('RGB')
                size = rgb.size
                if im.getexif().get(274, 1) != 1:
                    raise ValueError('EXIF orientation must be normalized before annotation')
                row['pixel_sha256'] = hashlib.sha256(str(size).encode() + rgb.tobytes()).hexdigest()
            if min(size) < min_side:
                if (allow_reviewed_low_resolution and row.get('low_resolution_exception') is True
                        and isinstance(row.get('low_resolution_reason'), str)
                        and row['low_resolution_reason'].strip()):
                    warnings.append(f'{sid}: reviewed low-resolution exception {size}: {row["low_resolution_reason"]}')
                else:
                    raise ValueError(f'original image {size} below minimum side {min_side}; explicit reviewed exception required, no upscaling workaround')
            if row.get('upsampled') is True:
                raise ValueError('upsampled original is not eligible')
            previous = seen_pixels.get(row['pixel_sha256'])
            if previous:
                raise ValueError(f'duplicate decoded image of {previous}')
            seen_pixels[row['pixel_sha256']] = sid
            doc = json.loads(Path(row['labelme']).read_text(encoding='utf-8'))
            if Path(doc.get('imagePath', '')).name != Path(row['image']).name:
                raise ValueError('LabelMe imagePath does not match image filename')
            masks, valid = rasterize(doc, size)
            complete = row.get('complete_targets', [])
            if not complete or not set(complete).issubset(TASKS):
                raise ValueError('complete_targets must explicitly name fully reviewed classes')
            if row.get('valid_region_policy') == 'reviewed_positive_only':
                if row['split'] != 'train' or row.get('train_only') is not True:
                    raise ValueError('partial positive-only annotation is train-only')
                valid &= np.logical_or.reduce(list(masks.values()))
            elif row.get('valid_region_policy', 'full') != 'full':
                raise ValueError('unknown valid_region_policy')
            if not valid.any():
                raise ValueError('no valid pixels')
            if row.get('normal_confirmed') is True:
                if doc['shapes'] or set(complete) != set(TASKS):
                    raise ValueError('normal requires empty shapes and both classes reviewed')
            elif not any(masks[t].any() for t in complete):
                raise ValueError('empty annotations are not negatives without normal confirmation')
            if not isinstance(row.get('normal_confirmed'), bool):
                raise ValueError('normal_confirmed must be true or false')
            row['positive_pixels'] = {t: int((m & valid).sum()) for t, m in masks.items()}
            row['valid_pixels'] = int(valid.sum())
            row['width'], row['height'] = size
            checked.append(row)
        except (ValueError, KeyError, TypeError, OSError) as exc:
            errors.append(f'{sid}: {exc}')
    if not rows:
        errors.append('No samples supplied')
    # Include invalid rows in metadata leakage checks, too, when keys are well formed.
    safe_rows = [r for r in rows if all(isinstance(r.get(k, []), list) for k in ('donor_groups', 'background_groups'))]
    try:
        errors.extend(check_leakage(safe_rows + checked))
    except (TypeError, AttributeError):
        errors.append('Malformed provenance identifiers')
    warnings.append('Hashes detect exact duplicates only. Human source/crop/scene checks remain required.')
    report = dict(passed=not errors, n_input=len(rows), n_valid=len(checked), errors=sorted(set(errors)),
                  warnings=warnings, counts=dict(Counter(f"{r['split']}/{r['data_type']}" for r in checked)))
    return checked, report


def verify_prepared(root, allowed_splits):
    root = Path(root)
    doc = json.loads((root / 'dataset.json').read_text(encoding='utf-8'))
    rows = doc['items']
    if not rows or any(r['split'] not in allowed_splits for r in rows):
        raise ValueError('Unexpected split: train/val and Test must be physically separated')
    errors = check_leakage(rows)
    for row in rows:
        for name, expected in row['file_hashes'].items():
            path = (root / name).resolve()
            if not path.is_relative_to(root.resolve()) or sha(path) != expected:
                errors.append(f"Prepared data changed: {row['id']}/{name}")
    if errors:
        raise ValueError('\n'.join(errors))
    return rows


def prepare(rows, out, min_side=512, allow_reviewed_low_resolution=False):
    checked, report = audit(rows, min_side, allow_reviewed_low_resolution)
    if not report['passed']:
        raise ValueError('\n'.join(report['errors']))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    save_json(out / 'audit.json', report)
    for kind, splits in [('trainval', {'train', 'val'}), ('test_sealed', {'test'})]:
        selected = [r for r in checked if r['split'] in splits]
        if not selected:
            continue
        root = out / kind
        for folder in ('images', 'labelme', 'crack', 'corrosion', 'valid'):
            (root / folder).mkdir(parents=True)
        exported = []
        for row in selected:
            sid = row['id']
            with Image.open(row['image']) as im:
                rgb = im.convert('RGB')
            label = json.loads(Path(row['labelme']).read_text(encoding='utf-8'))
            masks, valid = rasterize(label, rgb.size)
            if row.get('valid_region_policy') == 'reviewed_positive_only':
                valid &= np.logical_or.reduce(list(masks.values()))
            rgb.save(root / 'images' / f'{sid}.png')
            for name, mask in {**masks, 'valid': valid}.items():
                Image.fromarray(mask.astype('uint8') * 255).save(root / name / f'{sid}.png')
            label['imagePath'] = f'../images/{sid}.png'
            label['imageData'] = None
            save_json(root / 'labelme' / f'{sid}.json', label)
            item = {**row, 'image': f'images/{sid}.png', 'labelme': f'labelme/{sid}.json'}
            item['file_hashes'] = {f'{k}/{sid}.{ext}': sha(root / k / f'{sid}.{ext}')
                                  for k, ext in [('images', 'png'), ('labelme', 'json'), ('crack', 'png'), ('corrosion', 'png'), ('valid', 'png')]}
            exported.append(item)
        save_json(root / 'dataset.json', dict(schema_version=1, kind=kind, items=exported))
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['audit', 'prepare'])
    p.add_argument('--manifests', nargs='+', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--allow-reviewed-low-resolution', action='store_true',
                   help='Accept only reviewed rows with low_resolution_exception=true and a recorded reason')
    a = p.parse_args()
    rows = read_manifests(a.manifests)
    if a.command == 'audit':
        _, report = audit(rows, allow_reviewed_low_resolution=a.allow_reviewed_low_resolution)
        save_json(a.out, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        raise SystemExit(0 if report['passed'] else 2)
    print(json.dumps(prepare(rows, a.out, allow_reviewed_low_resolution=a.allow_reviewed_low_resolution), ensure_ascii=False))
