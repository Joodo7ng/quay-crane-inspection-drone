"""Validate reviewed LabelMe data and export portable, source-grouped binary masks."""
import argparse
import hashlib
import json
import random
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

TASKS = ('crack', 'corrosion')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(path)


def rasterize(doc, size):
    if (doc.get('imageWidth'), doc.get('imageHeight')) != size:
        raise ValueError('LabelMe dimensions differ from image dimensions')
    layers = {k: Image.new('L', size, 0) for k in (*TASKS, 'ignore')}
    for shape in doc['shapes']:
        label = shape['label'].strip().lower()
        if label not in layers:
            raise ValueError(f'Unknown label: {label}; allowed: crack, corrosion, ignore')
        if shape.get('shape_type') != 'polygon':
            raise ValueError('Only polygon shapes are supported; convert lines to reviewed polygons')
        pts = np.asarray(shape['points'], dtype=float)
        if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3 or not np.isfinite(pts).all():
            raise ValueError('Invalid polygon points')
        if (pts < 0).any() or (pts[:, 0] >= size[0]).any() or (pts[:, 1] >= size[1]).any():
            raise ValueError('Polygon outside image bounds')
        area = abs(np.dot(pts[:, 0], np.roll(pts[:, 1], 1)) - np.dot(pts[:, 1], np.roll(pts[:, 0], 1))) / 2
        if area <= 0:
            raise ValueError('Zero-area polygon')
        ImageDraw.Draw(layers[label]).polygon([tuple(p) for p in pts], fill=1)
    valid = np.array(layers['ignore']) == 0
    if not valid.any():
        raise ValueError('Image is entirely ignored')
    masks = {k: (np.array(layers[k]) > 0) & valid for k in TASKS}
    return masks, valid


def load_rows(manifest, require_review=True):
    manifest = Path(manifest).resolve()
    source = json.loads(manifest.read_text(encoding='utf-8'))
    rows, errors, seen, pixel_hashes = [], [], set(), {}
    for entry in source['items']:
        row = dict(entry)
        sid = row.get('id', '?')
        try:
            if not re.fullmatch(r'[A-Za-z0-9_-]+', sid) or sid in seen:
                raise ValueError('Invalid or duplicate ID')
            seen.add(sid)
            if not isinstance(row.get('source_group'), str) or not row['source_group'].strip():
                raise ValueError('source_group is required; do not use patch ID as a substitute')
            row['image'] = str((manifest.parent / row['image']).resolve())
            row['labelme'] = str((manifest.parent / row['labelme']).resolve())
            with Image.open(row['image']) as im:
                rgb = im.convert('RGB')
                size = rgb.size
                pixel_sha = hashlib.sha256(str(size).encode() + rgb.tobytes()).hexdigest()
            if pixel_sha in pixel_hashes:
                raise ValueError(f'Duplicate decoded image: {pixel_hashes[pixel_sha]}')
            pixel_hashes[pixel_sha] = sid
            doc = json.loads(Path(row['labelme']).read_text(encoding='utf-8'))
            masks, valid = rasterize(doc, size)
            if row.get('valid_region_policy') == 'reviewed_positive_only':
                # Partial labels must not turn unreviewed background into negatives.
                valid &= np.logical_or.reduce(list(masks.values()))
                if not valid.any() or row.get('train_only') is not True:
                    raise ValueError('Positive-only supervision requires nonempty labels and train_only=true')
            row['image_sha256'] = sha(row['image'])
            row['labelme_sha256'] = sha(row['labelme'])
            if require_review:
                for key in ('annotation_reviewed', 'training_allowed'):
                    if row.get(key) is not True:
                        raise ValueError(f'{key} is not true; photo acceptance is not label approval')
                if not row.get('review_evidence'):
                    raise ValueError('Missing review_evidence (review record path or note)')
                for key in ('image', 'labelme'):
                    if row.get('reviewed_' + key + '_sha256') != row[key + '_sha256']:
                        raise ValueError(f'{key} changed or reviewed hash is missing; review latest revision')
                complete = row.get('complete_targets', [])
                if not complete or not set(complete).issubset(TASKS):
                    raise ValueError('complete_targets must explicitly name the exhaustively reviewed tasks')
                if row.get('normal_confirmed') is True and (set(complete) != set(TASKS) or any(m.any() for m in masks.values())):
                    raise ValueError('Confirmed normal must have both targets reviewed and no defect pixels')
                if row.get('normal_confirmed') is not True and not any(masks[t].any() for t in complete):
                    raise ValueError('Empty target requires normal_confirmed=true; missing labels are not negatives')
            row['positive_pixels'] = {k: int(v.sum()) for k, v in masks.items()}
            row['valid_pixels'] = int(valid.sum())
            rows.append(row)
        except (ValueError, KeyError, OSError, TypeError) as exc:
            errors.append(f'{sid}: {exc}')
    if not source['items']:
        errors.append('Manifest has no items')
    if errors:
        raise ValueError('\n'.join(errors))
    return rows, source.get('group_constraints', [])


def grouped_split(rows, constraints, seed=20260926, val_fraction=.2):
    """Search assignments using only GT class counts; never use predictions."""
    parent = {r['id']: r['id'] for r in rows}
    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(ids):
        ids = [x for x in ids if x in parent]
        for x in ids[1:]:
            parent[root(x)] = root(ids[0])
    for field in ('source_group', 'original_sha256'):
        groups = {}
        for r in rows:
            if r.get(field):
                groups.setdefault(r[field], []).append(r['id'])
        for ids in groups.values():
            union(ids)
    for item in constraints:
        union(item['ids'] if isinstance(item, dict) else item)
    groups = {}
    for r in rows:
        groups.setdefault(root(r['id']), []).append(r['id'])
    keys = sorted(groups)
    train_only_groups = {root(r['id']) for r in rows if r.get('train_only') is True}
    if len(keys) < 2:
        raise ValueError('At least two independent source groups are needed')
    counts = {}
    for k, ids in groups.items():
        members = [r for r in rows if r['id'] in ids]
        counts[k] = np.array([len(members)] + [sum(t in r['complete_targets'] and r['positive_pixels'][t] > 0 for r in members) for t in TASKS] + [sum(r.get('normal_confirmed') is True for r in members)], float)
    total = sum(counts.values())
    rng, best = random.Random(seed), None
    for _ in range(20000):
        selected = {k for k in keys if rng.random() < val_fraction}
        if not selected or len(selected) == len(keys) or selected & train_only_groups:
            continue
        n = sum(counts[k] for k in selected)
        if (n[1:] == 0).any() or ((total - n)[1:] == 0).any():
            continue
        score = float(np.abs(n / total - val_fraction).sum())
        if best is None or score < best[0]:
            best = score, selected
    if best is None:
        raise ValueError('Cannot put both defect classes and confirmed normals in both splits without source leakage; add independent groups')
    return {r['id']: 'val' if root(r['id']) in best[1] else 'train' for r in rows}


def prepare(manifest, out, seed=20260926):
    rows, constraints = load_rows(manifest)
    split = grouped_split(rows, constraints, seed)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    for folder in ('images', 'crack', 'corrosion', 'valid', 'labelme'):
        (out / folder).mkdir()
    exported = []
    for r in rows:
        sid = r['id']
        with Image.open(r['image']) as im:
            rgb = im.convert('RGB')
        doc = json.loads(Path(r['labelme']).read_text(encoding='utf-8'))
        masks, valid = rasterize(doc, rgb.size)
        if r.get('valid_region_policy') == 'reviewed_positive_only':
            valid &= np.logical_or.reduce(list(masks.values()))
        rgb.save(out / 'images' / f'{sid}.png')
        for name, mask in {**masks, 'valid': valid}.items():
            Image.fromarray(mask.astype('uint8') * 255).save(out / name / f'{sid}.png')
        (out / 'labelme' / f'{sid}.json').write_bytes(Path(r['labelme']).read_bytes())
        item = {**r, 'split': split[sid], 'source_image': r['image'], 'source_labelme': r['labelme'], 'image': f'images/{sid}.png', 'labelme': f'labelme/{sid}.json'}
        item['file_hashes'] = {f'{f}/{sid}.{ext}': sha(out / f / f'{sid}.{ext}') for f, ext in [('images', 'png'), ('crack', 'png'), ('corrosion', 'png'), ('valid', 'png'), ('labelme', 'json')]}
        exported.append(item)
    save_json(out / 'dataset.json', dict(seed=seed, items=exported, group_constraints=constraints, selection='Source groups and GT counts only; no predictions', evaluation='Pilot validation for model selection; no independent test'))
    print(json.dumps({s: sum(r['split'] == s for r in exported) for s in ('train', 'val')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--audit-drafts', action='store_true', help='Geometry audit only; never exports a training dataset')
    args = parser.parse_args()
    if args.audit_drafts:
        rows, _ = load_rows(args.manifest, require_review=False)
        print(json.dumps({'geometry_valid': len(rows), 'training_approval_checked': False}))
    else:
        if args.out is None:
            parser.error('--out is required for export')
        prepare(args.manifest, args.out)
