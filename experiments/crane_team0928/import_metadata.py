"""Adapt the team's agreed Notion metadata.csv to a versioned audit manifest.

This records a team's declared review, not an independent human verification.
Missing/ambiguous review, prior-use or independence values stay unresolved.
"""
import argparse
import csv
from pathlib import Path
from data import save_json, sha


def boolean(value):
    value = str(value).strip().lower()
    return {'true':True, 'false':False, 'yes':True, 'no':False, '1':True, '0':False,
            '완료':True, '미완료':False}.get(value)


def adapt(metadata, out, evidence):
    metadata, out = Path(metadata).resolve(), Path(out).resolve()
    if out.exists(): raise FileExistsError(out)
    with metadata.open(encoding='utf-8-sig', newline='') as f: source = list(csv.DictReader(f))
    rows = []
    for raw in source:
        row = dict(raw)
        row['id'] = raw.get('image_id') or raw.get('id')
        for target, alias in [('image','image_file'), ('labelme','label_file')]:
            value = raw.get(alias) or raw.get(target)
            row[target] = str((metadata.parent/value).resolve()) if value else ''
        classes = set(raw.get('classes','').replace('+','|').replace(',','|').split('|'))
        classes = {s.strip() for s in classes if s.strip()}
        explicit_targets = raw.get('complete_targets','')
        row['complete_targets'] = explicit_targets.split('|') if explicit_targets else (['crack','corrosion'] if classes=={'normal'} else sorted(classes & {'crack','corrosion'}))
        row['normal_confirmed'] = classes=={'normal'}
        row['image_reviewed'] = boolean(raw.get('image_reviewed', raw.get('reviewed','')))
        row['annotation_reviewed'] = boolean(raw.get('reviewed',''))
        row['review_evidence'] = evidence
        row['review_source'] = 'Team metadata declaration; not independently re-reviewed by this converter'
        row['source_metadata_sha256'] = sha(metadata)
        prior = raw.get('prior_use','').strip().lower()
        known = {'none':(False,False), '사용 없음':(False,False), 'train':(True,False),
                 'val':(False,True), 'validation':(False,True), 'model_selection':(False,True),
                 'train+val':(True,True)}
        row['used_for_training'], row['used_for_model_selection'] = known.get(prior,(None,None))
        row['independence_reviewed'] = boolean(raw.get('independence_reviewed',''))
        row['independence_evidence'] = raw.get('independence_evidence','')
        row['donor_groups'] = [v.strip() for v in raw.get('defect_source_ids','').split('|') if v.strip()]
        row['background_groups'] = [v.strip() for v in raw.get('background_source_id','').split('|') if v.strip()]
        for field in ('train_only', 'low_resolution_exception'):
            if raw.get(field, '').strip():
                row[field] = boolean(raw[field])
        if row['image_reviewed'] is True and row['annotation_reviewed'] is True:
            for key in ('image','labelme'):
                if Path(row[key]).is_file(): row['reviewed_'+key+'_sha256'] = sha(row[key])
        rows.append(row)
    save_json(out, dict(status='requires_data_audit', items=rows))
    return rows


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--metadata', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--review-evidence', required=True, help='Team review record URL/version; conversion is not approval')
    a=p.parse_args()
    print('Imported records:',len(adapt(a.metadata,a.out,a.review_evidence)))
