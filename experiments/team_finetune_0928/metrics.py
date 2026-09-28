"""Shared binary pixel metrics. Undefined quantities stay null, never invented as 1."""
import numpy as np


def metrics(pred, gt, valid):
    pred, gt, valid = [np.asarray(a, bool) for a in (pred, gt, valid)]
    if pred.shape != gt.shape or gt.shape != valid.shape or not valid.any():
        raise ValueError('Metric arrays must have equal shape and valid pixels')
    tp = int((pred & gt & valid).sum())
    fp = int((pred & ~gt & valid).sum())
    fn = int((~pred & gt & valid).sum())
    tn = int((~pred & ~gt & valid).sum())
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, target_pixels=tp+fn, valid_pixels=int(valid.sum()),
                iou=tp/(tp+fp+fn) if tp+fp+fn else None,
                dice=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
                recall=tp/(tp+fn) if tp+fn else None,
                precision=tp/(tp+fp) if tp+fp else (0. if tp+fn else None))


def summarize(rows, normal_fp_pixels=0):
    """Image FP iff pixel count > configured threshold; default is any predicted pixel."""
    def average(rs, key):
        values = [r[key] for r in rs if r[key] is not None]
        return float(np.mean(values)) if values else None
    positive = [r for r in rows if r['target_pixels'] > 0]
    normal = [r for r in rows if r['normal_confirmed']]
    count = sum(r['fp'] > normal_fp_pixels for r in normal)
    tp, fp, fn = [sum(r[k] for r in rows) for k in ('tp', 'fp', 'fn')]
    return dict(n_images=len(rows), positive_n=len(positive), normal_n=len(normal),
                positive_macro={k: average(positive, k) for k in ('iou', 'dice', 'precision', 'recall')},
                micro_iou=tp/(tp+fp+fn) if tp+fp+fn else None,
                normal_false_positive_images=count,
                normal_false_positive_image_rate=count/len(normal) if normal else None,
                normal_mean_false_positive_area=float(np.mean([r['fp']/r['valid_pixels'] for r in normal])) if normal else None,
                normal_fp_pixels_threshold=normal_fp_pixels)


def grouped_summary(rows, normal_fp_pixels=0):
    # Intentionally no pooled real+synthetic headline: the two test populations differ.
    return {kind: summarize([r for r in rows if r['data_type'] == kind], normal_fp_pixels)
            for kind in ('real', 'synthetic')}
