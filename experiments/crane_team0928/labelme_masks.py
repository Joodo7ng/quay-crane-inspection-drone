"""Polygon rasterization retained from the reviewed 2026-09-26 pipeline."""
import numpy as np
from PIL import Image, ImageDraw
TASKS = ("crack", "corrosion")

def rasterize(doc, size):
    if (doc.get('imageWidth'), doc.get('imageHeight')) != size:
        raise ValueError('LabelMe dimensions differ from image dimensions')
    layers = {k: Image.new('L', size, 0) for k in (*TASKS, 'ignore')}
    for shape in doc['shapes']:
        label = shape['label'].strip().lower()
        # Preserve original graded JSON; merge only the binary training mask.
        if label in ('corrosion_fair', 'corrosion_poor', 'corrosion_severe'):
            label = 'corrosion'
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
