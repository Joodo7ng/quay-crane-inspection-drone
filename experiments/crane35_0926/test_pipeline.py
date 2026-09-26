"""Data integrity and synthetic end-to-end tests; synthetic scores are not research results."""
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from prepare_data import rasterize, load_rows, grouped_split, prepare, sha, save_json
from train_finetune import metrics, loss_value
import torch


def polygon(label, points):
    return dict(label=label, points=points, shape_type='polygon', flags={})


def fixture(root):
    rows = []
    for i in range(8):
        sid = f'sample_{i}'
        image = root/f'{sid}.png'
        label = root/f'{sid}.json'
        a = np.random.default_rng(i).integers(0,256,(64,64,3),dtype='uint8')
        Image.fromarray(a).save(image)
        shapes = [] if i < 2 else [polygon('crack', [[10,10],[15,10],[15,50],[10,50]]), polygon('corrosion', [[25,20],[45,20],[45,40],[25,40]])]
        save_json(label, dict(imageWidth=64,imageHeight=64,shapes=shapes))
        rows.append(dict(id=sid,image=image.name,labelme=label.name,source_group=f'g{i}',
                         normal_confirmed=i<2,annotation_reviewed=True,training_allowed=True,
                         complete_targets=['crack','corrosion'],review_evidence='synthetic test fixture only',
                         reviewed_image_sha256=sha(image),reviewed_labelme_sha256=sha(label)))
    manifest=root/'manifest.json'
    save_json(manifest,dict(items=rows,group_constraints=[['sample_2','sample_3']]))
    return manifest


class PipelineTests(unittest.TestCase):
    def test_overlap_and_ignore(self):
        shapes=[polygon(t,[[1,1],[5,1],[5,5],[1,5]]) for t in ('crack','corrosion')]
        shapes.append(polygon('ignore',[[3,3],[6,3],[6,6],[3,6]]))
        masks,valid=rasterize(dict(imageWidth=8,imageHeight=8,shapes=shapes),(8,8))
        self.assertTrue(masks['crack'][2,2] and masks['corrosion'][2,2])
        self.assertFalse(valid[4,4] or masks['crack'][4,4])

    def test_metrics_and_masked_loss(self):
        m=metrics([1,1,0,1],[1,0,1,0],[1,1,1,0])
        self.assertAlmostEqual(m['iou'],1/3)
        self.assertEqual(m['dice'],.5)
        x=torch.zeros((1,1,2,2),requires_grad=True)
        v=torch.tensor([[[[1.,1.],[1.,0.]]]])
        loss_value(x,torch.zeros_like(x),v,'crack').backward()
        self.assertEqual(x.grad[0,0,1,1].item(),0.)

    def test_review_hash_and_grouping(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);manifest=fixture(root)
            rows,constraints=load_rows(manifest)
            split=grouped_split(rows,constraints)
            self.assertEqual(split['sample_2'],split['sample_3'])
            prepare(manifest,root/'export')
            self.assertTrue((root/'export/dataset.json').is_file())
            data=json.loads(manifest.read_text())
            data['items'][0]['annotation_reviewed']=False
            save_json(manifest,data)
            with self.assertRaisesRegex(ValueError,'annotation_reviewed'):
                load_rows(manifest)
            data['items'][0]['annotation_reviewed']=True
            save_json(manifest,data)
            (root/'sample_0.json').write_text((root/'sample_0.json').read_text()+' ')
            with self.assertRaisesRegex(ValueError,'changed'):
                load_rows(manifest)

    def test_impossible_group_split(self):
        with tempfile.TemporaryDirectory() as d:
            rows,constraints=load_rows(fixture(Path(d)))
            for r in rows:r['source_group']='same_source'
            with self.assertRaisesRegex(ValueError,'independent'):
                grouped_split(rows,constraints)


if __name__=='__main__':
    unittest.main()
