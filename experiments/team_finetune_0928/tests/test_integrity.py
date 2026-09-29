import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch
import segmentation_models_pytorch as smp
from data import audit, check_leakage, prepare, save_json, sha, verify_prepared
from import_metadata import adapt
from labelme_masks import rasterize
from metrics import metrics, grouped_summary
from engine import loss_value, checkpoint_state
from run import train, lock, final_test


def fixture(root):
    rows=[]
    for i in range(12):
        sid=f'SB_fixture_{i}'
        path=root/f'{sid}.png'; label=root/f'{sid}.json'
        Image.fromarray(np.random.default_rng(i).integers(0,256,(64,64,3),dtype='uint8')).save(path)
        normal=i%3==0
        shapes=[] if normal else [dict(label=t,shape_type='polygon',points=p)
                for t,p in [('crack',[[10,5],[13,5],[13,50],[10,50]]),('corrosion',[[25,25],[40,25],[40,40],[25,40]])]]
        save_json(label,dict(imageWidth=64,imageHeight=64,imagePath=path.name,shapes=shapes))
        rows.append(dict(id=sid,image=str(path),labelme=str(label),split='train' if i<6 else 'val' if i<9 else 'test',
            data_type='real',source_group=f'case_{i}',source_url=f'https://example.invalid/case_{i}',
            image_reviewed=True,annotation_reviewed=True,review_evidence='GENERATED ARRAY TEST FIXTURE, NOT RESEARCH DATA',
            normal_confirmed=normal,complete_targets=['crack','corrosion'],used_for_training=False,used_for_model_selection=False,
            reviewed_image_sha256=sha(path),reviewed_labelme_sha256=sha(label),independence_reviewed=True,
            independence_evidence='Generated independent random arrays for software testing'))
    return rows


class IntegrityTests(unittest.TestCase):
    def test_low_resolution_exception_requires_opt_in_review_and_reason(self):
        with tempfile.TemporaryDirectory() as d:
            rows=fixture(Path(d))
            self.assertFalse(audit(rows)[1]['passed'])
            self.assertFalse(audit(rows,allow_reviewed_low_resolution=True)[1]['passed'])
            for row in rows:
                row['low_resolution_exception']=True
                row['low_resolution_reason']='Explicitly reviewed small image fixture'
            self.assertFalse(audit(rows)[1]['passed'])
            self.assertTrue(audit(rows,allow_reviewed_low_resolution=True)[1]['passed'])
            rows[0]['annotation_reviewed']=False
            self.assertFalse(audit(rows,allow_reviewed_low_resolution=True)[1]['passed'])
            rows[0]['annotation_reviewed']=True
            rows[0]['low_resolution_reason']=''
            self.assertFalse(audit(rows,allow_reviewed_low_resolution=True)[1]['passed'])

    def test_metadata_import_parses_exception_and_train_only_booleans(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); rows=fixture(root); path=root/'metadata.csv'
            sample=dict(image_id=rows[0]['id'],image_file=rows[0]['image'],label_file=rows[0]['labelme'],
                        classes='normal',split='val',data_type='real',prior_use='none',reviewed='true',
                        train_only='false',low_resolution_exception='true',low_resolution_reason='Reviewed fixture')
            with path.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(sample));w.writeheader();w.writerow(sample)
            imported=adapt(path,root/'imported.json','team record')[0]
            self.assertIs(imported['train_only'],False)
            self.assertIs(imported['low_resolution_exception'],True)
            self.assertEqual(imported['low_resolution_reason'],'Reviewed fixture')

    def test_metrics_exclude_ignored_pixels_and_keep_empty_domains_null(self):
        m=metrics([1,1,0,1],[1,0,1,0],[1,1,1,0])
        self.assertAlmostEqual(m['iou'],1/3)
        self.assertEqual(m['precision'],.5); self.assertEqual(m['recall'],.5)
        positive=dict(id='a',source_group='a',data_type='real',normal_confirmed=False,**m)
        normal=dict(id='b',source_group='b',data_type='real',normal_confirmed=True,**metrics([1,0,0,0],[0]*4,[1]*4))
        result=grouped_summary([positive,normal])
        self.assertEqual(result['real']['normal_false_positive_image_rate'],1)
        self.assertEqual(result['real']['normal_mean_false_positive_area'],.25)
        self.assertAlmostEqual(result['real']['positive_macro']['iou'],1/3)
        self.assertIsNone(result['synthetic']['positive_macro']['iou'])
        self.assertEqual(grouped_summary([normal],1)['real']['normal_false_positive_image_rate'],0)

    def test_masks_keep_dual_labels_and_ignore(self):
        shape=lambda t:dict(label=t,shape_type='polygon',points=[[1,1],[6,1],[6,6],[1,6]])
        doc=dict(imageWidth=8,imageHeight=8,shapes=[shape('crack'),shape('corrosion_severe')])
        masks,valid=rasterize(doc,(8,8))
        self.assertTrue(masks['crack'][2,2] and masks['corrosion'][2,2])
        doc['shapes'].append(dict(label='ignore',shape_type='polygon',points=[[3,3],[5,3],[5,5],[3,5]]))
        masks,valid=rasterize(doc,(8,8)); self.assertFalse(valid[4,4] or masks['crack'][4,4])

    def test_unreviewed_changed_and_duplicate_images_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            rows=fixture(Path(d)); self.assertTrue(audit(rows,64)[1]['passed'])
            bad=copy.deepcopy(rows); bad[0]['annotation_reviewed']=False
            self.assertIn('drafts',str(audit(bad,64)[1]['errors']))
            bad=copy.deepcopy(rows); Path(bad[0]['labelme']).write_text(Path(bad[0]['labelme']).read_text()+' ')
            self.assertIn('changed',str(audit(bad,64)[1]['errors']))
            bad=copy.deepcopy(rows); bad[1]['image']=rows[2]['image'];bad[1]['reviewed_image_sha256']=rows[2]['reviewed_image_sha256']
            self.assertFalse(audit(bad,64)[1]['passed'])

    def test_lineage_crosses_donor_background_and_prior_exposure(self):
        a=dict(id='SB_a',split='train',source_group='same_steel_image')
        b=dict(id='HM_b',split='test',source_group='synthetic_b',donor_groups=['same_steel_image'])
        self.assertTrue(check_leakage([a,b]))
        b['donor_groups']=[]; b['background_groups']=['same_steel_image']
        self.assertTrue(check_leakage([a,b]))
        b['background_groups']=[];b['used_for_model_selection']=True
        self.assertTrue(check_leakage([b]))

    def test_test_history_unknown_or_used_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            rows=fixture(Path(d)); rows[-1]['used_for_model_selection']=True
            self.assertIn('previously used',str(audit(rows,64)[1]['errors']))
            rows[-1]['used_for_model_selection']=False;rows[-1]['independence_reviewed']=False
            self.assertIn('independence',str(audit(rows,64)[1]['errors']))

    def test_rectangle_and_false_normal_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            rows=fixture(Path(d)); rows[1]['normal_confirmed']=True
            self.assertIn('empty shapes',str(audit(rows,64)[1]['errors']))
            rows[1]['normal_confirmed']=False
            p=Path(rows[1]['labelme']);doc=json.loads(p.read_text());doc['shapes'][0]['shape_type']='rectangle';save_json(p,doc)
            rows[1]['reviewed_labelme_sha256']=sha(p)
            self.assertIn('Only polygon',str(audit(rows,64)[1]['errors']))

    def test_masked_loss_ignores_excluded_region(self):
        logits=torch.zeros((1,1,2,2),requires_grad=True)
        valid=torch.tensor([[[[1.,1.],[1.,0.]]]])
        loss_value(logits,torch.zeros_like(logits),valid,'crack').backward()
        self.assertEqual(float(logits.grad[0,0,1,1]),0.)

    def test_notion_metadata_import_does_not_approve_missing_review(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); rows=fixture(root)
            path=root/'metadata.csv'
            sample=dict(image_id=rows[0]['id'],image_file=rows[0]['image'],label_file=rows[0]['labelme'],
                        classes='normal',split='test',data_type='real',prior_use='none',reviewed='false')
            with path.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=list(sample));w.writeheader();w.writerow(sample)
            imported=adapt(path,root/'imported.json','team record')
            self.assertFalse(imported[0]['annotation_reviewed'])
            self.assertNotIn('reviewed_labelme_sha256',imported[0])
            self.assertIsNone(imported[0]['independence_reviewed'])

    def test_full_train_lock_test_with_random_fixture_only(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);rows=fixture(root);prepare(rows,root/'prepared',64)
            with self.assertRaisesRegex(ValueError,'Unexpected split'):
                verify_prepared(root/'prepared/test_sealed',{'train','val'})
            # Random initialization is ONLY this software fixture, never the production runner fallback.
            torch.manual_seed(77)
            base=root/'fixture_only.pt'
            torch.save(smp.Unet(encoder_name='efficientnet-b0',encoder_weights=None,in_channels=3,classes=1).state_dict(),base)
            sealed=root/'prepared/test_sealed'
            hidden=root/'hidden_test';sealed.rename(hidden)
            config=dict(task='crack',data=str(root/'prepared/trainval'),weights=str(base),device='cpu',size=64,epochs=1,patience=1,batch_size=2)
            train(config,root/'run')  # Test directory is absent during learning.
            hidden.rename(sealed)
            with self.assertRaises(FileNotFoundError):final_test(root/'run',sealed,root/'final')
            lock(root/'run','Software fixture only; no scientific model selection claim')
            final_test(root/'run',sealed,root/'final','cpu')
            report=json.loads((root/'final/result.json').read_text())
            self.assertEqual(report['split'],'test')
            self.assertEqual(report['comparison']['real']['baseline']['n_images'],3)
            self.assertTrue((root/'final/summary.csv').exists())
            with self.assertRaises(FileExistsError):final_test(root/'run',sealed,root/'second_final','cpu')
            self.assertEqual(set(checkpoint_state(base)),set(checkpoint_state(root/'run/best.pt')))


if __name__=='__main__':unittest.main()
