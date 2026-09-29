import unittest
import json
import tempfile
from pathlib import Path
import torch
import segmentation_models_pytorch as smp
from experiments5 import select_conditions, execute
from test_integrity import fixture
from data import prepare

class Conditions(unittest.TestCase):
    def test_same_real_validation_and_synthetic_train_only(self):
        rows=[]
        for split in ('train','val'):
            for kind in ('real','synthetic'):
                for task in ('crack','corrosion','normal'):
                    rows.append(dict(id=f'{split}_{kind}_{task}',split=split,data_type=kind,
                                     complete_targets=['crack','corrosion'] if task=='normal' else [task],
                                     positive_pixels=dict(crack=int(task=='crack'),corrosion=int(task=='corrosion')),
                                     normal_confirmed=task=='normal'))
        c=select_conditions(rows)
        ids=lambda name,split:{r['id'] for r in c[name] if r['split']==split}
        self.assertEqual(ids('crack_real','val'),ids('crack_real_synthetic','val'))
        self.assertTrue(ids('crack_real','train') < ids('crack_real_synthetic','train'))
        self.assertTrue(all(r['data_type']=='real' for r in c['corrosion_real']))
        self.assertNotIn('val_synthetic_crack',ids('crack_real_synthetic','val'))
        with self.assertRaises(ValueError):select_conditions([r for r in rows if r['data_type']=='real'])

    def test_three_runs_equal_crack_updates_and_no_test(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            rows=fixture(root)
            rows=[r for r in rows if r['split']!='test']
            rows[4]['data_type']='synthetic'
            rows[5]['data_type']='synthetic'
            for i in (4,5):
                rows[i]['background_groups']=[f'fixture_background_{i}']
                rows[i]['donor_groups']=[f'fixture_donor_{i}']
            prepare(rows,root/'prepared',64)
            base=root/'fixture_only.pt'
            torch.save(smp.Unet(encoder_name='efficientnet-b0',encoder_weights=None,in_channels=3,classes=1).state_dict(),base)
            result=execute(root/'prepared/trainval',dict(crack=base,corrosion=base),root/'runs',
                           epochs=2,device='cpu',size=64)
            self.assertEqual(len(result),5)
            updates=[]
            for name in ('crack_real','crack_real_synthetic'):
                history=json.loads((root/'runs'/name/'history.json').read_text())
                updates.append(sum(r['optimizer_steps'] for r in history))
            self.assertEqual(updates,[4,4])
            self.assertFalse(list((root/'runs').rglob('test_evaluation_started.json')))

if __name__=='__main__':unittest.main()
