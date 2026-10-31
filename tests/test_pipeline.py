import unittest

import numpy as np
import torch

from prepare_data import BANDS, parse_patch, spatial_split
from tree_model import TemporalClassifier, hierarchical_loss, spectral_features, FEATURE_DIM


class PipelineTests(unittest.TestCase):
    def test_month_order_and_recomputed_indices(self):
        properties={}
        for m in range(8):
            suffix='' if m==0 else f'_{m}'
            for band in BANDS:
                properties[band+suffix]=np.full((5,5),.1+m*.01).tolist()
        properties['B8']=np.full((5,5),.6).tolist()
        properties['B4']=np.full((5,5),.2).tolist()
        properties['B2_1']=None
        x=parse_patch({'properties':properties})
        self.assertEqual(x.shape,(8,15,5,5))
        self.assertTrue(np.isnan(x[1,0]).all())
        self.assertAlmostEqual(float(x[0,10,2,2]),.5,places=5)
        self.assertAlmostEqual(float(x[7,0,2,2]),.17,places=5)

    def test_missing_month_does_not_become_an_observation(self):
        raw=np.full((2,8,15,5,5),np.nan,dtype=np.float32)
        raw[0,0]=.2
        filled,stats,mask=spectral_features(raw,np.full(15,.1),np.ones(15))
        self.assertEqual(int(mask.sum()),1)
        self.assertTrue(np.isfinite(filled).all())
        self.assertTrue(np.isfinite(stats).all())
        self.assertAlmostEqual(float(filled[1,0,0,0,0]),.1,places=5)

    def test_masked_months_cannot_change_prediction(self):
        torch.manual_seed(1)
        model=TemporalClassifier().eval()
        x=torch.randn(2,8,FEATURE_DIM)
        mask=torch.zeros(2,8,dtype=torch.bool);mask[:,0]=True
        changed=x.clone();changed[:,1:]=100
        with torch.no_grad():
            torch.testing.assert_close(model(x,mask),model(changed,mask))
        with self.assertRaises(ValueError):
            model(x,torch.zeros_like(mask))

    def test_spatial_groups_and_buffer(self):
        coords=np.array([[8+i*.2,48+j*.2] for i in range(15) for j in range(15)])
        y=np.arange(len(coords))%3
        splits,groups=spatial_split(coords,y)
        names=['train','val','test']
        for i,a in enumerate(names):
            for b in names[i+1:]:
                self.assertFalse(set(groups[splits[a]]) & set(groups[splits[b]]))
        self.assertTrue(all(len(index)>0 for index in splits.values()))

    def test_hierarchy_loss_backpropagates(self):
        logits=torch.randn(4,3,requires_grad=True)
        loss=hierarchical_loss(logits,torch.tensor([0,1,2,0]),torch.ones(3),
            torch.tensor([0,0,1]),torch.tensor([0,0,1]))
        loss.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())


if __name__=='__main__':
    unittest.main()
