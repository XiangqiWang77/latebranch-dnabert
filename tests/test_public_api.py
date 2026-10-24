"""Small functional checks; no pretrained download or scientific training run."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.stats import betabinom

from latebranch.data import read_table, check_partitions, encode_focal, AlleleDataset, AlleleCollator
from latebranch.evaluation import fit_calibration, evaluate
from latebranch.likelihood import nll_numpy, nll_torch
from latebranch.model import LateBranch
from latebranch.runtime import ModelConfig, adapted_keys
from latebranch._core import configure_backbone


class TinyTokenizer:
    cls_token_id, sep_token_id, pad_token_id = 1,2,0
    def __call__(self,sequences,**kwargs):
        if isinstance(sequences,str):sequences=[sequences]
        ids,offsets=[],[]
        for sequence in sequences:
            spans=[(i,min(i+3,len(sequence))) for i in range(0,len(sequence),3)]
            offsets.append(spans)
            ids.append([3+sum((j+1)*'ACGTN'.index(c) for j,c in enumerate(sequence[a:b])) for a,b in spans])
        return dict(input_ids=ids,offset_mapping=offsets)


class TinyEmbeddings(nn.Module):
    def __init__(self):
        super().__init__();self.word_embeddings=nn.Embedding(40,8)
    def forward(self,input_ids,token_type_ids=None):return self.word_embeddings(input_ids)


class TinyLayer(nn.Module):
    def __init__(self):super().__init__();self.linear=nn.Linear(8,8)
    def forward(self,value,*args,**kwargs):return value+.1*torch.tanh(self.linear(value))


class TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__();self.config=SimpleNamespace(hidden_size=8)
        self.embeddings=TinyEmbeddings();self.encoder=nn.Module()
        self.encoder.layer=nn.ModuleList([TinyLayer() for _ in range(12)])
        self.encoder.alibi=torch.zeros(1,1,128,128);self.encoder._current_alibi_size=128


def tiny_model(config):
    backbone=TinyBackbone();configure_backbone(backbone,False,config.train_last_n)
    model=LateBranch(backbone,len(config.tissues),len(config.assays),np.asarray(config.rho),
                     base_initialization=torch.randn(4,8),position_aware=True,branch_layer=8)
    model.rho=torch.tensor(config.rho,dtype=torch.float64)
    for module in model.modules():
        if isinstance(module,nn.Dropout):module.p=0
    return model,TinyTokenizer()


class PublicAPITests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(13)
        self.config=ModelConfig(assays=['CTCF'],tissues=['demo'],rho=[.1])

    def test_likelihood_and_gradient_match_independent_formula(self):
        k=np.array([0,1,25,99,10000]);n=np.array([10,10,100,100,10000]);tau=np.array([-3.,-.5,0.,2.,4.])
        p=1/(1+np.exp(-tau));rho=.1;concentration=(1-rho)/rho
        expected=-betabinom.logpmf(k,n,p*concentration,(1-p)*concentration)
        np.testing.assert_allclose(nll_numpy(k,n,tau,rho),expected,rtol=1e-9,atol=1e-9)
        t=torch.tensor(tau,requires_grad=True)
        result=nll_torch(torch.tensor(k),torch.tensor(n),t,torch.full((5,),rho,dtype=torch.float64))
        np.testing.assert_allclose(result.detach(),expected,rtol=1e-9,atol=1e-9)
        result.sum().backward();eps=1e-4
        difference=(nll_numpy(k,n,tau+eps,rho)-nll_numpy(k,n,tau-eps,rho))/(2*eps)
        np.testing.assert_allclose(t.grad,difference,rtol=1e-5,atol=1e-6)

    def test_focal_crop_retains_offset_and_reverse_identity(self):
        seq='AACCGGTTAAACCGG'
        encoded=encode_focal([seq],TinyTokenizer(),max_tokens=5)
        self.assertEqual(encoded['marker_span_length'].item(),3)
        self.assertEqual(encoded['marker_offset'].item(),1)
        self.assertEqual(encoded['attention_mask'].sum().item(),5)
        self.assertTrue(encoded['was_truncated'].item())
        frame=pd.DataFrame([dict(row_id='x',sequence=seq,ref='T',alt='T',assay='CTCF',tissue='demo')])
        data=AlleleDataset(frame,self.config);model,tok=tiny_model(self.config)
        for strand in ('forward','reverse'):
            batch=AlleleCollator(tok,self.config,strand)([data[0]])
            model.eval()
            self.assertEqual(model(batch)['tau'].item(),0.)
            update=model.candidate_update(batch['ref_id'],batch['marker_span_length'],batch['marker_offset'])
            self.assertTrue(torch.equal(update[0,batch['ref_id'][0]],torch.zeros(8)))

    def test_bad_reference_and_cross_partition_groups_rejected(self):
        frame=pd.DataFrame([dict(row_id='a',sequence='ACG',ref='A',alt='T',assay='CTCF',tissue='demo')])
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'input.csv';frame.to_csv(path,index=False)
            with self.assertRaisesRegex(ValueError,'REF does not match'):read_table(path)
            frame['ref']='C';frame['alt']='AC';frame.to_csv(path,index=False)
            with self.assertRaisesRegex(ValueError,'uppercase'):read_table(path)
        groups=pd.DataFrame(dict(group_id=['g','g'],split=['train','validation'],sequence=['AAA','CCC']))
        with self.assertRaisesRegex(ValueError,'crosses partitions'):check_partitions(groups)

    def test_calibration_uses_only_calibration_and_includes_zero_boundary(self):
        rows=pd.DataFrame(dict(split=['test']*30,assay=['CTCF']*30,n_ref=[5]*30,n_alt=[5]*30,tau_raw=[2.]*30,rho=[.1]*30))
        with self.assertRaisesRegex(ValueError,'split=calibration'):fit_calibration(rows,self.config)
        rows['split']='calibration';self.assertEqual(fit_calibration(rows,self.config)['scales'],[0.])

    def test_direction_ties_and_constant_correlations(self):
        frame=pd.DataFrame(dict(assay=['CTCF']*3,n_ref=[1,3,2],n_alt=[3,1,2],rho=[.1]*3,tau_raw=[0.]*3,tau=[0.]*3))
        row=evaluate(frame).iloc[0]
        self.assertEqual(row.direction_rows,2);self.assertEqual(row.direction_accuracy,.5)
        self.assertTrue(np.isnan(row.pearson));self.assertTrue(np.isnan(row.spearman))

    def test_tiny_training_keeps_test_rows_out_and_saves_all_adapted_parameters(self):
        from latebranch.training import train
        rng=np.random.default_rng(13);records=[]
        for i in range(8):
            seq=''.join(rng.choice(list('ACGT'),15));ref=seq[7]
            records.append(dict(row_id=str(i),group_id=str(i),sequence=seq,ref=ref,
                alt='ACGT'[('ACGT'.index(ref)+1)%4],assay='CTCF',tissue='unseen-test-only' if i>5 else 'demo',
                n_ref=4+i,n_alt=10-i,split='train' if i<4 else 'validation' if i<6 else 'test'))
        frame=pd.DataFrame(records)
        settings=dict(rho=[.1],assays=['CTCF'],max_epochs=1,min_epochs=1,batch_size=2,seed=13)
        def build(config,path):return tiny_model(config)
        with tempfile.TemporaryDirectory() as temporary,patch('latebranch.training.build_model',side_effect=build):
            out=Path(temporary)/'run';result=train(frame,settings,'unused',out,device='cpu')
            self.assertFalse(result['test_evaluated'])
            saved=torch.load(out/'model.pt',weights_only=True)
            config=ModelConfig(**saved['config']);model,_=tiny_model(config)
            self.assertEqual(set(saved['state_dict']),adapted_keys(model,config))
            validation=pd.read_csv(out/'validation_predictions.csv.gz')
            self.assertEqual(set(validation.row_id),{4,5})

    def test_archive_conversion_preserves_row_alignment_and_rejects_group_overlap(self):
        from scripts.export_study_data import export
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            # Sequence-array order deliberately differs from observation order.
            source=pd.DataFrame(dict(variant_id=[1,0],ref_id=[1,0],alt_id=[2,3],
                assay_canonical=['CTCF']*2,tissue=['demo']*2,n_ref=[4,6],n_alt=[5,7]))
            source.to_pickle(root/'frame.pkl.gz')
            np.save(root/'sequences.npy',np.array([[0,0,0],[1,1,1]],dtype=np.uint8))
            np.save(root/'components.npy',np.array([10,20]))
            np.savez(root/'partitions.npz',train=np.array([0]),val=np.array([1]))
            paths=[root/'frame.pkl.gz',root/'sequences.npy',root/'partitions.npz',root/'components.npy']
            self.assertEqual(export(*paths,root/'output.csv.gz'),2)
            converted=read_table(root/'output.csv.gz',counts=True);check_partitions(converted)
            self.assertEqual(converted.sequence.tolist(),['CCC','AAA'])
            self.assertEqual(converted.split.tolist(),['train','validation'])
            self.assertEqual(converted.n_alt.tolist(),[5,7])
            np.save(root/'components.npy',np.array([10,10]))
            with self.assertRaisesRegex(ValueError,'crosses partitions'):
                export(*paths,root/'rejected.csv.gz')
            self.assertFalse((root/'rejected.csv.gz').exists())


if __name__=='__main__':unittest.main()
