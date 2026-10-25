#!/usr/bin/env python3
"""Replay a released study comparator using real processed inputs and a pinned backbone."""
import argparse,json,os,sys
from pathlib import Path

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',type=Path,required=True,help='Unzipped processed/ directory')
    parser.add_argument('--bundle',type=Path,required=True,help='Unzipped models/ directory')
    parser.add_argument('--backbone',type=Path,required=True)
    parser.add_argument('--model',required=True,help='Registry key, e.g. chromosome__natural_offset_lb')
    parser.add_argument('--phase',choices=['validation','calibration','test'],default='test')
    parser.add_argument('--limit',type=int,default=64,help='0 means the complete partition')
    parser.add_argument('--device',default='cpu');parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    os.environ['LATEBRANCH_STUDY_DATA']=str(args.inputs.resolve())
    os.environ['LATEBRANCH_BACKBONE']=str(args.backbone.resolve())
    here=Path(__file__).resolve().parent;sys.path.insert(0,str(here/'runtime'))
    import numpy as np
    import torch
    from common import load_data
    from experiment import create_model,make_loader,predict
    from latebranch.runtime import verify_backbone,file_sha256
    torch.set_num_threads(args.threads);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    records=json.loads((here/'registry.json').read_text());record=next(r for r in records if r['key']==args.model)
    if record['family']=='matched':
        from matched_models import create_model,make_loader
    verify_backbone(args.backbone)
    frame,parts,sequences,rho=load_data(record['split'])
    model,tok=create_model(record['method'],frame,parts['train'],rho)
    folder=args.bundle/args.model;checkpoint=folder/'best_state.pt'
    if file_sha256(checkpoint)!=record['checkpoint_sha256']:raise ValueError('Changed comparator checkpoint')
    saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
    required={n for n,p in model.named_parameters() if p.requires_grad}
    if not required.issubset(saved):raise ValueError('Missing adapted parameters')
    missing=model.load_state_dict(saved,strict=False)
    if missing.unexpected_keys or any(not n.startswith('backbone.') for n in missing.missing_keys):raise ValueError(str(missing))
    model.to(args.device).eval();phase='val' if args.phase=='validation' else args.phase
    rows=parts[phase];rows=rows[:args.limit] if args.limit else rows
    output=predict(model,{s:make_loader(record['method'],frame,rows,sequences,tok,32,s) for s in ['forward','reverse']},args.device)
    archive_name='validation_predictions.npz' if phase=='val' else args.phase+'_predictions.npz'
    with np.load(folder/archive_name) as a:
        index={int(v):i for i,v in enumerate(a['row_index'])}
        expected=a['tau_raw' if phase=='test' else 'tau'][[index[int(v)] for v in rows]]
    np.testing.assert_allclose(output['tau'],expected,rtol=1e-4,atol=1e-4)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,**output)
    print(json.dumps(dict(passed=True,model=args.model,phase=args.phase,rows=len(rows),
        max_absolute_error=float(np.max(abs(output['tau']-expected))),checkpoint_sha256=record['checkpoint_sha256'])))

if __name__=='__main__':main()
