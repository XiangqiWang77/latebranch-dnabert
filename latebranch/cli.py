"""Command-line entry points for explicit local data and model artifacts."""
import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from .data import read_table, check_partitions
from .runtime import download_backbone, load_checkpoint, ModelConfig


def new_output(path):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True,exist_ok=True)
    return path


def main():
    parser=argparse.ArgumentParser(description="LateBranch: allele-specific DNABERT-2 prediction")
    sub=parser.add_subparsers(dest="command",required=True)
    p=sub.add_parser('download-backbone',help='Download and verify the pinned official DNABERT-2 files')
    p.add_argument('--output',required=True)
    p=sub.add_parser('check-data',help='Validate centered sequence/count inputs and optional partitions')
    p.add_argument('--input',required=True);p.add_argument('--training',action='store_true')
    p=sub.add_parser('train',help='Fit on train rows; select checkpoints on validation rows only')
    p.add_argument('--input',required=True);p.add_argument('--config',required=True)
    p.add_argument('--backbone',required=True);p.add_argument('--output',required=True)
    p.add_argument('--device',default='cuda');p.add_argument('--threads',type=int,default=4)
    p=sub.add_parser('predict',help='Average forward/RC logits; optionally apply saved calibration')
    p.add_argument('--input',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--backbone',required=True);p.add_argument('--output',required=True)
    p.add_argument('--calibration');p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--threads',type=int,default=4)
    p=sub.add_parser('calibrate',help='Fit assay scales on rows labeled split=calibration')
    p.add_argument('--predictions',required=True);p.add_argument('--model-config',required=True)
    p.add_argument('--output',required=True)
    p=sub.add_parser('evaluate',help='Compute descriptive raw/calibrated metrics from saved predictions')
    p.add_argument('--predictions',required=True);p.add_argument('--output',required=True)
    args=parser.parse_args()
    if hasattr(args,'threads'):
        torch.set_num_threads(args.threads)
    if args.command=='download-backbone':
        print(download_backbone(args.output));return
    if args.command=='check-data':
        frame=read_table(args.input,counts=args.training)
        if args.training:check_partitions(frame)
        print(json.dumps(dict(valid=True,rows=len(frame))));return
    if args.command=='train':
        from .training import train
        print(json.dumps(train(read_table(args.input,counts=True),json.loads(Path(args.config).read_text()),
                               args.backbone,args.output,args.device)));return
    out=new_output(args.output)
    if args.command=='predict':
        from .inference import predict_frame
        model,tokenizer,config=load_checkpoint(args.checkpoint,args.backbone,args.device)
        calibration=json.loads(Path(args.calibration).read_text()) if args.calibration else None
        result=predict_frame(model,tokenizer,config,read_table(args.input),args.batch_size,calibration)
        result.drop(columns=['sequence']).to_csv(out,index=False)
    elif args.command=='calibrate':
        from .evaluation import fit_calibration
        result=fit_calibration(pd.read_csv(args.predictions),ModelConfig(**json.loads(Path(args.model_config).read_text())))
        out.write_text(json.dumps(result,indent=2)+'\n')
    else:
        from .evaluation import evaluate
        evaluate(pd.read_csv(args.predictions)).to_csv(out,index=False)
    print(out)


if __name__=='__main__':
    main()
