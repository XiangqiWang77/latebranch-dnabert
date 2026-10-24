"""Count-supervised fitting with validation-only checkpoint selection."""
from dataclasses import asdict
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import AlleleDataset, AlleleCollator, check_partitions
from .evaluation import fit_rho
from .inference import predict_frame, move_batch
from .likelihood import nll_numpy, nll_torch
from .runtime import ModelConfig, build_model, save_checkpoint


def train(frame, settings, backbone_path, output, device="cuda"):
    check_partitions(frame)
    seed = int(settings.get("seed",13))
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    training = frame[frame.split.eq("train")].reset_index(drop=True)
    validation = frame[frame.split.eq("validation")].reset_index(drop=True)
    assays = settings.get("assays", sorted(training.assay.unique().tolist()))
    tissues = sorted(training.tissue.unique().tolist())
    if set(assays) != set(training.assay):
        raise ValueError("Configured assays must match the training vocabulary")
    rho = settings.get("rho")
    if rho is None:
        rho = fit_rho(training, assays)
    config = ModelConfig(assays=assays, tissues=tissues, rho=rho, seed=seed,
                        model_type=settings.get("model_type","latebranch"),
                        branch_layer=int(settings.get("branch_layer",8)),
                        position_aware=bool(settings.get("position_aware",True)),
                        train_last_n=int(settings.get("train_last_n",4)),
                        max_tokens=int(settings.get("max_tokens",128)))
    # Reject unknown held-out covariates before loading the large backbone.
    AlleleDataset(validation, config)
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    (output/'config.json').write_text(json.dumps(asdict(config),indent=2)+'\n')
    (output/'training_settings.json').write_text(json.dumps(settings,indent=2)+'\n')
    model, tokenizer = build_model(config, backbone_path)
    model.to(device)
    batch_size = int(settings.get("batch_size",128))
    loader = DataLoader(AlleleDataset(training,config), batch_size=batch_size, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(seed),
                        collate_fn=AlleleCollator(tokenizer,config,"random",seed))
    named = [(name,p) for name,p in model.named_parameters() if p.requires_grad]
    head = [p for name,p in named if not name.startswith('backbone.')]
    backbone = [p for name,p in named if name.startswith('backbone.')]
    groups = [dict(params=head,lr=float(settings.get('head_lr',1e-4)))]
    if backbone:
        groups.append(dict(params=backbone,lr=float(settings.get('backbone_lr',1e-5))))
    optimizer = torch.optim.AdamW(groups,weight_decay=float(settings.get('weight_decay',.01)))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=.5,patience=2)
    best, stale, history = float('inf'), 0, []
    max_epochs = int(settings.get('max_epochs',8))
    if max_epochs < 1 or batch_size < 1:
        raise ValueError("Positive epochs/batch size required")
    for epoch in range(1,max_epochs+1):
        started = time.monotonic(); model.train(); model.backbone.eval()
        for parameter in backbone:
            parameter.requires_grad_(epoch > int(settings.get('head_only_epochs',1)))
        total, count = 0., 0
        for raw in loader:
            batch = move_batch(raw,device); optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=str(device).startswith('cuda')):
                result = model(batch)
            loss = nll_torch(batch['n_alt'],batch['total'],result['tau'],result['rho']).mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for _,p in named],1.,error_if_nonfinite=True)
            optimizer.step(); size=len(batch['ref_id']); total+=float(loss.detach())*size; count+=size
        prediction = predict_frame(model,tokenizer,config,validation,batch_size)
        score=float(nll_numpy(prediction.n_alt.to_numpy(),(prediction.n_alt+prediction.n_ref).to_numpy(),
                              prediction.tau_raw.to_numpy(),prediction.rho.to_numpy()).mean())
        scheduler.step(score)
        record=dict(epoch=epoch,train_nll=total/count,validation_nll=score,seconds=time.monotonic()-started)
        history.append(record);print(json.dumps(record),flush=True)
        (output/'history.json').write_text(json.dumps(history,indent=2)+'\n')
        if score < best-float(settings.get('min_delta',1e-5)):
            best,stale=score,0
            save_checkpoint(output/'model.pt',model,config,best_epoch=epoch,validation_nll=best)
            prediction.drop(columns=['sequence']).to_csv(output/'validation_predictions.csv.gz',index=False)
        else:
            stale+=1
        if epoch >= int(settings.get('min_epochs',3)) and stale >= int(settings.get('patience',3)):
            break
    return dict(output=str(output),epochs=len(history),best_validation_nll=best,
                calibration_evaluated=False,test_evaluated=False)
