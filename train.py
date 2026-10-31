"""Train a temporal tree-species head using a published, frozen Sentinel-2 encoder."""
import argparse
import csv
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, TensorDataset

from tree_model import WEIGHTS_URL, TemporalClassifier, extract_features, hierarchical_loss, make_encoder


def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.inference_mode():
        for features, mask, y in loader:
            probs = model(features.to(device),mask.to(device)).softmax(-1)
            ys.append(y.numpy()); ps.append(probs.cpu().numpy())
    return np.concatenate(ys),np.concatenate(ps)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path('data/prepared'))
    parser.add_argument('--output',type=Path,default=Path('runs/ssl4eo_temporal'))
    parser.add_argument('--epochs',type=int,default=80)
    parser.add_argument('--patience',type=int,default=12)
    parser.add_argument('--batch-size',type=int,default=128)
    parser.add_argument('--encode-batch-size',type=int,default=64)
    parser.add_argument('--image-size',type=int,default=32)
    parser.add_argument('--lr',type=float,default=.0003)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a fresh --output directory to preserve previous results.')
    if args.epochs < 1:
        raise ValueError('--epochs must be positive')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    device = torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device)
    if device.type=='cuda':
        torch.cuda.manual_seed_all(args.seed)
    print(f'Using {device}; torch {torch.__version__}',flush=True)
    manifest = json.loads((args.data/'manifest.json').read_text(encoding='utf-8'))
    if manifest['missing_classes']:
        raise ValueError(f"Missing project classes: {manifest['missing_classes']}")
    classes = manifest['classes']; nclasses = len(classes)
    raw = np.load(args.data/'x.npy',mmap_mode='r')
    y = np.load(args.data/'y.npy')
    splits = dict(np.load(args.data/'splits.npz'))
    normalization = dict(np.load(args.data/'normalization.npz'))
    encoder = make_encoder(pretrained=True).to(device)
    cache = args.data/f'features_ssl4eo_{args.image_size}.npz'
    if cache.exists():
        saved = np.load(cache)
        if str(saved['fingerprint']) != manifest['fingerprint']:
            raise ValueError('Feature cache belongs to another dataset.')
        features, mask = saved['features'],saved['mask']
    else:
        features,mask = extract_features(encoder,raw,**normalization,device=device,
            image_size=args.image_size,batch_size=args.encode_batch_size,progress=True)
        np.savez(cache,features=features,mask=mask,fingerprint=manifest['fingerprint'])
    # All-cloud samples excluded before fitting, separately counted in the experiment.
    removed = {}
    for name,indices in splits.items():
        keep = mask[indices].any(axis=1)
        removed[name] = int((~keep).sum())
        splits[name] = indices[keep]
    if any(len(indices)==0 for indices in splits.values()):
        raise ValueError('A split is empty after missing-data filtering.')
    train_indices = splits['train']
    counts = np.bincount(y[train_indices],minlength=nclasses)
    if (counts==0).any():
        raise ValueError('A class has no usable training observation.')
    # Only valid training months fit the feature scaling.
    valid_features = features[train_indices][mask[train_indices]]
    fmean = valid_features.mean(axis=0)
    fstd = np.maximum(valid_features.std(axis=0),.001)
    normalized = np.clip((features-fmean)/fstd,-10,10).astype(np.float32)
    normalized[~mask] = 0
    loaders = {}
    for name,indices in splits.items():
        dataset = TensorDataset(torch.from_numpy(normalized[indices]),torch.from_numpy(mask[indices]),torch.from_numpy(y[indices]))
        loaders[name] = DataLoader(dataset,batch_size=args.batch_size,shuffle=name=='train',num_workers=0)
    model = TemporalClassifier(nclasses).to(device)
    weights = torch.as_tensor(np.minimum(np.sqrt(counts.max()/counts),10),dtype=torch.float32,device=device)
    genera = sorted({manifest['hierarchy'][s][1] for s in classes})
    leaves = sorted({manifest['hierarchy'][s][0] for s in classes})
    genus_map = torch.tensor([genera.index(manifest['hierarchy'][s][1]) for s in classes],device=device)
    leaf_map = torch.tensor([leaves.index(manifest['hierarchy'][s][0]) for s in classes],device=device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,args.epochs)
    args.output.mkdir(parents=True)
    config = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update({'device_used':str(device),'torch':str(torch.__version__),'encoder_weights':WEIGHTS_URL,
        'encoder_frozen':True,'dataset_fingerprint':manifest['fingerprint'],'all_cloud_removed':removed,
        'classes':classes,'split_counts':{k:len(v) for k,v in splits.items()}})
    (args.output/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    best, stale, history = -1,0,[]
    started = time.perf_counter()
    for epoch in range(1,args.epochs+1):
        model.train(); loss_sum=0; seen=0
        for f,m,t in loaders['train']:
            f,m,t = f.to(device),m.to(device),t.to(device)
            # Feature noise regularization; retain true observation masks.
            f = f + torch.randn_like(f)*.02*m.unsqueeze(-1)
            optimizer.zero_grad(set_to_none=True)
            logits = model(f,m)
            loss = hierarchical_loss(logits,t,weights,genus_map,leaf_map)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1)
            optimizer.step()
            loss_sum += loss.item()*len(t); seen += len(t)
        truth,probs = evaluate(model,loaders['val'],device)
        pred = probs.argmax(1)
        score = f1_score(truth,pred,labels=np.arange(nclasses),average='macro',zero_division=0)
        row = {'epoch':epoch,'train_loss':loss_sum/seen,'val_macro_f1':score,
            'val_accuracy':accuracy_score(truth,pred),'seconds':time.perf_counter()-started}
        history.append(row)
        print(json.dumps(row),flush=True)
        (args.output/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
        if score > best:
            best,stale = score,0
            torch.save({'head':{k:v.cpu().clone() for k,v in model.state_dict().items()},'epoch':epoch},args.output/'best_head.pt')
        else:
            stale+=1
        scheduler.step()
        if stale>=args.patience:
            break
    selected = torch.load(args.output/'best_head.pt',map_location='cpu',weights_only=True)
    model.load_state_dict(selected['head'])
    # Store encoder too: final checkpoint predicts without internet or weight download.
    checkpoint = {'head':selected['head'],'encoder':{k:v.cpu() for k,v in encoder.state_dict().items()},
        'classes':classes,'hierarchy':manifest['hierarchy'],'image_size':args.image_size,
        'mean':torch.from_numpy(normalization['mean']),'std':torch.from_numpy(normalization['std']),
        'feature_mean':torch.from_numpy(fmean),'feature_std':torch.from_numpy(fstd),
        'selected_epoch':selected['epoch'],'encoder_source':WEIGHTS_URL,'config':config}
    torch.save(checkpoint,args.output/'best.pt')
    for name in ['val','test']:
        truth,probs = evaluate(model,loaders[name],device)
        prediction = probs.argmax(1)
        report = classification_report(truth,prediction,labels=np.arange(nclasses),target_names=classes,output_dict=True,zero_division=0)
        metrics = {'accuracy':float(accuracy_score(truth,prediction)),
            'macro_f1':float(f1_score(truth,prediction,labels=np.arange(nclasses),average='macro',zero_division=0)),
            'weighted_f1':float(f1_score(truth,prediction,average='weighted',zero_division=0)),
            'top3_accuracy':float(np.mean([a in b for a,b in zip(truth,np.argsort(probs,axis=1)[:,-3:])])),
            'classification_report':report}
        (args.output/f'{name}_metrics.json').write_text(json.dumps(metrics,indent=2),encoding='utf-8')
        matrix = confusion_matrix(truth,prediction,labels=np.arange(nclasses))
        with (args.output/f'{name}_confusion.csv').open('w',newline='',encoding='utf-8') as stream:
            writer=csv.writer(stream);writer.writerow(['true / predicted']+classes)
            writer.writerows([[label]+row.tolist() for label,row in zip(classes,matrix)])
        coords=np.load(args.data/'coords.npy');ids=np.load(args.data/'ids.npy')
        with (args.output/f'{name}_predictions.csv').open('w',newline='',encoding='utf-8') as stream:
            writer=csv.writer(stream);writer.writerow(['id','longitude','latitude','true_species','predicted_species','confidence'])
            for i,t,p,prob in zip(splits[name],truth,prediction,probs):
                writer.writerow([ids[i],*coords[i],classes[t],classes[p],float(prob[p])])
        print(name,json.dumps({k:v for k,v in metrics.items() if k!='classification_report'}),flush=True)
    print(f"Saved self-contained trained model: {args.output/'best.pt'}",flush=True)


if __name__=='__main__':
    main()
