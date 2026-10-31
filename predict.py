"""Predict species from new GeoJSONs with the same monthly Sentinel-2 schema."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from prepare_data import features,parse_patch
from tree_model import TemporalClassifier,extract_features,make_encoder


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,default=Path('runs/ssl4eo_temporal/best.pt'))
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,default=Path('runs/predictions.csv'))
    p.add_argument('--batch-size',type=int,default=64)
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a fresh output path.')
    torch.set_num_threads(4)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    encoder=make_encoder().to(device);encoder.load_state_dict(ckpt['encoder'])
    model=TemporalClassifier(len(ckpt['classes'])).to(device)
    model.load_state_dict(ckpt['head']);model.eval()
    mean,std=ckpt['mean'].numpy(),ckpt['std'].numpy()
    fmean,fstd=ckpt['feature_mean'].numpy(),ckpt['feature_std'].numpy()
    paths=sorted(args.input.glob('*.geojson')) if args.input.is_dir() else [args.input]
    if not paths:
        raise ValueError('No input GeoJSON files found.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def predict_batch(batch,writer):
        raw=np.stack([parse_patch(f) for f in batch])
        feat,mask=extract_features(encoder,raw,mean,std,device,ckpt['image_size'],args.batch_size)
        valid=mask.any(1)
        probs=np.zeros((len(batch),len(ckpt['classes'])),dtype=np.float32)
        if valid.any():
            feat=np.clip((feat[valid]-fmean)/fstd,-10,10).astype(np.float32)
            feat[~mask[valid]]=0
            with torch.inference_mode():
                probs[valid]=model(torch.from_numpy(feat).to(device),torch.from_numpy(mask[valid]).to(device)).softmax(-1).cpu().numpy()
        for f,usable,prob in zip(batch,valid,probs):
            coords=f['geometry']['coordinates'][:2]
            if usable:
                top=np.argsort(prob)[-3:][::-1]
                species=ckpt['classes'][top[0]]
                leaf,genus=ckpt['hierarchy'][species]
                writer.writerow([f.get('id',''),*coords,species,genus,leaf,float(prob[top[0]]),
                    json.dumps([[ckpt['classes'][i],float(prob[i])] for i in top]),'ok'])
            else:
                writer.writerow([f.get('id',''),*coords,'','','','','','insufficient_observations'])
    with args.output.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.writer(stream)
        writer.writerow(['id','longitude','latitude','species','genus','leaf_type','confidence','top3','status'])
        for path in paths:
            batch=[]
            for feature in features(path):
                batch.append(feature)
                if len(batch)==args.batch_size:
                    predict_batch(batch,writer);batch=[]
            if batch:
                predict_batch(batch,writer)
    print(f'Saved {args.output}')


if __name__=='__main__':
    main()
