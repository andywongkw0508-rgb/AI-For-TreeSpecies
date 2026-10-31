"""Check split isolation and reproduce predictions using the saved local model."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.spatial import cKDTree
import torch

from prepare_data import BANDS


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path('data/prepared'))
    parser.add_argument('--run',type=Path,default=Path('runs/ssl4eo_temporal'))
    args=parser.parse_args()
    ckpt=torch.load(args.run/'best.pt',map_location='cpu',weights_only=True)
    y=np.load(args.data/'y.npy'); coords=np.load(args.data/'coords.npy')
    groups=np.load(args.data/'groups.npy');splits=dict(np.load(args.data/'splits.npz'))
    xy=np.column_stack([coords[:,0]*111320*np.cos(np.deg2rad(51)),coords[:,1]*111320])
    checks={}
    names=list(splits)
    for i,a in enumerate(names):
        for b in names[i+1:]:
            assert not set(groups[splits[a]]) & set(groups[splits[b]]), 'Spatial block overlap'
            distance=float(cKDTree(xy[splits[a]]).query(xy[splits[b]])[0].min())
            assert distance>=100, 'Buffer violation'
            checks[a+'_'+b+'_minimum_distance_m']=distance
    raw=np.load(args.data/'x.npy',mmap_mode='r')
    ids=np.load(args.data/'ids.npy')
    # One genuinely held-out sample per species, stripped of all label properties.
    indices=[next(int(i) for i in splits['test'] if y[i]==label) for label in range(len(ckpt['classes']))]
    sample_features=[]
    for index in indices:
        properties={}
        for m in range(8):
            suffix='' if m==0 else f'_{m}'
            for c,band in enumerate(BANDS):
                a=raw[index,m,c]
                properties[band+suffix]=[[float(v) if np.isfinite(v) else None for v in row] for row in a]
        sample_features.append({'type':'Feature','id':str(ids[index]),
            'geometry':{'type':'Point','coordinates':coords[index].tolist()},'properties':properties})
    sample_features.append({'type':'Feature','id':'no-observations',
        'geometry':{'type':'Point','coordinates':[10,51]},'properties':{}})
    example=args.run/'unlabeled_example.geojson'
    example.write_text(json.dumps({'type':'FeatureCollection','features':sample_features}),encoding='utf-8')
    prediction=args.run/'example_predictions.csv'
    # Predict uses exclusive creation; a prior verification output can be preserved.
    if prediction.exists():
        prediction=args.run/'example_predictions_recheck.csv'
    subprocess.run([sys.executable,'predict.py','--checkpoint',str(args.run/'best.pt'),
        '--input',str(example),'--output',str(prediction)],check=True)
    with prediction.open(encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    assert len(rows)==20
    assert rows[-1]['status']=='insufficient_observations'
    with (args.run/'test_predictions.csv').open(encoding='utf-8') as f:
        references={(r['id'],round(float(r['longitude']),7),round(float(r['latitude']),7)):r for r in csv.DictReader(f)}
    maximum_difference=0.
    for row in rows[:-1]:
        key=(row['id'],round(float(row['longitude']),7),round(float(row['latitude']),7))
        reference=references[key]
        assert row['species']==reference['predicted_species'], 'Reload changed prediction'
        difference=abs(float(row['confidence'])-float(reference['confidence']))
        maximum_difference=max(maximum_difference,difference)
        assert difference<.003, 'Reload changed probabilities beyond mixed-precision tolerance'
    checks.update({'checkpoint_load_weights_only':True,'unlabeled_predictions_reproduced':19,
        'missing_observation_rejected':True,'max_probability_difference':maximum_difference})
    (args.run/'verification.json').write_text(json.dumps(checks,indent=2),encoding='utf-8')
    print(json.dumps(checks,indent=2))


if __name__=='__main__':
    main()
