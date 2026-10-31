"""Stream Earth Engine patches into a reproducible local dataset. No GPU required."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import ijson
import numpy as np
from scipy.spatial import cKDTree
from sklearn.model_selection import GroupShuffleSplit

BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
INDICES = ['NDVI', 'EVI', 'EVI2', 'SAVI', 'NDWI']
HIERARCHY = {
    'european beech': ('broadleaf', 'beech'),
    'sessile oak': ('broadleaf', 'oak'), 'english oak': ('broadleaf', 'oak'),
    'red oak': ('broadleaf', 'oak'),
    **{s: ('broadleaf', 'long-lived deciduous') for s in ['sycamore maple', 'european ash', 'linden', 'cherry']},
    **{s: ('broadleaf', 'short-lived deciduous') for s in ['alder', 'poplar', 'birch']},
    'silver fir': ('needleleaf', 'fir'),
    **{s: ('needleleaf', 'larch') for s in ['european larch', 'japanese larch']},
    'norway spruce': ('needleleaf', 'spruce'),
    **{s: ('needleleaf', 'pine') for s in ['scots pine', 'black pine', 'weymouth pine']},
    'douglas fir': ('needleleaf', 'douglas fir'),
}


def features(path):
    with Path(path).open('rb') as stream:
        yield from ijson.items(stream, 'features.item', use_float=True)


def parse_patch(feature):
    """Return [month, channel, row, col]; nulls remain NaN, not fake reflectance."""
    props = feature['properties']
    x = np.full((8, 15, 5, 5), np.nan, dtype=np.float32)
    for month in range(8):
        suffix = '' if month == 0 else f'_{month}'
        for channel, band in enumerate(BANDS + INDICES):
            value = props.get(band + suffix)
            if value is None:
                continue
            array = np.asarray(value, dtype=np.float32)
            if array.shape != (5, 5):
                raise ValueError(f'{band + suffix}: expected a 5x5 patch, got {array.shape}')
            x[month, channel] = array
    # Recompute indices from the reflectance used by the model, consistently at inference.
    b = {name: x[:, i] for i, name in enumerate(BANDS)}
    def ratio(a, denominator):
        return np.divide(a, denominator, out=np.full_like(a, np.nan), where=np.abs(denominator) > 1e-6)
    nir, red, blue, swir = b['B8'], b['B4'], b['B2'], b['B11']
    x[:, 10] = ratio(nir-red, nir+red)
    x[:, 11] = ratio(2.5*(nir-red), nir+6*red-7.5*blue+1)
    x[:, 12] = ratio(2.5*(nir-red), nir+2.4*red+1)
    x[:, 13] = ratio(1.5*(nir-red), nir+red+0.5)
    x[:, 14] = ratio(nir-swir, nir+swir)
    # Earth Engine reflectances are already divided by 10000. Do not divide again.
    x[~np.isfinite(x)] = np.nan
    return x


def recover_missing(folder, archive_dir):
    """Copy only absent canonical species files; never extract arbitrary archive paths."""
    for archive in sorted(Path(archive_dir).glob('*.zip')):
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                base = Path(name).name
                if not base.endswith('.geojson'):
                    continue  # ignore duplicate exports named .geojson(1)
                target = folder / base
                if not target.exists():
                    print(f'Recovering {base} from {archive.name}', flush=True)
                    with z.open(name) as source, target.open('wb') as dest:
                        shutil.copyfileobj(source, dest)


def spatial_split(coords, y, seed=42):
    # Locally metric approximation; adequate for coarse 10 km blocks within Germany.
    xy = np.column_stack([coords[:, 0]*111320*np.cos(np.deg2rad(51)), coords[:, 1]*111320])
    cells = np.floor(xy / 10000).astype(np.int64)
    _, groups = np.unique(cells, axis=0, return_inverse=True)
    best = None
    for attempt in range(100):
        trainval, test = next(GroupShuffleSplit(n_splits=1, test_size=.15, random_state=seed+attempt).split(y, y, groups))
        a, b = next(GroupShuffleSplit(n_splits=1, test_size=.15/.85, random_state=seed+1000+attempt).split(y[trainval], y[trainval], groups[trainval]))
        train, val = trainval[a], trainval[b]
        # Remove nearby patches across boundaries to avoid overlapping 50 m windows.
        val = val[cKDTree(xy[test]).query(xy[val])[0] >= 100]
        train = train[cKDTree(xy[np.r_[val, test]]).query(xy[train])[0] >= 100]
        missing = sum(len(set(y)-set(y[index])) for index in [train, val, test])
        score = missing*100 + sum(abs(len(index)/len(y)-fraction) for index, fraction in zip([train,val,test],[.7,.15,.15]))
        if best is None or score < best[0]:
            best = (score, train, val, test)
        if missing == 0 and score < .06:
            break
    _, train, val, test = best
    if set(y[train]) != set(y):
        raise ValueError('A class has no spatially independent training samples. Inspect data and split.')
    return {'train': train, 'val': val, 'test': test}, groups


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, default=Path('TreeSatAI_Label_Data/TreeSatAI'))
    p.add_argument('--output', type=Path, default=Path('data/prepared'))
    p.add_argument('--recover-missing', action='store_true')
    args = p.parse_args()
    if args.output.exists():
        metadata = {'manifest.json', 'splits.npz', 'normalization.npz'}
        if any(path.name not in metadata for path in args.output.iterdir()):
            raise FileExistsError(f'{args.output} already contains prepared data; choose a fresh --output.')
    if args.recover_missing:
        recover_missing(args.input, Path('.'))
    classes = sorted(HIERARCHY)
    xs, ys, coordinates, ids, seen = [], [], [], [], {}
    counts, invalid, duplicates = Counter(), 0, 0
    sources = []
    for path in sorted(args.input.glob('*.geojson')):
        species_from_file = path.stem.split('_', 2)[-1]
        count = 0
        for f in features(path):
            species = f['properties'].get('l3_species', species_from_file)
            if species not in HIERARCHY:
                raise ValueError(f'Unknown species: {species}')
            coord = np.asarray(f['geometry']['coordinates'][:2], dtype=float)
            if not np.isfinite(coord).all() or not (-180 <= coord[0] <= 180 and -90 <= coord[1] <= 90):
                raise ValueError('Expected WGS84 point coordinates in GeoJSON.')
            key = tuple(np.round(coord, 7))
            if key in seen:
                if seen[key] != species:
                    raise ValueError(f'Conflicting labels at {key}')
                duplicates += 1
                continue
            patch = parse_patch(f)
            if not np.isfinite(patch[:, :10]).any():
                invalid += 1
                continue
            seen[key] = species
            xs.append(patch); ys.append(classes.index(species)); coordinates.append(coord)
            ids.append(str(f.get('id', len(ids))))
            counts[species] += 1; count += 1
        sources.append({'file': str(path), 'bytes': path.stat().st_size, 'accepted': count})
        print(f'{path.name}: {count} accepted', flush=True)
    if not xs:
        raise ValueError('No usable labeled patches found.')
    x, y, coords = np.stack(xs), np.asarray(ys, dtype=np.int64), np.asarray(coordinates)
    splits, groups = spatial_split(coords, y)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    for name, array in [('x',x),('y',y),('coords',coords),('ids',np.asarray(ids)),('groups',groups)]:
        np.save(out / f'{name}.npy', array)
    np.savez(out / 'splits.npz', **splits)
    train_x = x[splits['train']]
    with np.errstate(invalid='ignore'):
        mean = np.nanmean(train_x, axis=(0,1,3,4))
        std = np.nanstd(train_x, axis=(0,1,3,4))
    mean = np.nan_to_num(mean, nan=0).astype(np.float32)
    std = np.maximum(np.nan_to_num(std, nan=1), .001).astype(np.float32)
    np.savez(out / 'normalization.npz', mean=mean, std=std)
    manifest = {'classes': classes, 'hierarchy': HIERARCHY, 'bands': BANDS+INDICES,
        'months': ['2022-%02d' % m for m in range(3,11)], 'shape': list(x.shape),
        'counts': dict(counts), 'missing_classes': sorted(set(classes)-set(counts)),
        'invalid_samples_removed': invalid, 'duplicates_removed': duplicates,
        'missing_fraction': float(np.isnan(x).mean()),
        'fingerprint': hashlib.sha256(x.tobytes()+y.tobytes()+coords.tobytes()).hexdigest(),
        'sources': sources, 'split': {}, 'split_method': '10km spatial blocks; >=100m cross-split center separation; seed 42'}
    for name,index in splits.items():
        manifest['split'][name] = {'n':len(index), 'counts':dict(Counter(classes[i] for i in y[index])),
            'missing_classes': [classes[i] for i in sorted(set(y)-set(y[index]))]}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps({k:manifest[k] for k in ['shape','counts','missing_classes','missing_fraction','split']},indent=2))


if __name__ == '__main__':
    main()
