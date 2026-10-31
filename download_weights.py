"""Download the published SSL4EO-S12 encoder and verify its SHA-256 prefix."""
import hashlib
from pathlib import Path
import urllib.request

FILENAME='resnet18_sentinel2_all_moco-59bfdff9.pth'
URL=('https://huggingface.co/torchgeo/resnet18_sentinel2_all_moco/resolve/'
    '5b8cddc9a14f3844350b7f40b85bcd32aed75918/'+FILENAME)

if __name__=='__main__':
    folder=Path('models/pretrained');folder.mkdir(parents=True,exist_ok=True)
    path=folder/FILENAME
    if not path.exists():
        temporary=path.with_suffix('.download')
        print('Downloading published Sentinel-2 encoder...',flush=True)
        urllib.request.urlretrieve(URL,temporary)
        digest=hashlib.sha256(temporary.read_bytes()).hexdigest()
        if not digest.startswith('59bfdff9'):
            raise ValueError(f'Checksum mismatch: {digest}')
        temporary.replace(path)
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    if not digest.startswith('59bfdff9'):
        raise ValueError('Cached weights failed checksum verification.')
    print(f'{path}: SHA256 {digest}',flush=True)
