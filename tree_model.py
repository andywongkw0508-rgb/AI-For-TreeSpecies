"""SSL4EO-S12 transfer learning on eight monthly Sentinel-2 patches."""
from pathlib import Path
import warnings

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18

WEIGHTS_URL = ('https://huggingface.co/torchgeo/resnet18_sentinel2_all_moco/resolve/'
    '5b8cddc9a14f3844350b7f40b85bcd32aed75918/resnet18_sentinel2_all_moco-59bfdff9.pth')
# Original order: B1 B2 B3 B4 B5 B6 B7 B8 B8A B9 B10 B11 B12.
SELECTED_BANDS = [1,2,3,4,5,6,7,8,11,12]
FEATURE_DIM = 512 + 15*3 + 1


def make_encoder(pretrained=False, cache='models/pretrained'):
    encoder = resnet18(weights=None)
    encoder.conv1 = nn.Conv2d(10,64,7,stride=2,padding=3,bias=False)
    encoder.fc = nn.Identity()
    if pretrained:
        Path(cache).mkdir(parents=True, exist_ok=True)
        state = torch.hub.load_state_dict_from_url(WEIGHTS_URL, model_dir=cache,
            map_location='cpu', check_hash=True, weights_only=True)
        state = dict(state)
        state['conv1.weight'] = state['conv1.weight'][:, SELECTED_BANDS].clone()
        state.pop('fc.weight', None); state.pop('fc.bias', None)
        # strict=True prevents accidentally training with silently unmatched weights.
        encoder.load_state_dict(state, strict=True)
    encoder.requires_grad_(False)
    encoder.eval()
    return encoder


def spectral_features(raw, mean, std):
    valid = np.isfinite(raw)
    mask = valid[:,:,:10].mean(axis=(2,3,4)) >= .5
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        avg = np.nanmean(raw, axis=(-2,-1))
        spread = np.nanstd(raw, axis=(-2,-1))
    avg = (np.where(np.isfinite(avg),avg,mean)-mean)/std
    spread = np.nan_to_num(spread)/std
    center = (np.where(valid[:,:,:,2,2],raw[:,:,:,2,2],mean)-mean)/std
    stats = np.concatenate([avg,spread,center,valid.mean(axis=(2,3,4))[...,None]],axis=-1)
    stats = np.clip(stats, -10,10).astype(np.float32)
    filled = np.where(valid[:,:,:10],raw[:,:,:10],mean[:10,None,None])
    return np.clip(filled,0,1).astype(np.float32), stats, mask


@torch.inference_mode()
def extract_features(encoder, raw, mean, std, device, image_size=32, batch_size=64, progress=False):
    """Frozen encoder: original tiny patches resized, never claimed to add detail."""
    results, masks = [], []
    for start in range(0,len(raw),batch_size):
        filled, stats, mask = spectral_features(np.asarray(raw[start:start+batch_size]),mean,std)
        batch, months = filled.shape[:2]
        images = torch.from_numpy(filled.reshape(-1,10,5,5)).to(device)
        images = F.interpolate(images,size=(image_size,image_size),mode='bilinear',align_corners=False)
        with torch.autocast(device_type=device.type, enabled=device.type=='cuda'):
            embedding = encoder(images).float().reshape(batch,months,512).cpu().numpy()
        results.append(np.concatenate([embedding,stats],axis=-1))
        masks.append(mask)
        if progress and (start == 0 or (start//batch_size)%20==0):
            print(f'Encoded {min(start+batch_size,len(raw))}/{len(raw)} patches',flush=True)
    return np.concatenate(results), np.concatenate(masks)


class TemporalClassifier(nn.Module):
    def __init__(self, num_classes=19, dim=128, dropout=.2):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(FEATURE_DIM,dim),nn.LayerNorm(dim),nn.GELU())
        self.month_position = nn.Parameter(torch.randn(1,8,dim)*.02)
        layer = nn.TransformerEncoderLayer(dim,4,dim*2,dropout,batch_first=True,norm_first=True)
        self.temporal = nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        self.classifier = nn.Sequential(nn.LayerNorm(dim),nn.Dropout(dropout),nn.Linear(dim,num_classes))

    def forward(self, features, mask):
        if not mask.any(dim=1).all():
            raise ValueError('A sample has no usable month (>=50% valid spectral pixels).')
        sequence = self.project(features) + self.month_position
        sequence = self.temporal(sequence,src_key_padding_mask=~mask)
        pooled = (sequence*mask.unsqueeze(-1)).sum(1)/mask.sum(1,keepdim=True)
        return self.classifier(pooled)


def hierarchical_loss(logits, y, weights, species_to_genus, species_to_leaf, smoothing=.05):
    species_loss = F.cross_entropy(logits,y,weight=weights,label_smoothing=smoothing)
    logp = logits.log_softmax(dim=-1)
    loss = species_loss
    for mapping, factor in [(species_to_genus,.2),(species_to_leaf,.1)]:
        group_logp = torch.stack([torch.logsumexp(logp[:,mapping==i],dim=-1)
            for i in range(int(mapping.max())+1)],dim=-1)
        loss = loss + factor*F.nll_loss(group_logp,mapping[y])
    return loss
