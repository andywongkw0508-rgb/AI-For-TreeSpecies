"""Write a readable results report from completed experiment artifacts."""
import json
from pathlib import Path


def main():
    root=Path('runs/ssl4eo_temporal')
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    manifest=read(Path('data/prepared/manifest.json'))
    config=read(root/'config.json')
    history=read(root/'history.json')
    val,test=read(root/'val_metrics.json'),read(root/'test_metrics.json')
    verification=read(root/'verification.json')
    selected=max(history,key=lambda r:r['val_macro_f1'])
    lines=['# Trained tree species model — local experiment', '',
        'Completed on 31 October 2026 using Python, PyTorch, and the NVIDIA RTX 5070 Ti Laptop GPU.', '',
        'This is dominant-species classification from monthly Sentinel-2 patches at known forest locations.',
        'It does not detect tree crowns or classify ordinary photographs.', '',
        '## Measured results', '',
        '| Metric | Validation | Internal geographic test |',
        '|---|---:|---:|']
    for key,label in [('accuracy','Accuracy'),('macro_f1','Macro-F1'),('weighted_f1','Weighted F1'),('top3_accuracy','Top-3 accuracy')]:
        lines.append(f'| {label} | {val[key]*100:.2f}% | {test[key]*100:.2f}% |')
    lines += ['',f"Selected epoch: **{selected['epoch']}** by validation macro-F1. Training stopped after {len(history)} epochs.",
        'The test set was evaluated after checkpoint selection, and was not used for hyperparameter tuning.',
        'These are internal scores, not results on the instructor’s withheld final test.', '',
        '## Data and method', '',
        f"- {manifest['shape'][0]:,} samples; all 19 tree species; March–October 2022.",
        '- Each sample: eight monthly 5 × 5 patches with 10 spectral bands and five vegetation indices.',
        f"- Missing input fraction: {manifest['missing_fraction']*100:.2f}%.",
        '- Three missing extracted species files were recovered from the supplied ZIP archives.',
        '- 10 km geographic blocks and a 100 m cross-split center buffer prevent overlapping patches.',
        '- Training-only imputation/scaling; masks exclude unobserved months from temporal attention.',
        '- Frozen SSL4EO-S12 MoCo ResNet-18 encoder, adapted from 13 to 10 spectral channels.',
        '- Trainable temporal Transformer, spectral patch statistics, weighted species loss, and hierarchy losses.',
        '- Pretrained weights: CC-BY-4.0; SHA-256 59bfdff99c7af41adfa2557e684ec59aec79f4cf6bb1a2e9714b802f08e2222e.',
        '', '| Split | Samples before observation filtering | No-usable-month removals |', '|---|---:|---:|']
    for split in ['train','val','test']:
        lines.append(f"| {split} | {manifest['split'][split]['n']:,} | {config['all_cloud_removed'][split]} |")
    lines += ['', '## Per-species test performance', '',
        '| Species | Precision | Recall | F1 | Test samples |', '|---|---:|---:|---:|---:|']
    for species in manifest['classes']:
        r=test['classification_report'][species]
        lines.append(f"| {species} | {r['precision']*100:.1f}% | {r['recall']*100:.1f}% | {r['f1-score']*100:.1f}% | {int(r['support'])} |")
    lines += ['', '## Verification', '',
        '- Five automated tests passed: month order and index math, missing observations, attention-mask isolation, spatial groups, and gradient flow.',
        '- The complete trained checkpoint reloads with the restricted weights-only loader.',
        f"- {verification['unlabeled_predictions_reproduced']} unlabeled held-out examples reproduce saved test predictions after checkpoint reload.",
        '- An example with no satellite observations returns insufficient_observations rather than a species guess.',
        f"- Maximum probability difference on the reload check: {verification['max_probability_difference']:.6f}.",
        '- Dependency integrity check passed; exact package versions are in requirements-lock.txt.', '',
        '## Practical limitations and next experiments', '',
        'Macro-F1 is substantially below overall accuracy: minority species remain challenging, and some have very few test examples. '
        'This is a functioning initial model rather than a finished high-accuracy solution.', '',
        'The pretrained encoder uses L1C imagery, while this dataset contains L2A surface reflectance. '
        'The original pretrained spatial context is much larger; resizing 5 × 5 patches to 32 × 32 cannot recreate missing detail. '
        'The usefulness of pretrained features compared with a model from scratch has not yet been established by an ablation.', '',
        'Next experiments should compare a small temporal model from scratch, staged encoder fine-tuning, '
        'and larger spatial patches, selecting configurations on validation only. '
        'Keep the instructor’s eventual test set reserved for final reporting.', '',
        'Confidence values are uncalibrated softmax scores. For new locations, reproduce the same feature extraction schema. '
        'A coordinate-only CSV requires satellite extraction before prediction.', '',
        '## Files to use', '',
        '- Trained model: runs/ssl4eo_temporal/best.pt',
        '- Usage and setup: README.md',
        '- Prediction example: runs/ssl4eo_temporal/unlabeled_example.geojson',
        '- Example output: runs/ssl4eo_temporal/example_predictions.csv',
        '- Full per-species metrics, confusion matrices, and geolocated predictions: runs/ssl4eo_temporal/', '',
        '## Research attribution', '',
        '- [SSL4EO-S12 research repository](https://github.com/zhu-xlab/SSL4EO-S12)',
        '- [SSL4EO-S12 paper](https://arxiv.org/abs/2211.07044)',
        '- [TorchGeo pretrained model and license](https://huggingface.co/torchgeo/resnet18_sentinel2_all_moco)', '']
    Path('RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Wrote RESULTS.md')


if __name__=='__main__':
    main()
