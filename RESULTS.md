# Trained tree species model — local experiment

Completed on 31 October 2026 using Python, PyTorch, and the NVIDIA RTX 5070 Ti Laptop GPU.

This is dominant-species classification from monthly Sentinel-2 patches at known forest locations.
It does not detect tree crowns or classify ordinary photographs.

## Measured results

| Metric | Validation | Internal geographic test |
|---|---:|---:|
| Accuracy | 66.21% | 67.85% |
| Macro-F1 | 54.53% | 54.66% |
| Weighted F1 | 66.15% | 67.67% |
| Top-3 accuracy | 87.94% | 88.69% |

Selected epoch: **10** by validation macro-F1. Training stopped after 22 epochs.
The test set was evaluated after checkpoint selection, and was not used for hyperparameter tuning.
These are internal scores, not results on the instructor’s withheld final test.

## Data and method

- 37,907 samples; all 19 tree species; March–October 2022.
- Each sample: eight monthly 5 × 5 patches with 10 spectral bands and five vegetation indices.
- Missing input fraction: 5.22%.
- Three missing extracted species files were recovered from the supplied ZIP archives.
- 10 km geographic blocks and a 100 m cross-split center buffer prevent overlapping patches.
- Training-only imputation/scaling; masks exclude unobserved months from temporal attention.
- Frozen SSL4EO-S12 MoCo ResNet-18 encoder, adapted from 13 to 10 spectral channels.
- Trainable temporal Transformer, spectral patch statistics, weighted species loss, and hierarchy losses.
- Pretrained weights: CC-BY-4.0; SHA-256 59bfdff99c7af41adfa2557e684ec59aec79f4cf6bb1a2e9714b802f08e2222e.

| Split | Samples before observation filtering | No-usable-month removals |
|---|---:|---:|
| train | 26,863 | 0 |
| val | 5,748 | 0 |
| test | 5,235 | 0 |

## Per-species test performance

| Species | Precision | Recall | F1 | Test samples |
|---|---:|---:|---:|---:|
| alder | 59.6% | 37.4% | 45.9% | 273 |
| birch | 61.3% | 78.9% | 69.0% | 227 |
| black pine | 41.7% | 30.3% | 35.1% | 66 |
| cherry | 18.8% | 31.6% | 23.5% | 19 |
| douglas fir | 82.0% | 66.3% | 73.3% | 309 |
| english oak | 46.9% | 50.0% | 48.4% | 224 |
| european ash | 54.9% | 51.5% | 53.1% | 340 |
| european beech | 62.1% | 74.9% | 67.9% | 530 |
| european larch | 40.0% | 48.1% | 43.7% | 129 |
| japanese larch | 79.8% | 63.7% | 70.8% | 223 |
| linden | 5.6% | 11.1% | 7.4% | 9 |
| norway spruce | 85.9% | 86.6% | 86.3% | 1150 |
| poplar | 43.4% | 63.9% | 51.7% | 36 |
| red oak | 55.8% | 76.3% | 64.4% | 190 |
| scots pine | 87.8% | 86.5% | 87.1% | 717 |
| sessile oak | 49.5% | 47.7% | 48.6% | 300 |
| silver fir | 74.4% | 67.7% | 70.9% | 99 |
| sycamore maple | 45.1% | 37.2% | 40.8% | 312 |
| weymouth pine | 51.2% | 50.0% | 50.6% | 82 |

## Verification

- Five automated tests passed: month order and index math, missing observations, attention-mask isolation, spatial groups, and gradient flow.
- The complete trained checkpoint reloads with the restricted weights-only loader.
- 19 unlabeled held-out examples reproduce saved test predictions after checkpoint reload.
- An example with no satellite observations returns insufficient_observations rather than a species guess.
- Maximum probability difference on the reload check: 0.000241.
- Dependency integrity check passed; exact package versions are in requirements-lock.txt.

## Practical limitations and next experiments

Macro-F1 is substantially below overall accuracy: minority species remain challenging, and some have very few test examples. This is a functioning initial model rather than a finished high-accuracy solution.

The pretrained encoder uses L1C imagery, while this dataset contains L2A surface reflectance. The original pretrained spatial context is much larger; resizing 5 × 5 patches to 32 × 32 cannot recreate missing detail. The usefulness of pretrained features compared with a model from scratch has not yet been established by an ablation.

Next experiments should compare a small temporal model from scratch, staged encoder fine-tuning, and larger spatial patches, selecting configurations on validation only. Keep the instructor’s eventual test set reserved for final reporting.

Confidence values are uncalibrated softmax scores. For new locations, reproduce the same feature extraction schema. A coordinate-only CSV requires satellite extraction before prediction.

## Files to use

- Trained model: runs/ssl4eo_temporal/best.pt
- Usage and setup: README.md
- Prediction example: runs/ssl4eo_temporal/unlabeled_example.geojson
- Example output: runs/ssl4eo_temporal/example_predictions.csv
- Full per-species metrics, confusion matrices, and geolocated predictions: runs/ssl4eo_temporal/

## Research attribution

- [SSL4EO-S12 research repository](https://github.com/zhu-xlab/SSL4EO-S12)
- [SSL4EO-S12 paper](https://arxiv.org/abs/2211.07044)
- [TorchGeo pretrained model and license](https://huggingface.co/torchgeo/resnet18_sentinel2_all_moco)
