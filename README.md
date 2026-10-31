# Tree species classification with PyTorch

This project classifies the **dominant tree species at a supplied forest sample location**
using the course's monthly Sentinel-2 patches. It does not locate individual tree crowns
or accept ordinary phone photographs. It supports all 19 project species.

## Existing local installation

Run commands in PowerShell from this project directory. The project has its own `.venv`;
activation is unnecessary:

```powershell
# Train a new experiment (choose a fresh output directory for each run).
& .\.venv\Scripts\python.exe train.py --output runs/experiment_02

# Predict on a new Earth Engine GeoJSON with the SAME band/month/patch schema.
& .\.venv\Scripts\python.exe predict.py --checkpoint runs/ssl4eo_temporal/best.pt --input "path\to\new_samples.geojson" --output runs/new_predictions.csv

# Run the checks.
& .\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

`runs/ssl4eo_temporal/best.pt` is produced by the initial training run. Consult
`RESULTS.md` for the actual completed experiment, scores, and any limitations.

## Architecture and pretrained research

The image encoder is ResNet-18 pretrained with MoCo on **SSL4EO-S12 Sentinel-2**,
published by Zhu's research group. It uses the TorchGeo-converted weights:

- Research and citation: https://github.com/zhu-xlab/SSL4EO-S12
- Model and CC-BY-4.0 license: https://huggingface.co/torchgeo/resnet18_sentinel2_all_moco
- Weight metadata and band order: https://github.com/torchgeo/torchgeo/blob/main/torchgeo/models/resnet.py
- Paper: https://arxiv.org/abs/2211.07044

The pinned weight file is downloaded once into `models/pretrained/` and verified
against SHA-256 prefix `59bfdff9`. All matching ResNet weights load strictly.
The input convolution selects the 10 supplied spectral bands from the original 13,
dropping B1/B9/B10; it does not confuse B8 with B8A. No ImageNet RGB substitution.

The encoder stays frozen, including batch-normalization statistics. Each month is
encoded independently. A trainable two-layer temporal Transformer combines the
eight embeddings with month position embeddings. It also receives per-band patch
mean, standard deviation, center value, and the observation fraction. The input
includes 10 spectral bands and five recomputed vegetation indices. Hierarchical
auxiliary losses aggregate species probabilities into genus/group and leaf type.
Class weights use square-root inverse frequencies from training samples only.

**Transfer limitations:** the encoder was pretrained on S2 L1C top-of-atmosphere
imagery and larger patches; this course uses L2A surface reflectance and 5x5 patches.
The 5x5 patches are bilinearly resized to 32x32 for the encoder. This adds no spatial
information. The pretrained model is an initialization/feature source, not an
already validated 19-species classifier. Validation performance must establish
whether transfer helps. An experiment from scratch is a useful future comparison.

## Data preparation and evaluation

```powershell
# Already completed for the supplied data. A fresh output is required to rerun.
& .\.venv\Scripts\python.exe prepare_data.py --recover-missing --output data/prepared_v2
```

Preparation streams the large GeoJSON files rather than loading their text at once.
It recovers absent canonical `.geojson` files from the two supplied ZIPs, ignores
duplicate `.geojson(1)` entries, validates 5x5 patches, and deduplicates coordinates.
Conflicting species labels at identical coordinates stop preparation for review.

Input tensor: `[sample, 8 months, 15 channels, 5 rows, 5 columns]`.
Month order: March through October 2022; first month has no suffix, then `_1` to `_7`.
Channel order: `B2 B3 B4 B5 B6 B7 B8 B8A B11 B12 NDVI EVI EVI2 SAVI NDWI`.
The exported reflectance is already scaled by 10000; it is not divided again.
NDWI here means `(B8-B11)/(B8+B11)`, following the course script.

Null observations remain missing until imputation with **training-only** band means.
A month needs at least 50% valid spectral pixels to count as observed. Temporal
attention masks missing months. Samples with no usable month are excluded during
training and marked `insufficient_observations` during inference. No artificial
species prediction is returned for those samples.

Evaluation uses 10 km geographic blocks, approximately 70/15/15 train/validation/test.
Patch centers across different splits must be at least 100 m apart. The metric
coordinates use a local approximation within Germany, not a survey-grade projection.
Split selection checks class coverage and sample proportions, never model scores.
Feature normalization is fitted only on observed training months. Early stopping
selects the epoch by validation macro-F1; the internal test is evaluated afterwards.
Nearby geographic patterns can still be correlated beyond 100 m: this split is a
practical guard against overlap, not proof of independence at every ecological scale.

The internal test is **not the instructor's withheld final test**. Keep that final
test untouched until the project evaluation. A final CSV containing only coordinates
and labels must first have the same Sentinel-2 features extracted in Earth Engine;
coordinates alone are not valid model inputs. Match the dates/features to the final
test requirements when they are released.

## Outputs

- `data/prepared/manifest.json`: class counts, missingness, provenance, split counts.
- `data/prepared/*.npy`: patch tensors, labels, coordinates, IDs, spatial groups.
- `data/prepared/splits.npz`: reproducible split indices.
- `data/prepared/features_ssl4eo_32.npz`: frozen encoder feature cache.
- `runs/ssl4eo_temporal/best.pt`: self-contained model, encoder, labels, normalization.
- `runs/ssl4eo_temporal/config.json`: hyperparameters, versions, data fingerprint.
- `runs/ssl4eo_temporal/history.json`: training loss and validation metrics by epoch.
- `runs/ssl4eo_temporal/{val,test}_metrics.json`: accuracy, macro/weighted F1,
  top-3 accuracy, and per-species precision/recall/F1.
- `runs/ssl4eo_temporal/{val,test}_confusion.csv`: labeled confusion matrix.
- `runs/ssl4eo_temporal/{val,test}_predictions.csv`: geolocated held-out predictions.

Prediction confidence is the model's softmax score; it is not a calibrated probability
or a guarantee. Minority species have small test sample sizes; inspect per-class
support and recall alongside overall accuracy.

## Fresh installation on another computer

Install Python 3.12, then:

```powershell
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv\Scripts\python.exe download_weights.py
& .\.venv\Scripts\python.exe prepare_data.py --recover-missing
& .\.venv\Scripts\python.exe train.py --output runs/experiment_02
```

CUDA 12.8 wheels are used for the available NVIDIA RTX 5070. On a CPU-only machine,
use the CPU installation instructions at https://pytorch.org/get-started/locally/.
`requirements-lock.txt` records the environment used for the completed local run.
The initial trained checkpoint, experiment results, and small preparation metadata
are included in Git. Raw GeoJSON datasets, archives, large tensor/feature caches,
the downloaded original encoder, and the Python environment remain local.
