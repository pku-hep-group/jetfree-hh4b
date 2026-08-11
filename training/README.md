## Training infrastructure

This directory contains the infrastructure required to reproduce the training of the 138-class classifier presented in [arXiv:2508.15048](https://arxiv.org/abs/2508.15048).

The training code is built on top of weaver-core, a PyTorch-based library designed for high-energy physics workflows. The weaver-core repository is included as a subdirectory within this folder. We use only its data-loading functionality: it reads selected branches from HEP ROOT files according to a YAML card, performs the requested preprocessing (including the calculation of intermediate variables with NumPy and Awkward Array expressions), and supports event-range partitioning for training. The model and training pipeline are independent of Weaver and are implemented with PyTorch and PyTorch Lightning in `model.py` and `run.py`, respectively.

We hope this setup makes the code easier to read and understand, and facilitates customization and adaptation to your specific needs.

### Prepare the dataset

The dataset must be downloaded separately. Because of its size, we cannot provide a public download location suitable for all users. Please contact the authors, who can share the data through XRootD or a similar service.

The distributed data contains two directories, `training/` and `inference/`. Place both under the `datasets` directory. For the physics processes they contain, how the samples were generated, and how they correspond to the dataset described in arXiv:2508.15048, see [`datasets/checklist.md`](./datasets/checklist.md).

### Training recipe

First, install the Python environment. You may follow the instructions in [`./weaver-core`](./weaver-core) to install the Weaver dependencies and a compatible PyTorch version. Then install the additional packages `onnx`, `lightning`, and `ptflops`.

To train on four A100 GPUs, run:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python run.py --mode train --devices 4
```

> [!TIP]
> - The default initial learning rate is `2e-3`, the per-GPU batch size is `512`, and training runs on four GPUs for 80 epochs. These settings reproduce the training used in arXiv:2508.15048. For single-GPU training, manually scale the learning rate to one quarter of its default value by adding `--start-lr 5e-4`. If you change the batch size with `--batch-size`, scale the learning rate accordingly to keep the optimization behavior comparable.
> - Training uses a fixed training/validation split inherited from weaver-core. The training set uses the `[0, 0.8)` event range of every ROOT file, while validation uses `[0.8, 1.0)`.
> - The model is the standard Particle Transformer architecture implemented in `model.py`.
> - The YAML card presented [here](./data/hh4b_resolved_newsp4_allparts_nosel_138clswtop.yaml) determines how the data loader prepares each training batch. `selection` defines the selection applied to the ROOT files, `new_variables` defines derived variables, and `inputs` defines the four input groups accepted by ParT, each containing a set of particle-level input features. `labels` defines the truth label used for classification.

After training, inspect the `outputs` and `tensorboard` directories. They contain the training logs, checkpoints, and TensorBoard metrics listed below.

<details>
<summary>TensorBoard metrics</summary>

| Tag | Frequency/content |
| --- | --- |
| `Loss/train_step` | training cross-entropy at logged steps |
| `Loss/train_epoch` | epoch-average training cross-entropy |
| `Acc/train_epoch` | epoch-average training accuracy |
| `LR/train_epoch` | learning rate once per epoch |
| `Loss/val_epoch` | epoch-average validation cross-entropy |
| `Acc/val_epoch` | epoch-average validation accuracy |
| `AUC_HH4b_vs_QCD/val_epoch` | validation signal-vs-QCD ROC AUC |
| `BkgRej0p8_HH4b_vs_QCD/val_epoch` | QCD background rejection at 80% signal efficiency |
| `BkgRej0p5_HH4b_vs_QCD/val_epoch` | QCD background rejection at 50% signal efficiency |
| `BkgRej0p3_HH4b_vs_QCD/val_epoch` | QCD background rejection at 30% signal efficiency |
| `BkgRej0p2_HH4b_vs_QCD/val_epoch` | QCD background rejection at 20% signal efficiency |
| `BkgRej0p1_HH4b_vs_QCD/val_epoch` | QCD background rejection at 10% signal efficiency |
| `ROC_HH4b_vs_QCD/val_epoch` | logarithmic ROC figure |
| `AUC_HH4b_vs_TTbar/val_epoch` | validation signal-vs-$t\bar{t}$ ROC AUC |
| `BkgRej0p8_HH4b_vs_TTbar/val_epoch` | $t\bar{t}$ background rejection at 80% signal efficiency |
| `BkgRej0p5_HH4b_vs_TTbar/val_epoch` | $t\bar{t}$ background rejection at 50% signal efficiency |
| `BkgRej0p3_HH4b_vs_TTbar/val_epoch` | $t\bar{t}$ background rejection at 30% signal efficiency |
| `BkgRej0p2_HH4b_vs_TTbar/val_epoch` | $t\bar{t}$ background rejection at 20% signal efficiency |
| `BkgRej0p1_HH4b_vs_TTbar/val_epoch` | $t\bar{t}$ background rejection at 10% signal efficiency |
| `ROC_HH4b_vs_TTbar/val_epoch` | logarithmic ROC figure |

</details>

### Prediction recipe

Prediction is also supported. In this repository we provide `checkpoints/model{0,1,2}.ckpt`, the PyTorch Lightning checkpoints for the three models used to construct the ensemble described in arXiv:2508.15048.

To run inference on selected files, first choose a file under `datasets/inference`, then specify it in the `build_file_specs` function in `run.py`, which points to the training dataset by default. Files can be assigned to named groups using the same `GROUP_NAME:FILE_PATTERN` syntax used for the training samples.

Run inference with:

```bash
python run.py \
  --mode predict \
  --ckpt-path checkpoints/model0.ckpt \
  --predict-output predict/model0.root
```

This creates `predict/model0_<GROUP_NAME>.root`. The output ROOT file contains `cls_index`, the integer truth label from 0 through 137, and `score_0` through `score_137`, the predicted class probabilities.

### ONNX export recipe

ONNX export is also supported. The exported ONNX files can be used directly in the subsequent Delphes macro analysis described in the parent directory's documentation for `delphes`. The three exported models correspond to `../delphes/models/HH4b/model{0,1,2}.onnx`.

Run ONNX export with:

```bash
python run.py \
  --mode export \
  --ckpt-path checkpoints/model0.ckpt \
  --export-output export/model0.onnx
```

### Command-line reference

All command-line options are listed below for reference.

```text
--mode {train,predict,export}  Operation to run (default: train)
--ckpt-path PATH               Checkpoint for resume, prediction, or export
--predict-output PATH          ROOT or Parquet prediction destination
--export-output PATH           ONNX destination; must end in .onnx
--stat-only                    Print model/parameter/FLOP information and exit
--smoke-test                   Limit train/val or prediction to one batch
--devices N                    Number of GPUs; defaults to 4 and is capped by availability
--num-nodes N                  Number of Lightning training nodes (default: 1)
--batch-size N                 Per-process batch size override (default: 512)
--start-lr LR                  Initial learning-rate override (default: 2e-3)
```
