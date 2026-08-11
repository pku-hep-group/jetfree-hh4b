# jetfree-hh4b

This repository provides the source code and trained models associated with the study presented in [arXiv:2508.15048](https://arxiv.org/abs/2508.15048) (Potential of di-Higgs observation via a calibratable jet-free $HH\to 4b$ framework).

## Repository structure

### 1. `gen_configs`

This directory contains the configuration files for generating various Monte Carlo samples. The samples are organized into the following categories.

For detailed information about sample generation, see Supplementary Material Section B.1 of the paper.

#### `gen_configs/hh_incl`

Baseline Higgs boson pair (HH) samples generated without phase-space cuts. This includes:

- **ggF HH**: Gluon-gluon fusion HH production using standard Powheg settings, including modifications of $\kappa_\lambda$ to 0, 1, and 5 for both SM and BSM scenarios. These samples use official Powheg configurations (not included in this repository).

- **VBF HH**: Vector boson fusion HH production using MadGraph settings, including 6 different scenarios with modifications of $C_V$, $C_{2V}$, and $\kappa_\lambda$. The generation details are provided in the configuration files.

#### `gen_configs/sm_incl`

Baseline background samples generated in inclusive mode (without phase-space cuts). This includes:

- **$t\bar{t}$**
- **Single-top** (including $s$- and $t$-channel, and $tW$ modes)
- **Diboson** ($WW$, $WZ$, and $ZZ$)
- **$t\bar{t}W$** and **$t\bar{t}Z$**

The samples shown here are implemented using MadGraph and Pythia generators, and include both MadGraph and Pythia configuration files. Additional samples based on Powheg use official settings, including Higgs-related samples ($ggH$, $WH$, $ZH$, $t\bar{t}H$), which are not included in this repository.

#### `gen_configs/spec`

Special-purpose samples:

- **`QCD_DelphesHH4JTrig`**: QCD events generated using Pythia, including:
  - Pythia-level `pTHat` selection criteria
  - Generator-level jet $p_T$ and $H_T$ conditions applied via Pythia event filters
  - Optimized for the resolved "4j3b" trigger phase space (specifically the "4j" condition) to increase generation efficiency

- **`ZJetsToQQ_DelphesHH4JTrig`**: $Z$+jets to $q\bar{q}$ production using MadGraph+Pythia, including:
  - Matrix element-level phase-space requirements on parton $p_T$
  - Optimized for the resolved "4j3b" trigger phase space (specifically the "4j" condition) to increase generation efficiency

- **`HH4b_2HDM_H3VAR_H1H2_40to200`**: Variable mass $h_3 \to h_1 h_2 \to 4b$ samples used for training the jet-free model

### 2. `delphes`

This directory contains all scripts for fast detector simulation using Delphes.

#### `delphes/cards`

- **`delphes_card_CMS_JetClassII_lite.tcl`**: The Delphes configuration card used for high-fidelity simulation (corresponds to Supplementary Material Section A.1).

#### `delphes/readers`

Scripts for processing Delphes HepMC2→ROOT file generation. For large samples such as QCD, filters can be applied both before and after Delphes processing to improve efficiency:

- **Pre-Delphes filters**: Applied to generator-level stable particles, requiring total $p_T$ and $H_T$ cuts to exclude events that will definitely not pass the trigger. Implemented in `DelphesHepMC2WithFilterAK4.cpp`.

- **Post-Delphes filters**: Applied directly to reconstructed jets, requiring the "4j" trigger $p_T$ and $H_T$ conditions. Only events passing the filters are stored in the ROOT files.

**Note**: These filters are applied only to QCD samples.

#### `delphes/ana`

Delphes macro-based analyzers for converting Delphes ROOT files into ntuples. In addition to utility files that encapsulate basic functionality, the key analyzer file is:

- **`makeNtuplesHH4bAllObjectsOptionalSel.C`**: Reconstructs and stores the following variables:
  - **Particle-flow candidate particles** (`part_*`): All particles after pileup mitigation, which serve as inputs for jet and fatjet reconstruction in Delphes. Stores basic variables and information about which jet/fatjet they were reconstructed into (if any).
  - **Jets** (`jet_*`): Stores basic variables and b/c/light tagger scores from SophonAK4 model inference.
  - **Large-R jets/fatjets** (`fj_*`): Stores basic variables, substructure variables, and Xbb tagging scores from Sophon model inference.
  - **Leptons** (`lep_*`): Stores basic information for electrons and muons.
  - **Generator-level information**:
    - Gen-level Higgs variables for HH processes (`gen_higgs(1|2)_*`)
    - Variables for first-generation b-hadrons after hadronization (`gen_bhadron_*`)
    - Variables for key resonant gen-particles such as $t$, $W$, $Z$, and $H$ (`genpart_*`)

#### `delphes/models`

ONNX models used for the model inference described above:

- **`Sophon`** and **`SophonAK4`**: large-R and small-R jet tagging models for reproducing CMS resolved and boosted results
- **`HH4b`**: Three jet-free strategy models (138-class classification) used for ensemble inference

**Note**: Inference using all reconstructed particles for jet-free models is performed using ntuples as input through an additional ML R&D framework (the [Weaver](https://github.com/hqucms/weaver-core) framework). This framework is used to train these models and produce 138-dimensional outputs for inference samples. The framework itself is not included in this repository, but the trained models are provided in `delphes/models/HH4b/`.

### 3. `training`

This repository contains all code related to model training. See [`./training`](./training).

### 4. `ana_scripts`

This directory contains Jupyter notebooks and Python scripts that use the ntuples and the jet-free model inference scores (from the additional ML framework) to reproduce the figures shown in the paper.

#### `ana_scripts/dcb_fit`

Scripts for fitting the 136 signal values from the jet-free model to obtain the best-fit mass prediction $(\hat{m}_{h_1}, \hat{m}_{h_2})$ (corresponds to Supplementary Material Section C.2). By inputting the ML framework output containing the 138-dimensional model outputs, the fit can be performed and the best-fit prediction can be obtained.

- **`dcb_fit_single_job.py`**: Main script for performing the double-sided crystal ball (DCB) fit on single events

#### Notebooks

The Python notebooks can be run to reproduce the key analysis figures in the paper, as indicated by their filenames:
- **`PreProcessing.ipynb`**: Preprocessing scripts for the input ntuples in the following notebooks
- **`FIG1_ggHHvsQCD_ROC.ipynb`**: ROC curves for ggHH signal vs QCD background, comparing our and established $HH\to 4b$ strategies
- **`FIG2_2DHistForMainProcesses.ipynb`**: 2D histograms of the best-fit $(\hat{m}_{h_1}, \hat{m}_{h_2})$ for key physics processes
- **`FIG3a_LorentzBoostTest.ipynb`**: Study model performance across multiple Lorentz boosts
- **`FIG3b_HadronizationTest.ipynb`**: Study impact from hadronization models
- **`AppendixFIGS4_PlotsFromTensorboard.ipynb`**: Additional plots to show training metrics from TensorBoard output
- **`AppendixFIGS5_dcbFitSingleEvent.ipynb`**: DCB fit visualization for single events
- **`AppendixFIGS6_ggHHScatter.ipynb`**: Scatter plot of the corrected mass prediction points obtained with our strategy for the ggF HH signal
- **`AppendixFIGS7_ModelComparison.ipynb`**: ROC curves comparing the performance of individual jet-free ParT models and their ensemble combination for the ggF
$HH\to 4b$ signal against the QCD multijet background
- **`AppendixFIGS8_QCDSpectraComparison.ipynb`**: The $(\hat{m}_{h_1}, \hat{m}_{h_2})$ spectra of QCD multijet events obtained using different mass-prediction strategies
- **`AppendixFIGS9S10_BkgSplitROC.ipynb`**: ROC curves with full background components with color bands for background splitting
- **`AppendixFIGS11_MaxBinBkgSplitROC.ipynb`**: ROC curves with full background components with color bands for background splitting, using a simplified mass-prediction strategy that assigns each event to the bin with the maximum discretized $p(m_{h_1},\,m_{h_2})$

## Citation

If you use this code, please cite:

```
@article{Yang:2025txy,
    author = "Yang, Tianyi and Li, Congqiao",
    title = "{Potential of di-Higgs observation via a calibratable jet-free $HH\to 4b$ framework}",
    eprint = "2508.15048",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    month = "8",
    year = "2025"
}
```

