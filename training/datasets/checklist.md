## Dataset checklist

### Training samples

The training dataset contains three physics processes. The table below lists their selections, which correspond to the description in Supplementary Material Section C.1 of [arXiv:2508.15048](https://arxiv.org/abs/2508.15048).

| Path | Name | Number of events | Selection and explanation |
| --- | --- | --- | --- |
| HH4b_2HDM_H3VAR_H1H2_40to200_merged_ntuple | variable-mass $h_3 \to h_1 h_2 \to 4b$ samples | 70M | "4j": passes the 4j selection, including the baseline trigger-level kinematic requirements of HT>330 GeV and four leading-jet pT>[75, 60, 45, 40] GeV. (N.B. the corresponding description in the paper needs to be corrected: it states that the same 4j3b or 4j2b requirement as for QCD is used.) |
| QCD_DelphesHH4JTrig_merged_ntuple | QCD | 63M | "4j3b or 4j2b": in addition to the kinematic selection included in 4j above, requires either the 3b or 2b condition, based on different working points of the SophonAK4 flavor-tagging score. |
| TTbar_ntuple | $t\bar{t}$ | 12M | "3b or 2b": a relaxed version of "4j3b or 4j2b" in which the HT and four jet-pT requirements are removed, leaving only the 3b or 2b condition. |

### Inference samples

We share the main inference samples used in this study. The generator configurations for samples produced by us are provided in `gen_configs` in the repository root. Because some processes are expensive to generate, their generation was optimized specifically for the trigger requirements, and the resulting ntuples therefore contain only events passing a particular selection. Most other samples were generated inclusively without a selection. These samples correspond to the description in Supplementary Material Section B.1.

#### Signal processes

| Path | Name | Selection and explanation |
| --- | --- | --- |
| ggHH_kl_0_kt_1_ntuple | ggF $HH\to 4b$, $\kappa_\lambda = 0$ | "4j" |
| ggHH_kl_1_kt_1_ntuple | ggF $HH\to 4b$, $\kappa_\lambda = 1$ | "4j" |
| ggHH_kl_5_kt_1_ntuple | ggF $HH\to 4b$, $\kappa_\lambda = 5$ | "4j" |
| qqHH_CV_1_C2V_1_kl_1_ntuple | VBF $HH\to 4b$, $(C_V, C_{2V}, \kappa_\lambda)=(1, 1, 1)$ | "4j" |
| qqHH_CV_1_C2V_2_kl_1_ntuple | VBF $HH\to 4b$, $(C_V, C_{2V}, \kappa_\lambda)=(1, 2, 1)$ | "4j" |
| qqHH_CV_1_C2V_1_kl_2_ntuple | VBF $HH\to 4b$, $(C_V, C_{2V}, \kappa_\lambda)=(1, 1, 2)$ | "4j" |
| qqHH_CV_1_C2V_1_kl_0_ntuple | VBF $HH\to 4b$, $(C_V, C_{2V}, \kappa_\lambda)=(1, 1, 0)$ | "4j" |
| qqHH_CV_0p5_C2V_1_kl_1_ntuple | VBF $HH\to 4b$, $(C_V, C_{2V}, \kappa_\lambda)=(0.5, 1, 1)$ | "4j" |
| qqHH_CV_1p5_C2V_1_kl_1_ntuple | VBF $HH\to 4b$, $(C_V, C_{2V}, \kappa_\lambda)=(1.5, 1, 1)$ | "4j" |

#### Background processes

**QCQ and Z+jets**

| Path | Name | Selection and explanation |
| --- | --- | --- |
| QCD_DelphesHH4JTrig_forInfer_merged_ntuple, QCD_DelphesHH4JTrig_forInfer2_merged_ntuple | QCD | "4j3b or 4j2b" |
| ZJetsToQQ_DelphesHH4JTrig_merged_ntuple | $Z(qq)$+jets | "4j3b or 4j2b" |

**Top-quark backgrounds**

| Path | Name | Selection and explanation |
| --- | --- | --- |
| TTbar_forInfer_ntuple, TTbar_forInfer2_ntuple | $t\bar{t}$ | The `forInfer` directory passes "3b or 2b", as for the ttbar training sample above. `forInfer2` passes "4j3b or 4j2b". |
| SingleTop_ntuple | single-top (t/s-channel) | "3b or 2b" |
| TW_ntuple | $tW$ | "3b or 2b" |
| TTbarW_ntuple | $t\bar{t}W$ | "3b or 2b" |
| TTbarZ_ntuple | $t\bar{t}Z$ | "3b or 2b" |
| ttH_ntuple | $t\bar{t}H$ | "3b or 2b" |

**Diboson backgrounds**

| Path | Name | Selection and explanation |
| --- | --- | --- |
| WW_ntuple | $WW$ | "3b or 2b" |
| ZW_ntuple | $ZW$ | "3b or 2b" |
| ZZ_ntuple | $ZZ$ | "3b or 2b" |

**Single-Higgs backgrounds**

| Path | Name | Selection and explanation |
| --- | --- | --- |
| SingleHiggs_ntuple | ggF Higgs | "3b or 2b" |
| VBFH_ntuple | VBF Higgs | "3b or 2b" |
| WplusH_ntuple | $W^{+}H$ | "3b or 2b" |
| WminusH_ntuple | $W^{-}H$ | "3b or 2b" |
| ZH_ntuple | $ZH$ | "3b or 2b" |
