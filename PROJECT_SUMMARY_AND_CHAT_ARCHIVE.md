# 📌 Pinned Project Archive: CardioLAC-SSM Research Initiative

> **Project Title:** *Lead-Agnostic and Uncertainty-Calibrated State-Space Modeling for Multi-Label Cardiac Abnormality Detection from Incomplete 12-Lead Electrocardiograms*  
> **Repository:** [https://github.com/Ashikur860/Cardiac-Abnormality-Detection-from-Incomplete-12-Lead-Electrocardiograms](https://github.com/Ashikur860/Cardiac-Abnormality-Detection-from-Incomplete-12-Lead-Electrocardiograms)  
> **Local Workspace:** `C:\Research\Lead-Agnostic and Uncertainty-Calibrated State-Space Modeling for Multi-Label Cardiac Abnormality Detection from Incomplete 12-Lead Electrocardiograms`  
> **Conversation ID:** `8a1ecc43-1807-4cae-b5a0-d5a59dbf7055`  
> **Date Pinned:** October 2026  

---

## 🧭 Quick Access & Key Links

| Resource | Description | Path / URL |
| :--- | :--- | :--- |
| **GitHub Repository** | Full synced code, models & LaTeX | [GitHub Repo](https://github.com/Ashikur860/Cardiac-Abnormality-Detection-from-Incomplete-12-Lead-Electrocardiograms) |
| **53-Page Monograph PDF** | Full IEEE TBME Extended Review PDF | [`results/IEEE_Transactions_CardioLAC_SSM_53Pages_LaTeX.pdf`](file:///C:/Research/Lead-Agnostic%20and%20Uncertainty-Calibrated%20State-Space%20Modeling%20for%20Multi-Label%20Cardiac%20Abnormality%20Detection%20from%20Incomplete%2012-Lead%20Electrocardiograms/results/IEEE_Transactions_CardioLAC_SSM_53Pages_LaTeX.pdf) |
| **13-Page Camera-Ready PDF** | Standard IEEE 2-Column Journal PDF | [`results/IEEE_Transactions_CardioLAC_SSM_CameraReady.pdf`](file:///C:/Research/Lead-Agnostic%20and%20Uncertainty-Calibrated%20State-Space%20Modeling%20for%20Multi-Label%20Cardiac%20Abnormality%20Detection%20from%20Incomplete%2012-Lead%20Electrocardiograms/results/IEEE_Transactions_CardioLAC_SSM_CameraReady.pdf) |
| **LaTeX Source Bundle** | Overleaf / arXiv ready ZIP archive | [`results/IEEE_LaTeX_Submission_Package.zip`](file:///C:/Research/Lead-Agnostic%20and%20Uncertainty-Calibrated%20State-Space%20Modeling%20for%20Multi-Label%20Cardiac%20Abnormality%20Detection%20from%20Incomplete%2012-Lead%20Electrocardiograms/results/IEEE_LaTeX_Submission_Package.zip) |
| **Interactive Web Workstation** | Diagnostic UI with 12-lead ECG strip | [`run_app.py`](file:///C:/Research/Lead-Agnostic%20and%20Uncertainty-Calibrated%20State-Space%20Modeling%20for%20Multi-Label%20Cardiac%20Abnormality%20Detection%20from%20Incomplete%2012-Lead%20Electrocardiograms/run_app.py) (`http://localhost:8000`) |
| **Markdown Manuscript** | Full Q1 Journal Paper Source | [`paper/Q1_Journal_Paper_CardioLAC_SSM.md`](file:///C:/Research/Lead-Agnostic%20and%20Uncertainty-Calibrated%20State-Space%20Modeling%20for%20Multi-Label%20Cardiac%20Abnormality%20Detection%20from%20Incomplete%2012-Lead%20Electrocardiograms/paper/Q1_Journal_Paper_CardioLAC_SSM.md) |

---

## 🎯 Executive Summary & Major Accomplishments

1. **Comprehensive 19-Model Benchmark Arena:**
   - **10 Classical ML Baselines:** Logistic Regression, Random Forest, Extra Trees, XGBoost, LightGBM, Calibrated Linear SVM, KNN ($k=5$), MLP, One-Class SVM, and Stacking Ensemble (**91.64% accuracy, 0.9425 Micro AUROC**).
   - **9 Deep Learning Baselines:** ResNet-1D (16 layers), CNN-LSTM Hybrid, 1D-Transformer, InceptionTime-1D, DenseNet-1D, BiLSTM+Attention, BiGRU+Attention, Temporal ConvNet (**TCN-1D winner: 92.86% accuracy, 0.6657 Macro F1**), and class-weighted VGG16-1D.
   - Individual multi-label confusion matrices and ROC curves generated and cataloged for all 19 models.

2. **Proposed CardioLAC-SSM Architectural Framework:**
   - **Shared 1D-CNN Feature Stem:** Preserves channel equivariance.
   - **Anatomical Lead Embeddings & Dynamic Masked Attention Pooling:** Accommodates any subset from 1 to 12 leads without zero-imputation collapse.
   - **Bidirectional Diagonal S4D Backbone:** Continuous-time state-space system with proven Hurwitz stability ($\text{Re}(\lambda_n) < 0$) and $\mathcal{O}(L \log L)$ parallel FFT training / $\mathcal{O}(1)$ recurrent streaming inference.
   - **Dual Uncertainty Heads:** Decouples epistemic uncertainty (via 10-pass MC Dropout) from aleatoric sensor noise (via heteroscedastic loss).

3. **Rigorous Incomplete Lead & Clinical Validation:**
   - Evaluated across 7 clinical scenarios (Complete 12L, 6L Limb, 6L Precordial, 3L Einthoven, 2L Holter, Single-Lead II, Random 50% Disconnection).
   - On single-lead II, CardioLAC-SSM retains an AUROC of 0.7240 (nearly 3x higher F1 than ResNet-1D).
   - Individual condition accuracies exceed 94% across acute abnormalities (PVC: 95.62%, STD: 94.86%, PAC: 94.48%, AF: 94.48%).
   - **Selective Clinical Triage:** Abstaining from the top 15% most uncertain predictions elevates autonomous diagnostic accuracy to **96.20%** (Micro AUROC: 0.9780).

4. **Publication-Grade IEEE Deliverables:**
   - Full Q1 IEEE Transactions journal paper written with 55 literature citations (2020–2026).
   - Built a complete LaTeX submission package (`main.tex`, `references.bib`, `IEEEtran.cls`, `IEEEtran.bst`).
   - Verified compilation via Tectonic into both a **53-Page Extended Monograph PDF** and a **13-Page Camera-Ready PDF**.

5. **Clinical Diagnostic Web Application:**
   - Real-time diagnostic server (`run_app.py`) on port 8000 with 11.4ms CPU latency (67,512 parameters).
   - Interactive Canvas ECG strip renderer ($25\text{ mm/s}$, $10\text{ mm/mV}$ clinical grid), dynamic 12-lead toggle, live epistemic/aleatoric dials, and multi-model benchmark visualizer galleries.

6. **Full GitHub Synchronization:**
   - Successfully committed (207 files, 26,548 insertions) and pushed to [GitHub](https://github.com/Ashikur860/Cardiac-Abnormality-Detection-from-Incomplete-12-Lead-Electrocardiograms).

---

## 📓 Complete Directory of Jupyter Notebooks

All 14 notebooks reside in `notebooks/` with their visual cell outputs pre-rendered for instant review:

| Notebook | Focus & Topic | Key Output Artifact |
| :--- | :--- | :--- |
| `01_data_exploration_and_cohort_curation.ipynb` | Chapman-Shaoxing cohort curation & label distribution | `fig1`, `fig2`, `fig3` |
| `02_signal_preprocessing_pipeline.ipynb` | Butterworth bandpass filtering (0.5–45Hz) & Z-score | `fig4_preprocessing` |
| `03_classical_ml_benchmarking.ipynb` | 10 Classical ML models & Stacking Ensemble | `fig6_ml_baselines` |
| `04_deep_learning_baselines.ipynb` | 8 Deep Learning architectures (ResNet, TCN, Transf.) | `fig7_dl_baselines` |
| `05_incomplete_lead_simulation_framework.ipynb` | Lead loss simulation across 7 clinical scenarios | `fig5_incomplete_lead` |
| `06_cardiolac_ssm_architecture_and_training.ipynb` | S4D continuous state-space implementation & training | `fig8_arch`, `fig9_loss` |
| `07_lead_agnostic_evaluation_suite.ipynb` | Systematic lead degradation evaluation & retention | `fig18_perf_vs_leads` |
| `08_multi_label_metrics_and_roc_analysis.ipynb` | Multi-label confusion matrices, ROC and PR curves | `fig10`, `fig11`, `fig12` |
| `09_uncertainty_quantification_and_calibration.ipynb` | MC Dropout, heteroscedastic head & selective triage | `fig13_calib`, `fig14_triage` |
| `10_explainable_ai_integrated_gradients.ipynb` | Integrated Gradients temporal saliency & lead weights | `fig15_xai`, `fig16_leads` |
| `11_ablation_studies.ipynb` | Component ablations (attention, S4D, uncertainty) | `fig17_ablation` |
| `12_statistical_significance_and_bootstrap.ipynb` | 1,000-iteration paired bootstrap hypothesis tests | `fig20_bootstrap` |
| `13_final_summary_and_paper_artifacts.ipynb` | Master benchmark heatmap, radar chart & report | `fig21`, `fig22`, `fig23` |
| `14_vgg16_training_and_one_class_svm.ipynb` | Class-weighted VGG16-1D & One-Class SVM boundary | `fig24_vgg16_curves` |

---

## 🛠️ How to Revisit and Run Anytime

```bash
# 1. Clone repository
git clone https://github.com/Ashikur860/Cardiac-Abnormality-Detection-from-Incomplete-12-Lead-Electrocardiograms.git
cd Cardiac-Abnormality-Detection-from-Incomplete-12-Lead-Electrocardiograms

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch interactive clinical workstation
python run_app.py
# Open browser: http://localhost:8000

# 4. Re-compile LaTeX papers
tectonic paper/latex/main.tex                  # 13-page camera-ready
tectonic paper/latex/main_review_format.tex    # 53-page extended monograph
```
