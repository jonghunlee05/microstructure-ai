# UHCS Microstructure Classification (SEM)

**Goal**: Classify ultrahigh-carbon steel (UHCS) SEM micrographs into microconstituents (e.g., Spheroidite, Pearlite, Widmanstätten, Carbide network, mixed phases).

**Why it matters**: Microstructure → properties → performance. Automating recognition speeds up digital metallurgy (triage, QA, dataset curation).

## TL;DR

- **Data**: UHCS micrographs + labels from NIST UHCSDB (SQLite + CSV).
- **Features**: Classical GLCM + LBP (100-D), ResNet18 embeddings (512-D), and Hybrid (612-D → PCA to 48).
- **Classifier**: RBF-SVM with class balancing.
- **Validation**: Leak-proof, group-aware CV (group = sample_key) + nested CV for tuning (no leakage).
- **Deploy**: Saved pipeline in artifacts/ + Streamlit demo (app.py) for drag-and-drop predictions.

## Dataset

**Source**: UHCSDB (NIST) — images + metadata in `data/UHCSDB/microstructures.sqlite` and `data/UHCSDB/labels.csv`.

**Grouping key**: `sample_key` (prevents mixing crops/magnifications of the same sample across train/test).

Use the provided data for research/education and respect original licensing/attribution.

## Methods

### Classical Texture (GLCM + LBP)

- **GLCM**: distances = [1, 2, 4], angles = [0°, 45°, 90°, 135°], props = {contrast, dissimilarity, homogeneity, ASM, energy, correlation} → 72 dims
- **LBP**: uniform LBP at (P=8,R=1) and (P=16,R=2) → 10 + 18 = 28 dims
- **Total**: 100-D classical feature vector per image.

### Deep Embeddings

- **Backbone**: ResNet18 (ImageNet), penultimate layer → 512-D embedding.
- Grayscale SEMs converted to RGB; uint16 safely mapped to uint8; ImageNet normalization applied.

### Hybrid

- Concatenate 100 (classical) + 512 (deep) → 612-D, then PCA → 48 (chosen by nested CV).
- **Classifier**: SVM (RBF) with `class_weight='balanced'`.

## Evaluation Protocol (Rigor)

- **Leak-proof CV**: StratifiedGroupKFold with group = sample_key → zero overlap of samples (and their crops/magnifications) between train/test.
- **Nested CV**: inner, group-aware CV tunes PCA n_components, C, gamma; outer CV reports macro-F1 (mean ± std).

## Results (nested, leak-proof CV)

| Method | Macro-F1 (mean ± std) |
|--------|----------------------|
| Hybrid (GLCM+LBP+Deep) | 0.555 ± 0.035 |
| Deep (ResNet18) | ~0.536 ± see CSV |
| Classical (GLCM+LBP) | ~0.447 ± see CSV |

Exact numbers are saved in `results/nested_cv_summary.csv`.

## Final pipeline (saved)

- **PCA**: 48 comps
- **SVM**: C = 2.0, γ = scale
- **Features**: 100 (classical) + 512 (deep) → 48 (after PCA)
- **Samples**: n = 795

## Artifacts

```
artifacts/
  ├─ hybrid_svm_pipeline.joblib
  └─ hybrid_svm_meta.json
results/
  ├─ sgkf_cv_predictions_classical.csv
  ├─ sgkf_cv_predictions_deep.csv
  ├─ sgkf_cv_predictions_hybrid.csv
  ├─ nested_cv_summary.csv
  └─ nested_cv_best_params_per_fold.csv
```

## Quickstart

### Environment
```bash
pip install -r requirements.txt
```

### Reproduce (Notebook)

Open `microstructure_ai.ipynb` and run in order:

1. Data loading & label cleaning
2. Classical CV (GLCM+LBP)
3. Deep CV (ResNet18)
4. Hybrid CV
5. Ablation summary + Nested CV
6. Finalize model (writes artifacts/)

### Demo (Streamlit)
```bash
streamlit run app.py
```

Upload one or more SEM images to get class + confidence.

## Interpretability (optional)

Grad-CAM overlays for a few correct/incorrect examples
→ verify attention on lamellae, spheroidite colonies, colony plates—not on scale bars/annotations.

Saved to: `results/gradcam/TP/` and `results/gradcam/Errors/` (if you ran the explainability cell).

## Limitations & Next Steps

- **Mixed classes & imbalance**: lower recall on mixed phases; more data/augmentation helps.
- **Domain shift**: ImageNet features aren't SEM-specific; texture-pretrained or lightly fine-tuned backbones could help.
- **Scale sensitivity**: ensure µm/px handling if magnification varies widely; consider multi-scale crops.
- **(Optional) Calibration**: add Platt/temperature scaling + reliability curves for thresholded use.

## How to present this (resume/portfolio)

> Automated UHCS SEM microstructure classification with a hybrid GLCM/LBP + ResNet18 + SVM pipeline. Implemented leak-proof, group-aware validation and nested CV (PCA + hyperparameter tuning). Delivered a saved model and Streamlit demo; used Grad-CAM to show metallurgically meaningful attention.

## Acknowledgements

- **Data**: UHCSDB (NIST) — Ultrahigh-Carbon Steel microstructure dataset.
- **Libraries**: scikit-image, scikit-learn, PyTorch, torchvision, Streamlit.
