# Detecting Childhood Pneumonia from Chest X-rays

Deep-learning models that classify pediatric chest X-rays as **normal** or **pneumonia**, built with transfer learning and compared across two backbones (DenseNet121 and ResNet18). This is a group research project for *Deep Learning for Medical Imaging*.

**Best model:** DenseNet121 with an exponential moving average (EMA) of its weights — **AUC 0.978**, **93.3% test accuracy** at the tuned threshold, catching almost every affected child.

---

## Overview

Pneumonia is one of the leading causes of death in children under five, and it is usually confirmed by reading a chest X-ray. This project trains convolutional neural networks to do a binary screen — normal vs. pneumonia — quickly and consistently, so the model could act as a second reader where trained specialists are scarce.

The work answers one question: *can transfer-learning models reliably tell normal and pneumonia chest X-rays apart in young children, and which model and training choices give the most accurate and trustworthy results?*

---

## Results

Best model: **DenseNet121 + EMA**, evaluated on the held-out test set of 624 images.

| Threshold | Accuracy | Sensitivity | Specificity | AUC |
|-----------|:--------:|:-----------:|:-----------:|:---:|
| Default (0.50) | 90.0% | 0.987 | 0.752 | 0.978 |
| Tuned (0.80)   | 93.3% | 0.985 | 0.846 | 0.978 |

Tuning the decision threshold cut false alarms (specificity 0.75 → 0.85) while keeping sensitivity near 0.99, which is what a screening tool needs.

A chi-square test on the test confusion table confirms the result is not down to chance (**χ² = 388.8, df = 1, p < 0.0001**).

### Model and configuration comparison

All figures are on the same test set at the default 0.50 threshold.

| Configuration | Accuracy | AUC | Note |
|---------------|:--------:|:---:|------|
| **DenseNet121 + EMA, 50 epochs** | **90.0%** | **0.978** | Best overall |
| DenseNet121 + EMA, 100 epochs | 90.2% | 0.977 | No gain from longer training |
| DenseNet121, no EMA | 87.2% | 0.966 | EMA adds ~3 points |
| DenseNet121 + noise augmentation | 82.5% | 0.938 | Noise hurt the most; removed |
| ResNet18 | 86.9% | 0.968 | Strong and lighter, below DenseNet |

---

## Dataset

[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) — 5,863 pediatric chest radiographs (ages 1–5) from Guangzhou, China, released under CC BY 4.0.

The dataset ships with only **16 validation images**, which is too few to guide model selection reliably. We discarded that folder, merged the original train and validation images, and re-split them ourselves with a **patient-aware split** so that no patient appears in more than one subset. The original test folder was left untouched for the final evaluation.

| Subset | Normal | Pneumonia | Total |
|--------|:------:|:---------:|:-----:|
| Train  | 1,140 | 3,289 | 4,429 |
| Validation | 281 | 522 | 803 |
| Test   | 234 | 390 | 624 |

> **Note on the split.** We also compared patient-aware vs. image-level splitting with ResNet18. The final test scores were nearly identical (86.9% vs. 86.5%, AUC 0.968 both), because the test folder is held out separately either way. Patient-aware splitting is still the more honest choice — it keeps the validation estimate from becoming optimistic — but on this dataset it did not change the final test number.

---

## Methods

- **Transfer learning.** ResNet18 and DenseNet121 initialised with ImageNet weights. Grayscale X-rays are copied into 3 channels and resized to 224×224; the input layer is left unchanged so the pretrained weights are reused in full.
- **Fine-tuning.** Early layers frozen, deeper blocks fine-tuned. A compact classifier head (dropout → 128-unit ReLU → dropout → 2 classes) replaces the original head.
- **Optimisation.** AdamW (weight decay 1e-3), discriminative learning rates, cosine-annealing schedule over 50 epochs, batch size 32, mixed-precision (AMP), class weighting for the imbalance, and an exponential moving average (EMA, decay 0.999) of the weights used at evaluation.
- **Augmentation.** Small brightness/contrast changes, small rotations and translations, and a random resized crop (0.85–1.0). **No horizontal flip** — mirroring a chest X-ray would put the heart on the wrong side and teach an unrealistic pattern. A noise-based augmentation (Gaussian + salt-and-pepper) was tested and **removed** because it hurt performance.
- **Evaluation.** Accuracy, sensitivity (recall), specificity, precision, F1, and AUC. The decision threshold is tuned on the validation set for high sensitivity, and a chi-square test checks statistical significance.
- **Interpretability.** Grad-CAM heat maps confirm the model attends to the lung fields rather than image borders or markers.

---

## Repository structure

```
PNEUMONIA_MODEL/
├── densenet121.ipynb        # Best model: DenseNet121 + EMA (training, evaluation, Grad-CAM)
├── resnet18model.ipynb      # ResNet18 with the patient-aware split
├── resnet18model2.ipynb     # ResNet18: patient-aware vs. image-level split comparison
└── README.md
```

Notebooks are also hosted on Kaggle, where they run directly against the attached dataset:

- DenseNet121 — https://www.kaggle.com/code/mutalegeorge/densenet121
- ResNet18 — https://www.kaggle.com/code/mutalegeorge/resnet18model

---

## Getting started

The notebooks were developed on Kaggle with a GPU and the dataset attached, which is the easiest way to reproduce the results.

### Run on Kaggle
1. Open one of the Kaggle notebooks linked above.
2. Confirm the *Chest X-Ray Images (Pneumonia)* dataset is attached.
3. Enable the GPU accelerator and run all cells.

### Run locally
```bash
git clone https://github.com/GEORGEMUTALE/PNEUMONIA_MODEL.git
cd PNEUMONIA_MODEL

# create an environment and install the main dependencies
pip install torch torchvision scikit-learn scipy matplotlib numpy pillow jupyter
```
Download the dataset from Kaggle, update the data path at the top of the notebook, and run it. A CUDA-capable GPU is recommended.

**Main dependencies:** PyTorch, torchvision, scikit-learn, SciPy, matplotlib, NumPy, Pillow.

---

## Team

| Member | Role |
|--------|------|
| **George Mutale** | Transfer learning and model implementation |
| **Rhyan Lubega** | Data augmentation and hyperparameter tuning |
| **Emmanuel Boaz Onyango** | Evaluation and metrics |
| **Oscar Kyamuwendo** | Grad-CAM and model interpretability |

---

## References

- Kermany, D. S., et al. (2018). *Identifying medical diagnoses and treatable diseases by image-based deep learning.* Cell, 172(5), 1122–1131.
- He, K., et al. (2016). *Deep residual learning for image recognition.* CVPR (pp. 770–778).
- Huang, G., et al. (2017). *Densely connected convolutional networks.* CVPR (pp. 4700–4708).
- Selvaraju, R. R., et al. (2017). *Grad-CAM: Visual explanations from deep networks via gradient-based localization.* ICCV (pp. 618–626).
- Loshchilov, I., & Hutter, F. (2019). *Decoupled weight decay regularization (AdamW).* ICLR.
- Loshchilov, I., & Hutter, F. (2017). *SGDR: Stochastic gradient descent with warm restarts.* ICLR.
- Deng, J., et al. (2009). *ImageNet: A large-scale hierarchical image database.* CVPR (pp. 248–255).
- World Health Organization. (2022). *Pneumonia in children* [Fact sheet].

---

## Acknowledgements

Dataset by Paul Mooney on Kaggle, derived from Kermany et al. (2018). Released under CC BY 4.0.

*This project is for research and education. It is not a medical device and must not be used for clinical diagnosis.*