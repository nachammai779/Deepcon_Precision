# DEEPCON using Precision Features as Input

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Version](https://img.shields.io/badge/version-1.0-blue)

> Protein contact map prediction using the DeepCov architecture, trained with **precision matrix features** (441 channels) on 3456 proteins from the DeepCov dataset.

---

## 📋 Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Usage](#usage)
- [Evaluation](#evaluation)
- [Weights](#weights)

---

## 🔬 Overview

This project extends the DeepCov deep learning framework for protein contact prediction by replacing standard covariance features with **precision matrix features**. The model is trained and validated on 3456 proteins using 441 input channels, achieving improved contact prediction accuracy.

**Key details:**
- Dataset: DeepCov (3456 proteins)
- Input features: Precision matrix (441 channels)
- Task: Protein contact map prediction

---

## ⚙️ Requirements

```bash
pip install numpy tensorflow keras
```

---

## 🚀 Usage

### Predict
```bash
python ../deepcon-precision.py --aln ./16pkA0.aln --rr ./16pkA0.rr
```

### Covariance to Precision Conversion
```bash
python cov-pre-matrix-conversion-d0714_21c_batch1.py
```

---

## 📊 Evaluation

```bash
./coneva-lite.pl -pdb ./16pkA.pdb -rr ./16pkA0.rr
```

---

## 📦 Weights

The trained model weights (HDF5 format) are attached in **[Release v1.0](../../releases/tag/version1.0)**.
