[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/R05VM8Rg)

# 🧠 IIT Madras – DA2401 Machine Learning Lab: End Semester Project  
## Non-Neural Specialist Architecture for MNIST Classification  

---

### 📌 **Purpose**

This repository contains a complete end-to-end implementation of a **high-accuracy, non-neural network classifier** for the MNIST dataset.  
It demonstrates how **carefully engineered classical ML pipelines** (PCA, HOG, Zonal, Directional features) combined with **stacked ensembles and digit specialists** can approach neural-level accuracy — all without using deep learning.

The project is organized into **three progressive phases**:

| Phase | Description | Core Focus |
|:------|:-------------|:------------|
| **Phase 1** | Scikit-learn baseline | Architecture design & experimentation |
| **Phase 2** | Pure Python implementations | Self-coded models & logic replication |
| **Phase 3** | Runtime optimization | < 5 min training, high F1 without parallelization |

---

## 📁 **Repository Structure**

```
📦 IITM-DA2401-MNIST-NonNeural-Architecture
│
├── data/                         # Raw MNIST CSVs + cached features
│   ├── MNIST_train.csv
│   ├── MNIST_validation.csv
│   └── precomputed_features/
│
├── src/
│   ├── algorithms.py              # Custom implementations of LR, KNN, RF, Boost, Calibrator
│   ├── features.py                # PCA, HOG, Directional, Zonal feature extraction
│   ├── main_phase1_sklearn.py     # Phase 1 baseline using scikit-learn
│   ├── main_phase2_pure.py        # Phase 2 pure Python self-coded learners
│   ├── main_phase3_opt.py         # Phase 3 optimized <5 min version
│   ├── main_hybrid_eval.py        # Hybrid test proving architecture-limited performance
│
├── reports/
│   ├── MNIST_Final_Report.tex     # Complete formatted report
│   ├── MNIST_Final_Report.pdf
│   ├── performance_plots.png
│   └── runtime_accuracy_chart.py
│
├── notebooks/
│   ├── Feature_Inspection.ipynb
│   └── Misclassification_Analysis.ipynb
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## ⚙️ **Installation & Dependencies**

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/IITM-DA2401-MNIST-NonNeural-Architecture.git
   cd IITM-DA2401-MNIST-NonNeural-Architecture
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   **Required packages:**
   - numpy  
   - scipy  
   - matplotlib  
   - scikit-learn  
   - xgboost  

3. (Optional) Generate feature caches for faster training:
   ```bash
   python src/features.py --precompute
   ```

---

## ▶️ **Running the Code**

All experiments are reproducible from the command line.

### A. Phase 1 – Scikit-learn Baseline
```bash
python src/main_phase1_sklearn.py
```
- Expected Accuracy: ~97.4%
- Runtime: ~200s

### B. Phase 2 – Pure Python Implementation
```bash
python src/main_phase2_pure.py
```
- Accuracy: ~94.8%
- Runtime: ~720s
- No scikit-learn dependencies.

### C. Phase 3 – Optimized Runtime Version
```bash
python src/main_phase3_opt.py
```
- Accuracy: ~94.6%
- Weighted F1: ~0.945
- Runtime: ~277s (under 5 minutes)

### D. Hybrid Compute Verification
```bash
python src/main_hybrid_eval.py
```
- Confirms architecture is compute-limited.
- Accuracy: 97.4%, Runtime: 200s

---

## 🧪 **Results Summary**

| Phase | Implementation | Accuracy | Weighted F1 | Runtime (s) |
|-------|----------------|-----------|--------------|--------------|
| 1 | Scikit-learn | 97.4% | 0.974 | 200 |
| 2 | Self-coded (Pure Python) | 94.8% | 0.948 | 720 |
| 3 | Optimized Python | 94.6% | 0.945 | 277 |
| Hybrid | Cached features + sklearn models | **97.4%** | **0.974** | **200** |

> “Every extra percent of accuracy beyond 94.5% was compute-bound, not architecture-bound.”

---

## 🧾 **Authors**

**Khaja Mohammed**  
Department of Data Science & Artificial Intelligence  
IIT Madras (2025–26)

---

## ✅ **Best Practices**
- Code is modular and reproducible.  
- Preprocessing steps are isolated in `features.py`.  
- No pre-trained models or cached learned parameters are used.  
- Only precomputed feature transforms are cached for efficiency.  
- Training adheres to a strict 5-minute runtime limit.

---

## 🧩 **Acknowledgements**
This work was completed as part of the DA2401 Machine Learning Laboratory under the Department of Data Science and Artificial Intelligence, IIT Madras.

---
