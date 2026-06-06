# 🩺 Diabetes Detection - Machine Learning Classification

Proyek machine learning untuk mendeteksi **risiko diabetes** pada pasien wanita berdasarkan data medis menggunakan berbagai algoritma klasifikasi dengan pendekatan penanganan data imbalanced dan hidden missing values.

---

## 📌 Overview

Diabetes adalah salah satu penyakit kronis yang paling banyak diderita di dunia. Deteksi dini sangat penting untuk mencegah komplikasi serius. Proyek ini membangun model prediktif untuk mengidentifikasi pasien yang berisiko terkena diabetes berdasarkan data medis, sehingga tenaga kesehatan dapat mengambil tindakan pencegahan lebih awal.

---

## 📊 Dataset

| Atribut | Detail |
|---|---|
| Sumber | Kaggle — Pima Indians Diabetes Database (UCI) |
| Jumlah Sampel | 768 pasien wanita |
| Jumlah Fitur | 8 fitur medis |
| Target | `Outcome` (1 = Diabetes, 0 = Tidak Diabetes) |
| Distribusi Target | 65% Tidak Diabetes, 35% Diabetes (Imbalanced) |

**Fitur Utama:**

| Fitur | Deskripsi |
|---|---|
| `Pregnancies` | Jumlah kehamilan |
| `Glucose` | Kadar glukosa plasma |
| `BloodPressure` | Tekanan darah diastolik (mm Hg) |
| `SkinThickness` | Ketebalan lipatan kulit trisep (mm) |
| `Insulin` | Kadar insulin serum 2 jam (mu U/ml) |
| `BMI` | Indeks massa tubuh |
| `DiabetesPedigreeFunction` | Fungsi riwayat diabetes keluarga |
| `Age` | Usia pasien |

---

## 🛠️ Tech Stack

| Kategori | Library |
|---|---|
| Data Manipulation | `pandas`, `numpy` |
| Visualisasi | `matplotlib`, `seaborn` |
| Preprocessing | `scikit-learn` (StandardScaler) |
| Imbalanced Data | `imbalanced-learn` (SMOTE) |
| Modeling | `scikit-learn` (DecisionTree, RandomForest, LogisticRegression, GradientBoosting) |
| Hyperparameter Tuning | `scikit-learn` (GridSearchCV) |
| Evaluasi | `scikit-learn` (classification_report, ROC-AUC) |
| Model Saving | `joblib` |

---

## 🔄 Alur Proyek

```
Raw Data (768 pasien)
   │
   ▼
Exploratory Data Analysis (EDA)
├── Distribusi target
├── Histogram semua fitur
├── Heatmap korelasi
└── Boxplot fitur vs Outcome
   │
   ▼
Preprocessing
├── Deteksi Hidden Missing Values (nilai 0 tidak masuk akal)
├── Imputasi dengan Median
├── Feature Scaling (StandardScaler)
└── Train-Test Split (80:20, stratify)
   │
   ▼
Modeling
├── Decision Tree
├── Logistic Regression
├── Random Forest
├── Gradient Boosting
├── Random Forest + SMOTE
└── Random Forest + SMOTE + Hyperparameter Tuning (GridSearchCV)
   │
   ▼
Evaluasi & Perbandingan Model
├── Classification Report
├── ROC Curve & AUC Score
├── Confusion Matrix
└── Feature Importance
   │
   ▼
Simpan Model Terbaik
```

---

## ⚠️ Tantangan Utama

Dataset ini memiliki **hidden missing values** — nilai 0 yang tidak masuk akal secara medis:

| Fitur | Jumlah Nilai 0 | Persentase |
|---|---|---|
| Glucose | 5 | 0.7% |
| BloodPressure | 35 | 4.6% |
| SkinThickness | 227 | 29.6% |
| Insulin | 374 | **48.7%** |
| BMI | 11 | 1.4% |

> 💡 Nilai 0 pada fitur medis seperti Glucose, BloodPressure, dan Insulin tidak mungkin terjadi secara biologis. Nilai-nilai ini diganti dengan **median** masing-masing kolom.

---

## 📈 Hasil & Evaluasi

### Perbandingan Semua Model

| Model | Accuracy | Recall Diabetes | F1-Score | AUC Score |
|---|---|---|---|---|
| Decision Tree | 66.88% | 44.44% | 66.09% | 0.617 |
| Logistic Regression | 70.78% | 50.00% | 70.08% | 0.815 |
| Random Forest | 75.97% | 59.26% | 75.55% | 0.821 |
| Gradient Boosting | 75.97% | 59.26% | 75.55% | 0.828 |
| RF + SMOTE | 74.68% | 72.22% | 75.05% | 0.824 |
| **RF + SMOTE + Tuning ✅** | **75.97%** | **72.22%** | **76.27%** | **0.8255** |

### Model Terbaik — RF + SMOTE + Tuning

| Metrik | Nilai |
|---|---|
| Accuracy | **75.97%** |
| Recall Diabetes | **72.22%** |
| F1-Score | **76.27%** |
| AUC Score | **0.8255** |

> 💡 **Mengapa RF + SMOTE + Tuning?** Model ini memiliki keseimbangan terbaik antara Accuracy (75.97%) dan Recall Diabetes (72.22%). Dalam konteks medis, Recall tinggi sangat penting — lebih baik salah prediksi "berisiko diabetes" daripada melewatkan pasien yang benar-benar diabetes.

---

## 🔍 Key Insights

1. **Glucose** adalah fitur paling berpengaruh dalam memprediksi diabetes
2. **BMI** dan **Age** adalah fitur penting berikutnya
3. **Insulin** memiliki 48.7% hidden missing values — fitur paling bermasalah di dataset
4. Tanpa menghapus outlier, AUC Score meningkat signifikan — outlier mengandung informasi medis penting
5. SMOTE berhasil meningkatkan Recall Diabetes dari 59% → 72%

---

## 💡 Rekomendasi Medis

Berdasarkan hasil analisis:

- 🩸 **Pantau kadar Glucose secara rutin** — fitur paling prediktif untuk diabetes
- ⚖️ **Jaga BMI dalam rentang normal** — BMI tinggi berkorelasi kuat dengan diabetes
- 👴 **Perhatikan faktor usia** — risiko meningkat seiring bertambahnya usia
- 👨‍👩‍👧 **Riwayat keluarga penting** — DiabetesPedigreeFunction menunjukkan pengaruh genetik

---

---

## 🚀 Cara Menjalankan

1. Clone repository ini:
```bash
git clone https://github.com/ListyaDwiSubagya/diabetes-detection.git
cd diabetes-detection
```

2. Buat virtual environment:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn joblib
```

4. Jalankan notebook:
```bash
jupyter notebook diabetes_detection.ipynb
```

---

## 🏆 Pencapaian

- ✅ Berhasil mendeteksi dan menangani hidden missing values (48.7% pada kolom Insulin)
- ✅ Recall Diabetes meningkat dari 44% → 72% dengan SMOTE + Tuning
- ✅ AUC Score 0.8255 — model sangat baik membedakan diabetes vs tidak diabetes
- ✅ Menemukan bahwa mempertahankan outlier meningkatkan performa model pada data medis

---

## 👤 Author

**Listya Dwi Subagya**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat&logo=linkedin)](https://linkedin.com/in/listyadwis)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?style=flat&logo=github)](https://github.com/ListyaDwiSubagya)

---

## 📝 License

Proyek ini dibuat untuk keperluan pembelajaran dan pengembangan portofolio data science.
