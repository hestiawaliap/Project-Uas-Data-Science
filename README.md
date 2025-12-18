
# 📘 Judul Proyek
"ANALISIS PERBANDINGAN MODEL MACHINE LEARNING DAN DEEP LEARNING UNTUK PREDIKSI PENYAKIT PARKINSON MENGGUNAKAN DATASET UCI PARKINSONS"

## 👤 Informasi
- **Nama:** Hesti Awalia Putri
- **Repo:** (https://github.com/hestiawaliap/Project-Uas-Data-Science)  
- **Video:** (https://drive.google.com/file/d/17-SVHuQBFZjvwyyr8V3amNtCKk4kgovT/view?usp=sharing)  

---

# 1. 🎯 Ringkasan Proyek
- Menyelesaikan permasalahan sesuai domain  
- Melakukan data preparation  
- Membangun 3 model: **Baseline**, **Advanced**, **Deep Learning**  
- Melakukan evaluasi dan menentukan model terbaik  

---

# 2. 📄 Problem & Goals
**Problem Statements:**  
1.	Dataset UCI Parkinsons memiliki distribusi kelas yang tidak seimbang, sehingga diperlukan pendekatan pemodelan yang mampu menghasilkan prediksi penyakit Parkinson secara akurat dan stabil.
2.	Hubungan antara fitur-fitur suara vokal terhadap status Parkinson bersifat kompleks dan non-linear, sehingga model perlu mampu menangkap pola yang terdapat pada data.
3.	Diperlukan proses data preparation dan preprocessing yang tepat agar data dapat digunakan secara optimal oleh berbagai jenis model, termasuk model baseline, machine learning, dan deep learning.
4.	Diperlukan evaluasi dan perbandingan performa antara model baseline, model machine learning, dan model deep learning untuk menentukan pendekatan yang paling efektif dalam memprediksi penyakit Parkinson.


**Goals:**  
1.	Membangun model prediksi penyakit Parkinson menggunakan UCI Parkinsons Dataset dengan performa yang terukur berdasarkan metrik evaluasi klasifikasi.
2.	Mengembangkan dan membandingkan tiga pendekatan pemodelan, yaitu model baseline, model machine learning, dan model deep learning, untuk mengevaluasi efektivitas masing-masing pendekatan dalam memprediksi status penyakit Parkinson.
3.	Mengukur dan menganalisis performa setiap model menggunakan metrik evaluasi yang relevan, seperti accuracy, precision, recall, dan F1-score, guna menentukan model dengan kinerja terbaik.
4.	Menghasilkan pipeline analisis dan pemodelan yang dapat dijalankan secara reproducible melalui dokumentasi kode dan pengelolaan proyek yang terstruktur.


---
## 📁 Struktur Folder
```
project/
│
├── data/                               # Dataset (tidak di-commit, download manual)
│
├── images/                             # Visualizations
│   ├── boxplot_fitur_penting.png
│   ├── distribusi_kelas.png
│   ├── feature_importance.png
│   ├── heatmap_korelasi.png
│   ├── perbandingan_accuracy.png
│   ├── roc_curve.png
│   ├── training_history.png
│
├── models/                             # Saved models
│   ├── dl_model.h5
│   ├── rf_model.pkl
│   └── scaler.pkl
│   └── svm_model.pkl
│
├── notebooks/                          # Jupyter notebooks
│   └── Hesti_Awalia_Putri_233307016_UAS_SC.ipynb
│
├── src/                                # Source code
│   ├── __init__.py
│   ├── config.py
│   ├── evaluate.py
│   ├── main.py                    # disediakan sebagai modul evaluasi opsional dan tidak digunakan langsung dalam notebook eksperimen.
│   ├── predict.py
│   ├── preprocess.py
│   ├── train.py
│   ├── utils.py
│   
├── .gitignore
├── Checklist Submit.md                  # Checklist
├── Laporan Proyek Machine Learning.md   # Laporan
├── README.md                     
└── requirements.txt                     # Dependencies
```
---

# 3. 📊 Dataset
- **Sumber:** (https://archive.ics.uci.edu/dataset/545/rice+cammeo+and+osmancik) 
- **Jumlah Data:** 3810 
- **Tipe:** Tabular  

### Fitur Utama
| Nama Fitur        | Deskripsi                                                    |
| ----------------- | ------------------------------------------------------------ |
| Area              | Luas area butir beras hasil segmentasi citra                 |
| Perimeter         | Panjang keliling (boundary) butir beras                      |
| Major_Axis_Length | Panjang sumbu utama elips yang memodelkan bentuk butir beras |
| Minor_Axis_Length | Panjang sumbu minor elips yang memodelkan bentuk butir beras |
| Eccentricity      | Tingkat kelonjongan bentuk elips (0–1)                       |
| Convex_Area       | Luas area convex hull dari butir beras                       |
| Extent            | Rasio antara area objek dengan bounding box                  |
| Class             | Label varietas beras (target klasifikasi)                    |

---

# 4. 🔧 Data Preparation
Transformasi:
- Encoding
- Scaling

---

# 5. 🤖 Modeling
- **Model 1 – Baseline:** Logistic Regression  
- **Model 2 – Advanced ML:** Random Forest  
- **Model 3 – Deep Learning:** Multilayer Perceptron  

---

# 6. 🧪 Evaluation
**Metrik:**
- Accuracy
- F1-Score 

### Hasil Singkat
| Model | Score (Accuracy) | Catatan |
|-------|------------------|---------|
| Logistic Regression | 0.916 | Cocok sebagai pembanding awal, cepat dan efisien |
| Random Forest | 0.919 | Memberikan performa terbaik secara keseluruhan |
| MLP | 0.915 | Tidak memberikan peningkatan signifikan dibanding model klasik |


---

# 7. 🏁 Kesimpulan
- Model terbaik: Random Forest 
- Alasan: 
    - Memberikan performa terbaik  
    - Menghasilkan jumlah kesalahan paling rendah
    - Memberikan keseimbangan yang baik antara performa dan kompleksitas.
- Insight penting: 
    - Model machine learning tradisional seperti Random Forest dapat mengungguli deep learning pada data tabular dengan ukuran kecil hingga menengah.
    - Deep learning (MLP) tidak selalu memberikan peningkatan performa yang signifikan, terutama jika kompleksitas data tidak terlalu tinggi.

---

# 8. 🔮 Future Work
✅ Feature engineering lebih lanjut

✅ Hyperparameter tuning lebih ekstensif

✅ Ensemble methods (combining models)

✅ Membuat API (Flask/FastAPI)

✅ Membuat web application (Streamlit/Gradio)

✅ Improving inference speed

✅ Reducing model size

---

# 9. 🔁 Reproducibility

**Python Version:** 3.12.5

**Main Libraries & Versions:**
numpy==2.3.5  
pandas==2.3.3  
scikit-learn==1.8.0  
matplotlib==3.10.8  
seaborn==0.13.2  
joblib==1.5.2  

**Deep Learning Framework**
tensorflow_cpu==2.20.0 

**Additional Libraries:**

ucimlrepo
