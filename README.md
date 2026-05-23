# Proyek-Clustering-dan-Klasifikasi Mental Health Clustering Analysis
## Domain Proyek
Gangguan kesehatan mental (*mental health disorders*) merupakan isu global yang kian mendesak dan memengaruhi jutaan orang di seluruh dunia. Berdasarkan data dari World Health Organization (WHO), prevalensi gangguan kesehatan mental seperti depresi, kecemasan, skizofrenia, dan gangguan bipolar terus meningkat signifikan selama beberapa dekade terakhir. Masalah ini tidak hanya berdampak pada kesejahteraan emosional dan sosial individu, melainkan juga memicu beban ekonomi yang besar bagi negara akibat penurunan produktivitas.

Secara konvensional, penanganan kesehatan mental bersifat reaktif dan sering kali terlambat karena minimnya segmentasi data wilayah atau populasi yang berisiko tinggi. Melalui pendekatan data science dan *Machine Learning*, kita dapat mengidentifikasi pola penyebaran atau karakteristik prevalensi gangguan mental lintas negara dan tahun. Analisis berbasis pengelompokan (*Clustering*) dan pengenalan pola otomatis (*Classification*) dapat membantu para pengambil kebijakan kesehatan masyarakat (*public health*) untuk melakukan alokasi sumber daya medis secara lebih tepat sasaran, memahami klaster populasi yang rentan, serta memprediksi kategori risiko wilayah secara instan.

---

## Business Understanding

### Problem Statements
1. Bagaimana mengelompokkan wilayah/negara berdasarkan prevalensi berbagai gangguan mental tanpa adanya label klasifikasi awal?
2. Berapa jumlah klaster optimal untuk memisahkan tingkat keparahan gangguan mental secara objektif?
3. Bagaimana membangun model prediktif yang akurat untuk mengklasifikasikan wilayah ke dalam klaster risiko kesehatan mental tersebut demi mendukung keputusan medis yang cepat?
4. Apakah optimasi melalui hyperparameter tuning dapat mendongkrak performa model klasifikasi secara signifikan?

### Goals
1. Mengelompokkan data prevalensi gangguan kesehatan mental global menggunakan algoritma *Unsupervised Learning* (K-Means Clustering).
2. Menentukan jumlah kelompok (*K*) optimal berdasarkan metrik evaluasi objektif (*Elbow Method* dan *Silhouette Score*).
3. Membangun model *Supervised Learning* (Decision Tree dan Random Forest) untuk memprediksi kategori klaster risiko wilayah secara presisi.
4. Mengevaluasi dampak dari penerapan *Hyperparameter Tuning* berbasis *Grid Search* terhadap efisiensi dan akurasi model klasifikasi.

### Solution Statements
1. **Pendekatan Analisis Data & Clustering**: Menggunakan algoritma **K-Means Clustering** didahului oleh prapemrosesan data (pembersihan *outliers*, standardisasi, dan reduksi dimensi dengan PCA) untuk mengekstrak struktur kelompok yang laten.
2. **Pendekatan Klasifikasi Prediktif**: Menerapkan model **Decision Tree Classifier** dan **Random Forest Classifier** pada data berlabel hasil *clustering* untuk mengotomatisasi pengenalan klaster risiko.
3. **Optimasi & Eksperimen**: Melakukan eksperimen tuning parameter utama menggunakan **GridSearchCV** serta memetakan *Learning Curve* untuk mendeteksi indikasi *overfitting* atau *underfitting* pada model.

---

## Data Understanding
Dataset yang digunakan dalam proyek ini berwujud data runtun waktu global mengenai metrik gangguan mental, yang bersumber dari platform Kaggle: [Mental Health Dataset](https://www.kaggle.com/datasets/imtkaggleteam/mental-health/code).

### Karakteristik Dataset:
* **Jumlah Baris**: 6.420 baris data observasi.
* **Kondisi Label**: Data awal tidak memiliki label target (*unlabeled*).
* **Fitur Utama (Variabel)**:
  * `Negara` / `Country`: Entitas nama wilayah/negara (Kategorikal).
  * `Tahun` / `Year`: Tahun pencatatan data (Numerik).
  * `Schizophrenia disorders`: Persentase prevalensi gangguan skizofrenia (Numerik).
  * `Depressive disorders`: Persentase prevalensi gangguan depresi (Numerik).
  * `Anxiety disorders`: Persentase prevalensi gangguan kecemasan (Numerik).
  * `Bipolar disorders`: Persentase prevalensi gangguan bipolar (Numerik).
  * `Eating disorders`: Persentase prevalensi gangguan makan (Numerik).

---

## Data Preprocessing
Beberapa tahapan prapemrosesan yang diterapkan guna menjamin integritas data sebelum memasuki fase permodelan meliputi:
1. **Exploratory Data Analysis (EDA)**: Visualisasi distribusi fitur, pengecekan korelasi antar-variabel, dan deteksi dini data pencilan (*outliers*).
2. **Pembersihan Outliers**: Menghapus atau menyesuaikan data ekstrim yang dapat mendistorsi penempatan titik pusat klaster (*centroids*).
3. **Standardisasi/Normalisasi**: Mengubah skala fitur numerik menggunakan `StandardScaler` atau `MinMaxScaler` sehingga seluruh variabel memiliki kontribusi bobot yang seimbang.
4. **Reduksi Dimensi (PCA)**: Menerapkan *Principal Component Analysis* untuk mereduksi kompleksitas kolom sekaligus meningkatkan visualisasi dan keterpisahan kelompok data (*Silhouette Score* meningkat dari **0.4311** menjadi **0.5619** setelah aplikasi PCA).

---

## Modeling & Hasil Analisis

### 1. Pembangunan & Evaluasi Model Clustering (K-Means)
Penentuan jumlah klaster terbaik diuji lewat rentang nilai *K = 2* hingga *K = 9*:
* $K = 2$, Silhouette Score = 0.3543
* **$K = 3$, Silhouette Score = 0.4311 (Dipilih sebagai jumlah kelompok optimal)**
* $K = 4$, Silhouette Score = 0.3895

#### Interpretasi Karakteristik Hasil Cluster (Setelah Inverse Transform):
* **Cluster 0 (Tingkat Gangguan Tinggi)**: Memiliki rata-rata tingkat gangguan mental tertinggi secara umum, khususnya pada aspek *Depressive disorders* ($\mu \approx 3.65$), *Anxiety disorders* ($\mu \approx 4.61$), dan *Bipolar disorders* ($\mu \approx 0.87$).
* **Cluster 1 (Tingkat Gangguan Ringan)**: Menunjukkan rata-rata tingkat gangguan mental yang jauh lebih rendah dan stabil dibandingkan Cluster 0 (*Depressive*: $\mu \approx 3.08$, *Anxiety*: $\mu \approx 3.48$).
* **Cluster 2 (Tingkat Depresi Dominan Tinggi)**: Unik karena menunjukkan tingkat *Depressive disorders* yang paling ekstrem tinggi ($\mu \approx 4.67$), namun untuk jenis gangguan mental lainnya seperti kecemasan cenderung relatif lebih rendah daripada Cluster 0.

---

### 2. Pembangunan & Evaluasi Model Klasifikasi
Setelah pelabelan klaster disematkan ke dalam dataset (`hasil_clustering.csv`), data dipecah (*Data Splitting*) menjadi subset pelatihan (*training*) dan subset pengujian (*testing*) untuk melatih dua arsitektur *supervised learning*: **Decision Tree** dan **Random Forest**.

#### Perbandingan Performa Sebelum & Sesudah Tuning:

| Model | Kondisi | Accuracy | F1-Score | Recall |
| :--- | :--- | :---: | :---: | :---: |
| **Random Forest** | Sebelum Tuning | 0.998274 | 0.998274 | 0.998274 |
| | Setelah Tuning (Grid Search) | **0.998274** | **0.998273** | 0.998274 |
| **Decision Tree** | Sebelum Tuning | 0.997412 | 0.997412 | 0.997412 |
| | Setelah Tuning (Grid Search) | **0.997412** | **0.997412** | 0.997412 |

*Catatan Parameter Optimal (Grid Search):*
* **Random Forest**: `{'criterion': 'gini', 'max_depth': 10, 'min_samples_split': 2, 'n_estimators': 300}`
* **Decision Tree**: `{'criterion': 'gini', 'max_depth': 10, 'min_samples_split': 5}`

#### Analisis Hasil Evaluasi Klasifikasi:
1. **Dampak Hyperparameter Tuning**: Proses *tuning* via `GridSearchCV` tidak memberikan eskalasi performa yang signifikan baik pada Random Forest maupun Decision Tree. Performa model tetap bertahan stabil di kisaran akurasi **99.8%** dan **99.7%**. Hal ini mengindikasikan bahwa konfigurasi parameter bawaan (*default*) bawaan dari pustaka Scikit-Learn sudah sangat optimal dalam memetakan batas keputusan (*decision boundary*) dari data hasil klasterisasi ini.
2. **Analisis Learning Curve**:
   * Pada visualisasi kurva pembelajaran, grafik awal sempat menunjukkan riak indikasi *overfitting* di mana model bekerja sempurna pada data latih namun sedikit renggang pada data uji berskala kecil. 
   * Sebaliknya, pada variasi parameter tertentu ditemukan gejala *underfitting* akibat pembatasan kedalaman pohon (*max_depth*) yang terlalu ketat sehingga model kurang optimal menyerap pola esensial data. Keseimbangan terbaik dicapai pada tingkat kompleksitas sedang (`max_depth: 10`).

---

## Kesimpulan
Proyek ini sukses menerapkan pipeline analitik komprehensif dari hulu ke hilir:
* **Segmentasi Terbentuk**: Data kesehatan mental global berhasil terbagi secara objektif ke dalam 3 tingkatan risiko (Tinggi, Ringan, dan Depresi Dominan).
* **Prediksi Akurat**: Model klasifikasi mampu mengenali dan memprediksi kategori klaster wilayah baru dengan tingkat akurasi yang sangat impresif melampaui **99.7%**. Model ini sangat andal untuk dijadikan sistem otomasi diagnosis risiko wilayah bagi lembaga kesehatan internasional.
Project submission untuk pembelajaran *Machine Learning Clustering*.
