# Proyek-Clustering-dan-Klasifikasi
# Mental Health Clustering Analysis

## Deskripsi Proyek

Proyek ini merupakan implementasi *unsupervised learning* menggunakan metode clustering untuk menganalisis data kesehatan mental (*Mental Health Dataset*). Tujuan utama dari proyek ini adalah mengelompokkan data berdasarkan karakteristik tertentu sehingga dapat ditemukan pola atau segmentasi yang tersembunyi di dalam dataset.

Proyek ini dibuat menggunakan Python dan dijalankan melalui Jupyter Notebook.

---

## Dataset

Dataset yang digunakan berasal dari Kaggle:

* Dataset: *Mental Health Dataset*
* Sumber: urlKaggle - Mental Health Dataset[https://www.kaggle.com/datasets/imtkaggleteam/mental-health](https://www.kaggle.com/datasets/imtkaggleteam/mental-health)

### Karakteristik Dataset

* Tidak memiliki label (*unlabeled dataset*)
* Memiliki data numerik dan kategorikal
* Jumlah data: 6420 baris

---

## Tahapan Analisis

Proyek ini mencakup beberapa tahapan utama:

1. Import library
2. Memuat dataset
3. Exploratory Data Analysis (EDA)
4. Data preprocessing

   * Handling missing values
   * Encoding data kategorikal
   * Normalisasi data
   * Penanganan outlier
5. Pembangunan model clustering
6. Evaluasi clustering menggunakan Silhouette Score
7. Visualisasi hasil clustering menggunakan PCA
8. Interpretasi cluster

---

## Algoritma yang Digunakan

Beberapa metode dan library yang digunakan pada proyek ini:

* K-Means Clustering
* PCA (*Principal Component Analysis*)
* Silhouette Score Evaluation
* Elbow Method

### Library Utama

```python
pandas
numpy
matplotlib
seaborn
scikit-learn
yellowbrick
```

---

## Struktur File

```bash
├── [Clustering]_Submission_Akhir_BMLP_Meicha_Salsabila_Budiyanti_(Updated).ipynb
├── hasil_clustering.csv
└── README.md
```

### Penjelasan File

* `ipynb` : Notebook utama yang berisi seluruh proses analisis dan clustering.
* `hasil_clustering.csv` : Dataset hasil clustering yang sudah ditambahkan label cluster.
* `README.md` : Dokumentasi proyek.

---

## Preview Visualisasi

### Elbow Method

Tambahkan hasil visualisasi Elbow Method pada folder `images/` lalu tampilkan dengan syntax berikut:

```markdown
![Elbow Method](images/elbow_method.png)
```

### Visualisasi Cluster PCA

```markdown
![PCA Clustering](images/pca_clustering.png)
```

### Distribusi Cluster

```markdown
![Cluster Distribution](images/cluster_distribution.png)
```

> Simpan seluruh gambar pada folder `images` agar README lebih rapi dan mudah dibaca.

Contoh struktur repository:

```bash
├── images
│   ├── elbow_method.png
│   ├── pca_clustering.png
│   └── cluster_distribution.png
├── hasil_clustering.csv
├── notebook.ipynb
└── README.md
```

---

## Hasil Clustering

Model clustering berhasil membagi data menjadi **3 cluster utama** berdasarkan kemiripan karakteristik gangguan kesehatan mental pada setiap negara dan tahun.

### Distribusi Cluster

| Cluster | Jumlah Data |
| ------- | ----------- |
| 0       | 1972        |
| 1       | 2240        |
| 2       | 1580        |

### Variabel yang Digunakan

* Schizophrenia disorders
* Depressive disorders
* Anxiety disorders
* Bipolar disorders
* Eating disorders

### Evaluasi Model

Evaluasi kualitas clustering dilakukan menggunakan:

* Elbow Method
* Silhouette Score

### Output Project

Hasil akhir clustering disimpan pada file:

```bash
hasil_clustering.csv
```

File tersebut berisi data yang telah ditambahkan label cluster pada kolom:

```python
Cluster
```

Visualisasi clustering dilakukan menggunakan PCA (*Principal Component Analysis*) untuk mereduksi dimensi data menjadi 2 dimensi sehingga pola antar cluster lebih mudah dianalisis.

---

Project submission untuk pembelajaran *Machine Learning Clustering*.
