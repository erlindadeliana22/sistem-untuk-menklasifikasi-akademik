# Klasifikasi Performa Akademik Siswa Menggunakan Algoritma K-Means Clustering Berdasarkan Kehadiran

Proyek ini bertujuan untuk mengklasifikasikan performa akademik siswa menggunakan algoritma K-Means Clustering berdasarkan nilai akademik dan tingkat kehadiran siswa. Aplikasi ini dibangun menggunakan Python dengan antarmuka web berbasis Streamlit.

## Fitur Utama

- Mengklasifikasikan siswa ke dalam 3 kategori berdasarkan nilai akademik dan kehadiran:
  - Performa Tinggi
  - Performa Sedang
  - Performa Rendah
- Menampilkan hasil klasifikasi dalam bentuk tabel dan visualisasi grafik
- Menyediakan ringkasan statistik untuk setiap kategori
- Menggunakan model machine learning K-Means yang dilatih secara otomatis

## Prasyarat Sistem

Sebelum menjalankan proyek ini, pastikan komputer Anda telah terinstal:

- Python 3.8 atau lebih tinggi
- Pip (package installer untuk Python)

## Instalasi

1. Clone atau download repository ini ke komputer lokal Anda
2. Buka terminal/command prompt dan arahkan ke direktori proyek
3. Buat virtual environment (opsional tapi direkomendasikan):

   ```bash
   python -m venv venv
   ```

   Aktifkan virtual environment:

   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. Instal semua dependensi yang diperlukan:
   ```bash
   pip install -r requirements.txt
   ```

## Struktur File

- `main.py`: File utama aplikasi Streamlit
- `Data Siswa.csv`: File data siswa yang akan dianalisis
- `requirements.txt`: Daftar dependensi Python yang dibutuhkan
- `kmeans_model.pkl`: Model K-Means yang telah dilatih (akan dibuat otomatis saat pertama kali dijalankan)
- `scaler.pkl`: Scaler untuk preprocessing data (akan dibuat otomatis saat pertama kali dijalankan)

## Format Data

File `Data Siswa.csv` harus memiliki format sebagai berikut:

```
Nama Siswa;;;Nilai Akademik;;Kehadiran(%)
;;;;;
Aidan Muhammad Prasetya;;;83;;94
Aluna Zahra;;;90;;99
...
```

Kolom yang diperlukan:

1. Nama Siswa
2. Nilai Akademik (numerik)
3. Kehadiran(%) (numerik)

## Cara Menjalankan Aplikasi

Untuk menjalankan aplikasi, gunakan perintah berikut di terminal:

```bash
streamlit run main.py
```

Setelah dijalankan, browser akan terbuka secara otomatis menampilkan antarmuka aplikasi.

## Cara Menggunakan Aplikasi

1. Setelah aplikasi dijalankan, sistem akan secara otomatis membaca data dari file `Data Siswa.csv`
2. Aplikasi akan melakukan clustering menggunakan algoritma K-Means dan menampilkan hasilnya
3. Hasil klasifikasi akan ditampilkan dalam bentuk tabel dengan warna berbeda untuk setiap kategori
4. Visualisasi scatter plot akan menunjukkan distribusi siswa berdasarkan nilai dan kehadiran
5. Ringkasan statistik per kategori akan ditampilkan di bagian bawah halaman

## Penjelasan Kategori

Berdasarkan hasil clustering menggunakan K-Means pada data nilai akademik dan kehadiran, kategori diinterpretasikan sebagai berikut:

- **Tinggi** → **Performa Sangat Baik**
  Siswa memiliki nilai akademik tinggi dan kehadiran sangat baik. Mereka konsisten dan berpotensi menjadi teladan.

- **Sedang** → **Performa Cukup**
  Siswa memiliki kombinasi nilai dan kehadiran yang stabil, namun masih ada ruang untuk peningkatan.

- **Rendah** → **Performa Perlu Perhatian**
  Siswa memiliki nilai dan/atau kehadiran yang relatif rendah. Disarankan untuk pendampingan akademik atau konseling kehadiran.

## Teknologi yang Digunakan

- [Python](https://www.python.org/) - Bahasa pemrograman utama
- [Streamlit](https://streamlit.io/) - Framework untuk membuat aplikasi web
- [Pandas](https://pandas.pydata.org/) - Untuk manipulasi dan analisis data
- [NumPy](https://numpy.org/) - Library untuk komputasi numerik
- [Scikit-learn](https://scikit-learn.org/) - Library untuk machine learning
- [Matplotlib](https://matplotlib.org/) - Library untuk visualisasi data

## Lisensi

Proyek ini merupakan proyek open source yang dapat digunakan secara bebas untuk keperluan pendidikan.

## Kontak

Jika ada pertanyaan atau masalah terkait penggunaan aplikasi ini, silakan hubungi pengembang.
