# Laporan Proyek Machine Learning - Fahru Rahman

## Project Overview

Dalam era digital saat ini, pengguna platform streaming seperti Netflix, Disney+, atau Prime Video menghadapi tantangan dalam memilih film karena banyaknya pilihan yang tersedia. Sistem rekomendasi membantu pengguna menemukan konten yang sesuai dengan preferensi mereka tanpa harus mencari secara manual. Salah satu pendekatan efektif yang banyak digunakan adalah **content-based filtering**, yang memberikan rekomendasi berdasarkan kemiripan konten antar film.

Proyek ini membangun sistem rekomendasi film menggunakan pendekatan content-based filtering dengan memanfaatkan data film dari [Top-Rated TMDB Movies Dataset](https://www.kaggle.com/datasets/ahsanaseer/top-rated-tmdb-movies-10k/data). Sistem ini menggabungkan informasi **genre** dan **overview** (deskripsi film) untuk menghitung kemiripan antar film, lalu merekomendasikan film yang mirip berdasarkan input pengguna.

Menurut Ricci et al. (2011), sistem rekomendasi berbasis konten memanfaatkan atribut-atribut item untuk menyarankan item yang mirip dengan preferensi pengguna sebelumnya [1].

**Referensi**:  
[1] F. Ricci, L. Rokach, and B. Shapira, *Introduction to Recommender Systems Handbook*. Springer, 2011.

## Business Understanding
Sistem rekomendasi film sangat penting dalam membantu pengguna menemukan film yang sesuai dengan minat mereka tanpa harus menelusuri ribuan judul secara manual. Terutama pada pengguna baru (cold-start), rekomendasi berbasis interaksi tidak dapat digunakan. Oleh karena itu, dibutuhkan pendekatan berbasis konten (content-based filtering) yang memanfaatkan informasi dari film itu sendiri.

### Problem Statements

- Bagaimana merekomendasikan film yang relevan tanpa menggunakan riwayat interaksi pengguna?
- Bagaimana mengenali kemiripan antar film berdasarkan kombinasi genre dan deskripsi cerita?

### Goals

- Mengembangkan sistem rekomendasi berbasis konten yang mampu memberikan 10 rekomendasi film mirip berdasarkan kemiripan genre dan overview.
- Menjaga cakupan data sebanyak mungkin agar sistem dapat bekerja pada seluruh film yang tersedia dalam dataset.

### Solution Approach

- Content-Based Filtering menggunakan TF-IDF + Cosine Similarity
    - Menggabungkan fitur `genre` dan `overview` menjadi satu representasi teks.
    - Menggunakan **TF-IDF Vectorizer** untuk mengekstrak kata-kata penting.
    - Mengukur kemiripan antar film menggunakan **Cosine Similarity**.
    - Menghasilkan rekomendasi film berdasarkan skor kemiripan tertinggi.

## Data Understanding
Dataset yang digunakan berisi informasi film dari berbagai genre, termasuk judul, genre, ringkasan cerita, dan rata-rata rating pengguna. Dataset berjumlah sekitar **10.000 film** dan bersumber dari kaggle [Top-Rated TMDB Movies Dataset](https://www.kaggle.com/datasets/ahsanaseer/top-rated-tmdb-movies-10k/data).

### Fitur Dataset

![image.png](https://i.imgur.com/hrDLRlU.png)

Berikut adalah deskripsi masing-masing fitur dalam dataset:

| Fitur               | Tipe Data     | Deskripsi                                                                 |
|---------------------|---------------|---------------------------------------------------------------------------|
| `id`                | Integer       | ID film sesuai database sumber                                            |
| `title`             | String        | Judul film                                                                |
| `genre`             | String        | Genre film (contoh: "Crime", "Adventure", dsb.)                           |
| `original_language` | String        | Bahasa asli saat film dirilis                                             |
| `overview`          | String        | Ringkasan cerita dari film                                                |
| `popularity`        | Float         | Indeks popularitas film                                                   |
| `release_date`      | Date          | Tanggal perilisan film                                                    |
| `vote_average`      | Float         | Nilai rata-rata rating dari pengguna                                      |
| `vote_count`        | Integer       | Jumlah rating yang diberikan pengguna                                     |

### Missing & Duplicated Values
![image.png](https://i.imgur.com/PYUvabv.png)
- Dataset tidak memiliki data yang duplikat.
- Sebagian film tidak memiliki nilai `genre` atau `overview`.

## Exploratory Data Analysis
### Distribusi Rating Film
![image.png](https://i.imgur.com/c6gkd9e.png)
- Distribusi rating film menunjukkan puncak di sekitar 6.5 hingga 7.5, dengan mayoritas film memiliki rating di atas 6.0, mencerminkan dataset yang didominasi oleh film berkualitas tinggi.

### Genre Terpopuler
![image.png](https://i.imgur.com/907wKHW.png)

## Data Preparation

Langkah-langkah yang dilakukan:

1. **Mengisi nilai kosong** di kolom `overview` dan `genre` dengan string kosong (`''`) agar tidak kehilangan data.
2. **Menggabungkan Fitur**
   - Fitur `genre` dan `overview` digabungkan menjadi satu kolom `combined_features`.
   - Tujuannya agar representasi teks lebih kaya, mencakup tema dan ringkasan cerita film.
3. **Penerapan TF-IDF Vectorizer**:
   - Stopwords dihapus agar hanya kata penting yang dihitung.
   - Menghasilkan matriks sparse dari kombinasi teks.
   - TF-IDF (**Term Frequency - Inverse Document Frequency**) adalah metode untuk mengubah teks menjadi vektor numerik.
   - Rumus dasar: 
    ![image.png](https://quicklatex.com/cache3/31/ql_def842c37ffe4b1b96c38e775602d131_l3.png)

     di mana:
     - TF (Term Frequency): frekuensi kata \( t \) muncul dalam dokumen \( d \).
     - IDF (Inverse Document Frequency): menghitung keunikan kata dalam seluruh dokumen, dirumuskan sebagai:
        ![image.png](https://quicklatex.com/cache3/b1/ql_1cf96e09b4d8a76ca94550b3bacc18b1_l3.png)

     dengan \( N \) = jumlah total dokumen, dan DF(t) = jumlah dokumen yang mengandung kata \( t \).

   - Kata-kata yang sering muncul di satu dokumen tapi jarang muncul di dokumen lain akan memiliki nilai TF-IDF tinggi, menandakan kata tersebut penting dalam dokumen tersebut.

## Modeling

Model yang digunakan dalam proyek ini adalah **Content-Based Filtering**, yaitu pendekatan yang merekomendasikan item berdasarkan kemiripan fitur kontennya. Dalam konteks ini, yang dibandingkan adalah **genre** dan **overview** dari masing-masing film.

### Menggunakan Cosine Similarity
   - Setelah teks dikonversi ke vektor TF-IDF, digunakan **cosine similarity** untuk mengukur kemiripan antar film.
   - Cosine similarity mengukur sudut antara dua vektor dalam ruang berdimensi tinggi.
   - Rumus cosine similarity:
        ![image.png](https://quicklatex.com/cache3/d8/ql_8caf88e8f020890cac0c68fb49fa3ad8_l3.png)

     di mana:
     - A.B adalah dot product antara vektor A dan B.
     - \( \|A\| \) dan \( \|B\| \) adalah panjang (magnitudo) dari masing-masing vektor.

   - Nilai cosine similarity berkisar dari 0 hingga 1:
     - 1 artinya vektor identik (film sangat mirip).
     - 0 artinya tidak ada kemiripan.

### Rekomendasi Film
- Dihitung skor kemiripan antara satu film dengan semua film lain.
- Diambil **10 film dengan skor tertinggi** (selain dirinya sendiri) sebagai rekomendasi.

### Contoh Output Rekomendasi:
![image.png](https://i.imgur.com/COEXxbP.png)

## Evaluation

Evaluasi dilakukan menggunakan metrik **Precision@K**, yaitu proporsi film relevan di antara *K* rekomendasi teratas yang diberikan oleh sistem. Dalam konteks ini, kita ingin mengetahui seberapa banyak dari 10 film yang direkomendasikan benar-benar mirip atau relevan dengan film input menurut data ground truth atau anotasi manual.

### Langkah-langkah Evaluasi dengan Precision@10:

1. **Pilih beberapa film sebagai titik uji** (misalnya: `Midsommar`, `RoboCop`, `Parasite`).
2. Untuk setiap film tersebut:
   - Gunakan model untuk menghasilkan 10 rekomendasi film teratas.
   - Bandingkan hasil rekomendasi tersebut dengan daftar film yang dianggap relevan secara manual atau dari sumber referensi eksternal.
3. Hitung **Precision@10** untuk setiap film menggunakan rumus:

    ![image.png](https://quicklatex.com/cache3/fe/ql_7b9b5356ff65abbda0f1ba37a04ffffe_l3.png)

### Hasil Evaluasi Precision@10:

![image.png](https://i.imgur.com/nL8JbDk.png)

### Analisis:
- Model memiliki kemampuan generalisasi yang baik untuk film dari berbagai genre.
- Precision tinggi (≥ 0.90) menunjukkan sistem mampu merekomendasikan film yang sesuai preferensi atau tema film input.

