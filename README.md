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

![image.png](https://i.postimg.cc/SKR6WGmT/image.png)

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
![image.png](https://i.postimg.cc/Bb0Pw5GG/image.png)
- Dataset tidak memiliki data yang duplikat.
- Sebagian film tidak memiliki nilai `genre` atau `overview`.
- Untuk menjaga cakupan data secara maksimal, nilai kosong diisi dengan **string kosong ('')** agar semua film tetap dapat direkomendasikan dalam sistem.

## Exploratory Data Analysis
### Distribusi Rating Film
![image.png](https://i.postimg.cc/90Yw34Ww/image.png)
- Distribusi rating film menunjukkan puncak di sekitar 6.5 hingga 7.5, dengan mayoritas film memiliki rating di atas 6.0, mencerminkan dataset yang didominasi oleh film berkualitas tinggi.

### Genre Terpopuler
![image.png](https://i.postimg.cc/9FcqdBpd/image.png)

## Data Preparation

Langkah-langkah yang dilakukan:

1. **Mengisi nilai kosong** di kolom `overview` dan `genre` dengan string kosong (`''`) agar tidak kehilangan data.
2. **Menggabungkan kolom `genre` dan `overview`** menjadi `combined_features` agar sistem mengenali tema dan deskripsi film secara bersamaan.
3. **Penerapan TF-IDF Vectorizer**:
   - Stopwords dihapus agar hanya kata penting yang dihitung.
   - Menghasilkan matriks sparse dari kombinasi teks.
4. **Perhitungan cosine similarity** antara film berdasarkan hasil TF-IDF.

## Modeling

Model yang digunakan dalam proyek ini adalah **Content-Based Filtering**, yaitu pendekatan yang merekomendasikan item berdasarkan kemiripan fitur kontennya. Dalam konteks ini, yang dibandingkan adalah **genre** dan **overview** dari masing-masing film.

### Langkah-Langkah Modeling

1. **Menggabungkan Fitur**
   - Fitur `genre` dan `overview` digabungkan menjadi satu kolom `combined_features`.
   - Tujuannya agar representasi teks lebih kaya, mencakup tema dan ringkasan cerita film.

2. **TF-IDF Vectorization**
   - TF-IDF (**Term Frequency - Inverse Document Frequency**) adalah metode untuk mengubah teks menjadi vektor numerik.
   - Rumus dasar: 
    ![image.png](https://quicklatex.com/cache3/31/ql_def842c37ffe4b1b96c38e775602d131_l3.png)

     di mana:
     - TF (Term Frequency): frekuensi kata \( t \) muncul dalam dokumen \( d \).
     - IDF (Inverse Document Frequency): menghitung keunikan kata dalam seluruh dokumen, dirumuskan sebagai:
        ![image.png](https://quicklatex.com/cache3/b1/ql_1cf96e09b4d8a76ca94550b3bacc18b1_l3.png)

     dengan \( N \) = jumlah total dokumen, dan DF(t) = jumlah dokumen yang mengandung kata \( t \).

   - Kata-kata yang sering muncul di satu dokumen tapi jarang muncul di dokumen lain akan memiliki nilai TF-IDF tinggi, menandakan kata tersebut penting dalam dokumen tersebut.

3. **Cosine Similarity**
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

4. **Rekomendasi Film**
   - Dihitung skor kemiripan antara satu film dengan semua film lain.
   - Diambil **10 film dengan skor tertinggi** (selain dirinya sendiri) sebagai rekomendasi.

### Contoh Output Rekomendasi:
![image.png](https://i.postimg.cc/Cx1BtL9d/image.png)

## Evaluation

Sistem rekomendasi ini menggunakan pendekatan **content-based filtering** yang memanfaatkan kombinasi fitur teks dari `genre` dan `overview` film. Evaluasi dilakukan secara kualitatif berdasarkan relevansi rekomendasi terhadap film acuan.

![image.png](https://i.postimg.cc/Vvdppt9D/image.png)

Sebagai contoh, sistem memberikan 10 rekomendasi film yang mirip dengan *Midsommar* (genre: Horror, Drama, Mystery). Hasil rekomendasi menunjukkan bahwa mayoritas film memiliki genre yang serupa.

# Penghitungan Presisi Model Rekomendasi

Untuk menghitung **presisi** model rekomendasi yang dibuat, kita perlu menentukan seberapa banyak rekomendasi yang relevan dibandingkan dengan total rekomendasi yang diberikan.

## Langkah-langkah Menghitung Presisi

1. **Menentukan Relevansi**  
   Sebuah film dianggap relevan jika genrenya memiliki setidaknya satu genre yang sama dengan film input, yaitu *Midsommar* (**Horror, Drama, Mystery**). Ini adalah pendekatan umum dalam sistem rekomendasi berbasis genre.

2. **Menghitung Rekomendasi yang Relevan**  
   Kita periksa genre setiap film yang direkomendasikan dan bandingkan dengan genre *Midsommar* (**Horror, Drama, atau Mystery**).

3. **Menghitung Presisi**  
   Presisi dihitung dengan rumus:

   ![iamge.png](https://quicklatex.com/cache3/e2/ql_6d2af2d532d10ea2a6bfa329bfb15ce2_l3.png)

## Data Input

- **Genre Midsommar**: Horror, Drama, Mystery  
- **Rekomendasi (10 film)**:

  1. *The Experiment*: Thriller, Drama (**mengandung Drama**)
  2. *He's Out There*: Horror, Thriller (**mengandung Horror**)
  3. *No One Gets Out Alive*: Horror, Thriller, Mystery (**mengandung Horror, Mystery**)
  4. *A Christmas Horror Story*: Horror, Fantasy (**mengandung Horror**)
  5. *Aftermath*: Horror, Crime, Drama, Thriller (**mengandung Horror, Drama**)
  6. *The Crazies*: Mystery, Horror, Action (**mengandung Horror, Mystery**)
  7. *Don't Hang Up*: Horror, Thriller (**mengandung Horror**)
  8. *Munna Bhai M.B.B.S.*: Comedy, Drama (**mengandung Drama**)
  9. *The Children*: Horror, Mystery, Thriller (**mengandung Horror, Mystery**)
  10. *The Visit*: Horror, Thriller, Mystery (**mengandung Horror, Mystery**)

## Analisis

- **Total rekomendasi**: 10 film  
- **Rekomendasi yang relevan**: Kita cek apakah setiap film memiliki setidaknya satu genre yang cocok dengan *Midsommar*:

  - *The Experiment*: ✅ Relevan (ada Drama)
  - *He's Out There*: ✅ Relevan (ada Horror)
  - *No One Gets Out Alive*: ✅ Relevan (ada Horror, Mystery)
  - *A Christmas Horror Story*: ✅ Relevan (ada Horror)
  - *Aftermath*: ✅ Relevan (ada Horror, Drama)
  - *The Crazies*: ✅ Relevan (ada Horror, Mystery)
  - *Don't Hang Up*: ✅ Relevan (ada Horror)
  - *Munna Bhai M.B.B.S.*: ✅ Relevan (ada Drama)
  - *The Children*: ✅ Relevan (ada Horror, Mystery)
  - *The Visit*: ✅ Relevan (ada Horror, Mystery)

Semua **10 film** yang direkomendasikan memiliki **setidaknya satu genre** (Horror, Drama, atau Mystery) yang cocok dengan Midsommar. Jadi, semua rekomendasi dianggap relevan.

## Perhitungan Presisi

![iamge.png](https://quicklatex.com/cache3/35/ql_780aa16aff341502991d119d38793535_l3.png)

## Hasil Akhir

Presisi dari model rekomendasi yang dibuat adalah **1.0** atau **100%**.  
Artinya, semua film yang direkomendasikan relevan dengan genre *Midsommar*.