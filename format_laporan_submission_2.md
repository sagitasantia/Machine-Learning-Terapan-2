# Laporan Proyek Machine Learning - Anggraini Sagita Santia Putri

## Project Overview

  Seiring berkembangnya kebutuhan pengguna dalam memilih bahan bacaan yang sesuai, sistem rekomendasi menjadi salah satu solusi penting untuk membantu menemukan buku yang relevan dengan preferensi masing-masing individu. Banyaknya pilihan yang tersedia dapat menimbulkan kebingungan, sehingga diperlukan sistem cerdas yang dapat memberikan saran buku secara otomatis.

Proyek ini bertujuan untuk membangun sistem rekomendasi buku dengan pendekatan machine learning menggunakan dua metode utama, yaitu content-based filtering dan collaborative filtering. Content-based filtering memberikan rekomendasi berdasarkan kemiripan konten buku, seperti nama penulis atau genre, sedangkan collaborative filtering memberikan rekomendasi berdasarkan pola perilaku pengguna lain yang memiliki preferensi serupa.

Penelitian oleh Hendrayana dan Wibowo (2024) menunjukkan bahwa metode content-based filtering efektif dalam memberikan hasil rekomendasi buku berdasarkan kesamaan kata kunci dan kategori buku. Mereka berhasil memperoleh nilai kemiripan tertinggi sebesar 0,775 untuk buku "Panduan Lengkap Agama Islam Secara Kafah" berdasarkan kata kunci "Panduan Agama Islam". Pendekatan serupa juga telah diterapkan oleh BISA AI Academy dalam proyek sistem rekomendasi buku dengan memanfaatkan algoritma content-based dan collaborative filtering, yang terbukti mampu memberikan rekomendasi yang tepat sasaran bagi pengguna.

## Business Understanding

   Setiap pembaca tentu memiliki preferensi dan minat yang berbeda dalam memilih buku. Ada yang menyukai karya dari penulis tertentu, ada pula yang lebih tertarik pada genre atau bahkan sampul buku. Perbedaan preferensi ini menjadi alasan kuat untuk menghadirkan sistem rekomendasi buku yang mampu menyesuaikan saran dengan kebiasaan dan minat pembaca. Dengan bantuan sistem rekomendasi, pengguna dapat lebih mudah menemukan bacaan yang relevan dan sesuai dengan selera mereka.

Namun, terdapat tantangan dalam menghubungkan preferensi pengguna dengan rekomendasi buku yang tepat. Oleh karena itu, problem statement dan goal proyek ini dirumuskan sebagai berikut:

### Problem Statements

- Bagaimana merancang sistem yang efektif dalam memberikan rekomendasi buku yang relevan dengan preferensi pengguna berdasarkan penulis buku yang mereka sukai?
- Bagaimana memastikan rekomendasi buku juga mempertimbangkan popularitas atau penilaian tertinggi dari pengguna lain untuk meningkatkan kepuasan pengguna?

### Goals

- Membangun sistem rekomendasi yang mampu memberikan rekomendasi buku secara akurat berdasarkan penulis yang disukai pengguna.
- Mengukur dan mengevaluasi efektivitas sistem rekomendasi menggunakan data rating pengguna lain untuk meningkatkan relevansi dan kepuasan pengguna.

**Solution Approach**:

Proyek ini mengambil dua pendekatan dalam sistem rekomendasi berbasis machine learning, yaitu:

### Content-Based Filtering
- Metode ini memberikan saran berdasarkan karakteristik item (buku) yang sebelumnya disukai oleh pengguna. Sistem mempelajari preferensi pengguna melalui fitur-fitur seperti nama penulis, lalu merekomendasikan buku lain yang memiliki kemiripan.
- Model preferensi pengguna yang terbentuk dari histori interaksi.
- Informasi deskriptif dari item (buku), seperti penulis.

### Collaborative Filtering
- Berbeda dengan content-based, metode ini memanfaatkan data dari pengguna lain untuk memberikan rekomendasi. Sistem mencari pola kesamaan antar pengguna dalam memberikan rating terhadap buku, dan menyarankan buku yang disukai oleh pengguna dengan preferensi serupa. Terdapat dua pendekatan umum dalam collaborative filtering:
- Memory-based: menggunakan seluruh data interaksi pengguna secara langsung.
- Model-based: menggunakan algoritma machine learning untuk mempelajari pola hubungan antar pengguna dan item.

Pada proyek ini, pendekatan content-based digunakan untuk memberikan rekomendasi berdasarkan penulis buku, sedangkan pendekatan collaborative filtering digunakan untuk merekomendasikan buku-buku dengan rating tinggi yang diberikan oleh pengguna lain.

## Data Understanding
Sumber data yang digunakan dalam proyek ini berasal dari Kaggle Book Recommendation Dataset, yang berisi informasi mengenai buku, pengguna, dan penilaian (rating) yang diberikan.(https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset).

Books.csv: Memuat 271.360 entri data buku, mencakup:

- ISBN (kode unik buku)

- Book-Title (judul buku)

- Book-Author (penulis)

- Year-Of-Publication (tahun terbit)

- Publisher (penerbit)

- Tautan gambar sampul buku berukuran kecil hingga besar

Ratings.csv: Terdiri dari 1.149.780 entri penilaian oleh pengguna, mencakup:

- User-ID (identitas pengguna)

- ISBN (kode buku yang dinilai)

- Book-Rating (nilai penilaian dari 0 hingga 10)

Berdasarkan hasil eksplorasi awal, tipe data dari kolom Book-Rating dan User-ID adalah numerik, sementara kolom ISBN bersifat objek/string. Pada berkas Books.csv, seluruh kolom bertipe objek, termasuk Year-Of-Publication, sehingga perlu konversi tipe data ke numerik saat tahap praproses.

Distribusi rating menunjukkan banyak pengguna memberikan nilai nol, yang mungkin menandakan tidak memberikan rating eksplisit. Selain itu, beberapa kolom seperti nama penulis dan penerbit memiliki data null yang perlu ditangani. Distribusi tahun terbit menunjukkan rentang tahun yang luas, namun fokus analisis dibatasi pada buku-buku terbitan 1960 hingga 2021 untuk menjaga relevansi data.bahwa pengguna tidak menyukai atau tidak memberikan penilaian eksplisit. Distribusi tahun penerbitan didominasi oleh buku yang terbit antara tahun 1960–2021.

![image](https://github.com/user-attachments/assets/33ba5c74-9f29-4a53-8526-871e5a757b7d)

Perintah rating.isnull().sum() dan book.isnull().sum() digunakan untuk menghitung jumlah nilai missing (null) atau kosong dalam setiap kolom dari dataset rating dan book.

Hasil untuk dataset rating:
user_id: Tidak ada nilai kosong, semua data lengkap (0 missing).
ISBN: Tidak ada nilai kosong, semua data lengkap (0 missing).
rating: Tidak ada nilai kosong, semua data lengkap (0 missing).

Hasil untuk dataset book:
ISBN: Tidak ada nilai kosong, semua data lengkap (0 missing).
title: Tidak ada nilai kosong, semua data lengkap (0 missing).
author: Ada 2 nilai kosong di kolom penulis.
year: Tidak ada nilai kosong, semua data lengkap (0 missing).
Publisher: Ada 2 nilai kosong di kolom penerbit.
Image-URL-S, Image-URL-M: Tidak ada nilai kosong, semua data lengkap (0 missing).
Image-URL-L: Ada 3 nilai kosong di kolom URL gambar ukuran besar.
Dengan kata lain, dataset rating tidak memiliki data yang hilang, sementara dataset book memiliki beberapa nilai yang hilang pada kolom author, Publisher, dan Image-URL-L.

![image](https://github.com/user-attachments/assets/027c1a37-eed5-4e85-bbf1-b72a44ae8119)

Outliers pada kolom 'rating':

Output menunjukkan tidak ditemukan outlier pada kolom rating, artinya semua nilai rating berada dalam batas normal atau range nilai yang diharapkan.

Outliers pada kolom 'Year-Of-Publication':

Output menampilkan total 9095 data yang dianggap sebagai outlier. Data ini terdiri dari tahun-tahun publikasi buku yang terlalu lama (misalnya, tahun-tahun di bawah 1970) atau bahkan bernilai 0.
Tahun publikasi seperti 0 dianggap sebagai data invalid karena tidak masuk akal secara logika dalam konteks penerbitan buku.
Tahun yang sangat tua (misalnya tahun 1929, 1952, 1961, 1968, 1966, dan 1970) dianggap outlier karena kemungkinan di luar periode yang diinginkan atau kurang relevan untuk analisis rekomendasi buku terkini.


## EDA 

### EDA Univariate

![download](https://github.com/user-attachments/assets/5fbf081a-13a9-4ff1-a15a-5a9e62af2f0c)

Gambar pertama menunjukkan distribusi rating buku. Terlihat ada banyak rating 0, yang mungkin menunjukkan data yang tidak lengkap atau rating kosong. Setelah itu, distribusi rating agak merata dengan sebagian kecil buku yang mendapat rating tinggi (8, 9, 10). Ini bisa menunjukkan bahwa sebagian besar buku diberi rating rendah, atau banyak data yang tidak memiliki rating.

![download](https://github.com/user-attachments/assets/acea94b0-52ef-4772-85c7-6ef8142fc49b)

Grafik ini menunjukkan jumlah buku yang diterbitkan setiap tahun. Terlihat bahwa penerbitan buku mulai meningkat pesat setelah tahun 1980, dengan lonjakan terbesar di sekitar tahun 2000-an. Ini menandakan adanya perubahan besar dalam industri penerbitan, mungkin terkait dengan kemajuan teknologi percetakan dan digitalisasi. Setelah 2010, jumlah penerbitan mulai stagnan atau menurun sedikit, yang bisa disebabkan oleh tren baru seperti e-book dan self-publishing.

![download](https://github.com/user-attachments/assets/a5a427de-10d5-4b15-8acf-700fc7bc39df)

Grafik ini menunjukkan rata-rata rating yang diberikan untuk buku berdasarkan tahun penerbitannya. Dapat dilihat bahwa rata-rata rating untuk buku yang diterbitkan antara tahun 1960 hingga 2000-an cukup stabil, dengan angka sekitar 3. Hal ini menandakan bahwa tingkat kepuasan pembaca terhadap buku yang diterbitkan pada periode tersebut relatif seragam.

Namun, setelah tahun 2010, terdapat lonjakan besar pada rata-rata rating, dengan angka yang jauh lebih tinggi dibandingkan tahun-tahun sebelumnya. Hal ini mungkin menunjukkan bahwa buku-buku yang lebih baru mendapatkan sambutan lebih positif dari pembaca, atau ada perubahan dalam cara pembaca memberikan rating, seperti lebih banyak buku dari penulis populer atau peningkatan interaksi pembaca melalui platform digital.

Secara keseluruhan, grafik ini menunjukkan bahwa buku yang lebih baru cenderung mendapatkan rating yang lebih tinggi, sementara buku yang lebih lama memiliki rating yang lebih konsisten dan lebih rendah.

![download](https://github.com/user-attachments/assets/695a7cc9-f00f-4771-bcfe-673e095c7fc5)

Grafik ini menunjukkan rata-rata rating per penerbit (publisher). Setiap batang menggambarkan rata-rata rating yang diberikan untuk buku yang diterbitkan oleh masing-masing penerbit. Dari grafik, bisa terlihat bahwa beberapa penerbit, seperti Penguin Books dan Perennial, mendapatkan rata-rata rating yang lebih tinggi, sementara penerbit lain seperti Silhouette memiliki rating yang lebih rendah.

### EDA Multivariate

![download](https://github.com/user-attachments/assets/0c958499-f330-4779-a95e-d5e70b341918)

Grafik ini menunjukkan jumlah rating yang diberikan per tahun publikasi buku. Setiap batang mewakili jumlah total rating untuk buku-buku yang diterbitkan pada tahun tertentu, dengan warna yang mewakili kategori rating (misalnya, rating 0 hingga rating 10).

![download](https://github.com/user-attachments/assets/04f00bad-5b29-4c36-85c4-60c110063a05)

Grafik ini menunjukkan hubungan antara **rating** dan **tahun publikasi** buku dari beberapa penulis terkenal. Setiap titik mewakili buku yang diterbitkan pada tahun tertentu dan diberi rating oleh pembaca. Warna titik menunjukkan penulis buku tersebut.

Dari grafik, terlihat bahwa sebagian besar buku dengan **rating tinggi** (di atas 8) diterbitkan antara **1990-an hingga 2000-an**, yang menunjukkan bahwa buku-buku yang lebih baru mendapatkan rating yang lebih tinggi. Beberapa penulis seperti **Stephen King**, **Danielle Steel**, dan **John Grisham** menunjukkan bahwa **buku-buku baru** cenderung mendapatkan rating yang lebih tinggi, sementara beberapa penulis dengan karier lebih lama, seperti **Mary Higgins Clark**, memiliki rating yang lebih beragam, meskipun dengan angka lebih rendah. Grafik ini memberikan gambaran jelas bahwa **buku yang lebih baru** mendapatkan lebih banyak perhatian positif dari pembaca.

### Matriks Korelasi

![download](https://github.com/user-attachments/assets/5787c921-478a-4986-acdf-c7a699b56d2d)

Grafik ini menunjukkan **matriks korelasi** antara dua variabel, yaitu **rating** dan **tahun** publikasi buku. Matriks ini menggambarkan hubungan statistik antara kedua variabel tersebut.

Dari matriks tersebut, terlihat bahwa **korelasi antara rating dan tahun publikasi sangat rendah**, dengan nilai **0.04**. Artinya, tidak ada hubungan yang kuat antara **tahun buku diterbitkan** dan **rating** yang diterima buku tersebut. Ini menunjukkan bahwa meskipun ada buku yang lebih baru, **rating** buku tidak tergantung secara signifikan pada tahun penerbitannya.

Korelasi yang rendah ini menunjukkan bahwa **rating buku** tidak terpengaruh secara langsung oleh **tahun publikasi**. Mungkin ada faktor lain yang lebih mempengaruhi rating buku, seperti **genre**, **penulis**, atau **popularitas** buku tersebut.

## Data Preparation

Dikarenakan jumlah data yang sangat besar, dilakukan pengambilan sampel agar proses pelatihan model menjadi lebih efisien. Dari dataset Books.csv, diambil sebanyak 25.000 baris data, sedangkan dari Ratings.csv, diambil 5.000 baris data.

![image](https://github.com/user-attachments/assets/3c8078de-92e7-43c3-ba47-39a213f44a7d)

Menghapus data duplikat dan nilai kosong pada kolom-kolom penting seperti author dan Publisher

![image](https://github.com/user-attachments/assets/f1dda7fa-7c82-4ac3-9c44-2acb55752779)

![image](https://github.com/user-attachments/assets/235a7805-a1a9-4aa1-a1cb-848141e650c9)

Mengambil masing-masing kolom `ISBN`, `title`, `author`, dan `year` dari DataFrame `book` lalu mengubahnya menjadi list dengan `.tolist()`
- Menyusun DataFrame baru bernama `new_book` yang berisi kolom `book_ISBN`, `book_title`, `book_author`, dan `book_year_of_publication`
- Mengonversi kolom `book_year_of_publication` menjadi tipe data integer agar kompatibel dengan proses modeling dan analisis statistik
- Mengonversi `book_year_of_publication` menjadi tipe integer untuk memastikan kompatibilitas dengan proses pemodelan 

![image](https://github.com/user-attachments/assets/2877a0bf-26e2-4795-b359-7fbe7d4a3dbf)
![image](https://github.com/user-attachments/assets/ae2a55fb-43a4-445a-9f9f-94e61f02b81e)

mengubah ID pengguna menjadi angka agar model bisa memprosesnya dengan lebih mudah. Ini dilakukan dengan mengambil semua ID unik dari pengguna dan memberi angka urut untuk masing-masing ID. 

![image](https://github.com/user-attachments/assets/a1d59b46-0a3c-4551-8887-05df98751d2a)

Setelah data rating diubah menjadi tipe angka, kode ini menunjukkan jumlah pengguna (679), jumlah buku (4688), dan rentang rating dari 0 hingga 10.

![image](https://github.com/user-attachments/assets/d50661d1-794f-4bdb-837a-f84ac36266de)

mengacak urutan data dalam dataset rating dengan menggunakan fungsi .sample(frac=1, random_state=42). Hasilnya adalah data yang sudah diacak, yang memudahkan untuk membagi dataset menjadi data latih dan data uji secara acak.

![image](https://github.com/user-attachments/assets/9188c44b-4e6e-43d7-8bb3-15117e5f05c1)
- fungsi `TfidfVectorizer()` dari library sklearn. Di sini, beberapa kata penting dari kolom `book_author` akan diambil untuk mengidentifikasi sistem rekomendasi berdasarkan penulis yang sama.
- Selanjutnya, lakukan proses fit dan transformasi pada daftar book_author menggunakan TfidfVectorizer(), yang akan mengubahnya menjadi matriks. Sehingga tercipta output seperti dibawah ini. 

![image](https://github.com/user-attachments/assets/a65c0e33-ab99-4b41-b150-d08ca3445219)

hitung derajat kesamaan antar buku menggunakan teknik cosine similarity. Dengan menggunakan fungsi `cosine_similarity` dari library sklearn, kita dapat menghitung seberapa mirip setiap buku berdasarkan representasi numeriknya, yang menghasilkan output berupa matriks kesamaan antar buku.


## Modeling
### Content Based Filtering

Content-Based Filtering memiliki kelebihan utama dalam hal personalisasi yang tinggi. Model ini memberikan rekomendasi yang sangat relevan dengan preferensi pengguna berdasarkan atribut buku yang sudah diketahui, seperti penulis, genre, atau deskripsi buku. Karena model ini mengandalkan data tentang buku itu sendiri, ia tidak membutuhkan data pengguna lain untuk memberikan rekomendasi, yang menjadikannya sangat berguna terutama pada sistem dengan sedikit pengguna atau data interaksi yang terbatas. Selain itu, model ini memiliki transparansi yang jelas, karena kita bisa dengan mudah menjelaskan mengapa sebuah buku direkomendasikan, misalnya karena buku tersebut memiliki genre atau penulis yang serupa dengan buku yang sudah disukai pengguna. Ini memungkinkan pengguna untuk memahami dasar dari setiap rekomendasi yang diberikan. Kelebihan lainnya adalah kemampuannya untuk memberikan rekomendasi yang sangat spesifik dan langsung sesuai dengan kebutuhan individu, sehingga meningkatkan pengalaman pengguna.

Cara kerja dari pendekatan ini adalah dengan memanfaatkan metode TF-IDF (Term Frequency-Inverse Document Frequency). Metode ini menghitung pentingnya sebuah kata berdasarkan seberapa sering kata tersebut muncul dalam suatu dokumen dibandingkan dengan seluruh dokumen yang tersedia. Dalam konteks ini, dokumen merujuk pada nama penulis buku.

Setelah kata-kata dalam nama penulis diubah ke dalam bentuk numerik, langkah selanjutnya adalah menghitung tingkat kesamaan antar buku menggunakan cosine similarity. Cosine similarity digunakan untuk mengukur sejauh mana dua buku memiliki kesamaan berdasarkan kata-kata yang digunakan dalam nama penulis.

Dari hasil perhitungan tersebut, rekomendasi buku dibuat dengan mengambil buku yang memiliki nilai kesamaan paling tinggi dengan buku yang sudah dibaca pengguna.

Hasil rekomendasi contoh untuk buku "The Star Rover":

The Sea Wolf - Jack London

The Call of the Wild: Complete and Unabridged - Jack London

The Call of the Wild: And Selected Stories - Jack London

White Fang - Jack London

Kelebihan:

Cepat dalam menghasilkan rekomendasi yang sesuai dengan preferensi spesifik pengguna.

Mudah diterapkan dan dipahami.

Kekurangan:

Terbatas hanya mempertimbangkan aspek tertentu seperti penulis, tanpa melibatkan aspek popularitas atau preferensi umum lainnya.

### Evaluation Content Based Filtering

Evaluasi model dilakukan dengan mengukur akurasi hasil rekomendasi berdasarkan buku yang telah dibaca oleh pengguna. Akurasi diukur berdasarkan seberapa sering buku yang direkomendasikan memiliki penulis yang sama dengan buku yang telah dibaca oleh pengguna sebelumnya. Model ini menggunakan teknik pengukuran precision untuk menentukan akurasi rekomendasi. 

![image](https://github.com/user-attachments/assets/9f7607dd-ad45-4404-8c94-b817b166cb33)

- Hasil rekomendasinya yang diberikan oleh sistem berdasarkan buku di atas yaitu.

![image](https://github.com/user-attachments/assets/b8b0016d-1833-4395-9792-e1ce5b457a47)

### Collaborative Filtering

Collaborative Filtering, di sisi lain, memiliki kelebihan dalam hal kemampuan untuk memanfaatkan data dari banyak pengguna untuk memberikan rekomendasi. Dengan mengandalkan pola interaksi antara pengguna dan buku, model ini dapat merekomendasikan buku yang mungkin tidak terduga oleh pengguna, berdasarkan preferensi pengguna lain yang memiliki kesamaan. Hal ini membuat Collaborative Filtering sangat efektif untuk menemukan buku yang relevan yang mungkin tidak dikenal oleh pengguna sebelumnya, memperkenalkan variasi dalam rekomendasi. Selain itu, model ini semakin kuat seiring dengan pertambahan jumlah pengguna dan data interaksi, karena semakin banyak data yang tersedia, semakin akurat sistem dalam memberikan rekomendasi. Salah satu keunggulan Collaborative Filtering adalah kemampuannya untuk memberikan rekomendasi yang lebih luas, yang tidak terikat pada atribut spesifik buku, melainkan lebih pada preferensi kolektif dari komunitas pengguna. Namun, kelemahan dari model ini adalah adanya masalah "cold start", yaitu ketika ada buku baru atau pengguna baru yang belum memiliki cukup interaksi untuk membuat rekomendasi yang relevan.

### Collaborative Filtering

**Cara Kerjanya:**

1. **Persiapan Data:**

   * **Mengubah ID Pengguna dan Buku ke Angka**: Sebelum sistem bisa bekerja, kita perlu mengubah informasi pengguna dan buku menjadi angka. Misalnya, setiap pengguna akan diberi angka 0, 1, 2, dan seterusnya. Begitu juga dengan setiap buku yang diberi angka unik.
2. **Membangun Model Rekomendasi:**

   * Setelah data diubah menjadi angka, kita menggunakan sistem cerdas (seperti otak buatan) untuk mempelajari pola dari kebiasaan pengguna. Sistem ini akan mempelajari buku mana yang sering dibaca oleh pengguna dengan kesukaan yang sama.
3. **Proses Pelatihan:**

   * Model ini dilatih untuk memprediksi buku apa yang mungkin disukai oleh pengguna berdasarkan pola yang ditemukan dari data pengguna dan buku yang sudah mereka beri rating.

**Kelebihan:**

* Sistem ini bisa memberikan rekomendasi buku yang lebih sesuai dengan minat setiap pengguna, berdasarkan apa yang disukai pengguna lain dengan kesukaan yang mirip.
* Bisa menangani banyak data dan memberikan hasil yang baik.

**Kekurangan:**

* Untuk mendapatkan rekomendasi yang akurat, sistem ini membutuhkan banyak data agar bisa bekerja dengan baik. 

### Evaluation Collaborative Filtering

![download](https://github.com/user-attachments/assets/3fa17a65-276d-4d5d-8b99-51d832a24a2b)

- model mengalami penurunan yang stabil dalam loss dan RMSE baik pada data pelatihan maupun data validasi, meskipun ada sedikit fluktuasi pada epoch tertentu. Model terus memperbaiki dirinya selama 20 epoch, tetapi ada sedikit tanda bahwa penurunan error mulai melambat setelah beberapa epoch terakhir, yang mungkin menunjukkan bahwa model sudah mulai mencapai titik optimumnya.

- hasil rekomendasinya sebagai berikut.
  
![image](https://github.com/user-attachments/assets/08751636-fe51-4174-bb1a-c88b0a01943f)


### Evaluation

Dalam proyek sistem rekomendasi buku ini, dua metrik utama digunakan untuk mengevaluasi kinerja masing-masing model, yaitu Akurasi untuk Content-Based Filtering dan Root Mean Squared Error (RMSE) untuk Collaborative Filtering. Metrik ini dipilih karena masing-masing sesuai dengan tujuan dan karakteristik dari kedua metode yang digunakan.

- Akurasi (Accuracy) untuk Content-Based Filtering:

Mengapa memilih akurasi?

Akurasi digunakan untuk mengukur seberapa baik rekomendasi yang diberikan oleh model sesuai dengan preferensi pengguna, terutama berdasarkan penulis yang disukai oleh pengguna. Akurasi dihitung dengan membandingkan rekomendasi yang diberikan dengan preferensi atau minat pengguna.

Model Content-Based Filtering bekerja dengan mempelajari penulis atau genre buku yang sudah disukai pengguna, dan rekomendasi yang relevan diharapkan bisa lebih tepat sesuai dengan keinginan pengguna.

- Cara menghitung akurasi:

Akurasi dihitung dengan melihat persentase rekomendasi yang sesuai dengan preferensi pengguna, misalnya apakah buku yang direkomendasikan oleh sistem sesuai dengan apa yang biasanya disukai oleh pengguna (berdasarkan penulis yang mereka sukai).

Hasil 80% akurasi menunjukkan bahwa sistem dapat memberikan rekomendasi yang cukup tepat sesuai dengan penulis buku yang disukai oleh pengguna, namun masih ada ruang untuk perbaikan agar dapat lebih relevan lagi.

Root Mean Squared Error (RMSE) untuk Collaborative Filtering:

Mengapa memilih RMSE?

RMSE digunakan untuk mengukur seberapa besar kesalahan atau selisih antara rating yang diprediksi oleh model dan rating asli yang diberikan oleh pengguna. RMSE cocok untuk Collaborative Filtering karena sistem ini memprediksi rating buku berdasarkan perilaku dan rating pengguna lain yang memiliki kesamaan preferensi.

RMSE memberikan gambaran yang jelas mengenai seberapa baik model dalam memprediksi rating yang akan diberikan pengguna terhadap buku yang belum mereka beri rating.

- Cara menghitung RMSE:

RMSE dihitung dengan menghitung selisih antara rating yang diprediksi oleh model dengan rating yang sebenarnya. Semakin kecil nilai RMSE, semakin akurat prediksi rating yang diberikan oleh model.

Pada hasil pelatihan, RMSE menurun dari 0.4298 pada epoch pertama menjadi 0.3475 pada epoch terakhir, yang menunjukkan bahwa model Collaborative Filtering semakin baik dalam memprediksi rating yang akan diberikan oleh pengguna seiring berjalannya waktu.

Relevansi Metrik dengan Proyek
Akurasi sangat relevan untuk Content-Based Filtering karena tujuan utama model ini adalah memberikan rekomendasi yang tepat berdasarkan preferensi pengguna terhadap penulis atau genre. Metrik ini akan menunjukkan seberapa baik sistem dalam mencocokkan rekomendasi dengan minat pengguna yang sudah ada.

RMSE sangat cocok untuk Collaborative Filtering karena model ini berfokus pada memahami hubungan antar pengguna dan memberikan rekomendasi berdasarkan rating atau penilaian pengguna lain. Metrik ini membantu mengukur seberapa akurately model memprediksi rating yang akan diberikan oleh pengguna, yang merupakan tujuan utama dari metode Collaborative Filtering.


**Hasil Evaluasi:**

1. **Content-Based Filtering**:

   * Model ini memberikan hasil yang baik dalam **memberikan rekomendasi berdasarkan penulis buku** yang relevan dengan preferensi pengguna. Dengan **akurasi 80%**, model ini menunjukkan bahwa pendekatan berbasis penulis memberikan hasil yang efektif.

2. **Collaborative Filtering**:

   * Model ini memberikan hasil yang baik dalam **mempertimbangkan rating pengguna lain** untuk memberikan rekomendasi berdasarkan **popularitas buku**. Dengan **penurunan RMSE** yang menunjukkan peningkatan akurasi prediksi rating, metode ini terbukti efektif dalam memberikan rekomendasi berdasarkan pola rating pengguna lain.

**Komparasi Skema:**

* **Content-Based Filtering** lebih cocok untuk **memberikan rekomendasi berbasis penulis**, dan memberikan hasil yang cepat serta relevan.
* **Collaborative Filtering** lebih tepat untuk **mempertimbangkan rating dan popularitas buku** berdasarkan data pengguna lain, meskipun memerlukan dataset yang lebih besar dan pelatihan yang lebih lama.

**Model Terbaik:**

* **Content-Based Filtering** lebih disarankan untuk **rekomendasi buku berdasarkan penulis**, sesuai dengan **problem statement pertama**.
* **Collaborative Filtering** lebih baik digunakan untuk **memberikan rekomendasi berdasarkan rating dan popularitas**, sesuai dengan **problem statement kedua**.

### Hubungan dengan Business Understanding

* **Problem Statement 1**: **Bagaimana merancang sistem yang efektif dalam memberikan rekomendasi buku yang relevan dengan preferensi pengguna berdasarkan penulis buku yang mereka sukai?**

  * **Jawaban**: **Content-Based Filtering** sangat efektif dalam memberikan rekomendasi yang relevan berdasarkan **penulis buku** yang disukai pengguna.

* **Problem Statement 2**: **Bagaimana memastikan rekomendasi buku juga mempertimbangkan popularitas atau penilaian tertinggi dari pengguna lain untuk meningkatkan kepuasan pengguna?**

  * **Jawaban**: **Collaborative Filtering** memberikan rekomendasi yang mempertimbangkan rating dan popularitas buku berdasarkan **penilaian pengguna lain**.

* **Goals**:

  * **Goal 1**: Membangun sistem rekomendasi yang akurat berdasarkan penulis buku yang disukai pengguna.

    * **Jawaban**: Telah tercapai dengan baik menggunakan **Content-Based Filtering**.
  * **Goal 2**: Mengukur dan mengevaluasi efektivitas sistem rekomendasi berdasarkan rating pengguna lain.

    * **Jawaban**: Telah tercapai dengan menggunakan **Collaborative Filtering** dan evaluasi **RMSE**.

### Kesimpulan

Dari dua metode yang digunakan dalam sistem rekomendasi buku, yaitu **Content-Based Filtering** dan **Collaborative Filtering**, keduanya memiliki keunggulan masing-masing dalam menjawab **bagaimana merancang sistem yang efektif dalam memberikan rekomendasi buku yang relevan dengan preferensi pengguna berdasarkan penulis buku yang mereka sukai**, serta **bagaimana memastikan rekomendasi buku juga mempertimbangkan popularitas atau penilaian tertinggi dari pengguna lain untuk meningkatkan kepuasan pengguna**.

* **Content-Based Filtering** efektif dalam memberikan rekomendasi buku berdasarkan **penulis buku yang disukai pengguna**. Sistem ini dirancang untuk **menganalisis kesamaan antara penulis buku** dan memberikan rekomendasi berdasarkan **preferensi spesifik pengguna** terhadap penulis tertentu. Dengan hasil **akurasi model mencapai 80%**, ini menunjukkan bahwa model ini berhasil memberikan rekomendasi yang tepat berdasarkan preferensi pengguna terhadap penulis.

* **Collaborative Filtering** dapat memberikan rekomendasi buku berdasarkan **rating pengguna lain** dan **popularitas buku**, yang memungkinkan sistem untuk **mempertimbangkan penilaian tertinggi dari pengguna lain** dalam meningkatkan kepuasan pengguna. Dengan menggunakan **RMSE** untuk mengukur kesalahan prediksi, model ini menunjukkan penurunan **RMSE** yang konsisten, yang menandakan peningkatan akurasi dalam memprediksi rating buku berdasarkan data pengguna lain.

Dengan demikian, kedua model ini telah berhasil mencapai tujuan yang diharapkan, memberikan dampak positif dalam meningkatkan akurasi serta relevansi rekomendasi buku sesuai dengan **preferensi pengguna** dan **rating pengguna lain**.


   
## REFERENSI 

- https://journal.stekom.ac.id/index.php/elkom/article/view/1927/1482
- https://bisa.ai/portofolio/detail/MzM4OQ
