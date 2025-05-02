# Laporan Proyek Machine Learning - Anggraini Sagita Santia Putri

## Project Overview

   Seiring berkembangnya kebutuhan pengguna dalam memilih bahan bacaan yang sesuai, sistem rekomendasi menjadi salah satu solusi penting untuk membantu menemukan buku yang relevan dengan preferensi masing-masing individu. Banyaknya pilihan yang tersedia dapat menimbulkan kebingungan, sehingga diperlukan sistem cerdas yang dapat memberikan saran buku secara otomatis.
   
   Proyek ini bertujuan untuk membangun sistem rekomendasi buku dengan pendekatan machine learning menggunakan dua metode utama, yaitu content-based filtering dan collaborative filtering. Content-based filtering memberikan rekomendasi berdasarkan kemiripan konten buku, seperti nama penulis atau genre, sedangkan collaborative filtering memberikan rekomendasi berdasarkan pola perilaku pengguna lain yang memiliki preferensi serupa.
    
   Penelitian oleh Hendrayana dan Wibowo (2024) menunjukkan bahwa metode content-based filtering efektif dalam memberikan hasil rekomendasi buku berdasarkan kesamaan kata kunci dan kategori buku. Mereka berhasil memperoleh nilai kemiripan tertinggi sebesar 0,775 untuk buku "Panduan Lengkap Agama Islam Secara Kafah" berdasarkan kata kunci "Panduan Agama Islam". Pendekatan serupa juga telah diterapkan oleh BISA AI Academy dalam proyek sistem rekomendasi buku dengan memanfaatkan algoritma content-based dan collaborative filtering, yang terbukti mampu memberikan rekomendasi yang tepat sasaran bagi pengguna.

## Business Understanding

   Setiap pembaca tentu memiliki preferensi dan minat yang berbeda dalam memilih buku. Ada yang menyukai karya dari penulis tertentu, ada pula yang lebih tertarik pada genre atau bahkan sampul buku. Perbedaan preferensi ini menjadi alasan kuat untuk menghadirkan sistem rekomendasi buku yang mampu menyesuaikan saran dengan kebiasaan dan minat pembaca. Dengan bantuan sistem rekomendasi, pengguna dapat lebih mudah menemukan bacaan yang relevan dan sesuai dengan selera mereka.

### Problem Statements

- Bagaimana merancang sistem yang dapat merekomendasikan buku berdasarkan kesamaan penulis?
- Bagaimana menyusun rekomendasi buku dengan mempertimbangkan penilaian tertinggi dari pengguna lain?

### Goals

- Mengidentifikasi metode untuk memberikan rekomendasi buku berdasarkan informasi penulis.
- Mengetahui cara kerja sistem rekomendasi berdasarkan data rating yang diberikan oleh pengguna.

**Solution Approach**:

Proyek ini mengambil dua pendekatan dalam sistem rekomendasi berbasis machine learning, yaitu:

### Content-Based Filtering
Metode ini memberikan saran berdasarkan karakteristik item (buku) yang sebelumnya disukai oleh pengguna. Sistem mempelajari preferensi pengguna melalui fitur-fitur seperti nama penulis, lalu merekomendasikan buku lain yang memiliki kemiripan.
- Model preferensi pengguna yang terbentuk dari histori interaksi.
- Informasi deskriptif dari item (buku), seperti penulis.

### Collaborative Filtering
Berbeda dengan content-based, metode ini memanfaatkan data dari pengguna lain untuk memberikan rekomendasi. Sistem mencari pola kesamaan antar pengguna dalam memberikan rating terhadap buku, dan menyarankan buku yang disukai oleh pengguna dengan preferensi serupa. Terdapat dua pendekatan umum dalam collaborative filtering:
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
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model sisten rekomendasi yang Anda buat untuk menyelesaikan permasalahan. Sajikan top-N recommendation sebagai output.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menyajikan dua solusi rekomendasi dengan algoritma yang berbeda.
- Menjelaskan kelebihan dan kekurangan dari solusi/pendekatan yang dipilih.

## Evaluation
Pada bagian ini Anda perlu menyebutkan metrik evaluasi yang digunakan. Kemudian, jelaskan hasil proyek berdasarkan metrik evaluasi tersebut.

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.
