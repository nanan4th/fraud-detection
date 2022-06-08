# Credit Card Fraud Detection - Nathanael Victor Darenoh

## Domain Proyek

Pembayaran digital terus berkembang, begitu juga kejahatan cyber yang kerap mengikuti. Penipuan yang berhubungan dengan keuangan (Financial Fraud) merupakan ancaman yang terus berkembang dan berbahaya. Ketergantungan pada teknologi dan dunia digital telah membuat meningkatnya penggunaan transaksi dengan kartu kredit.Sehingga kartu kredit menjadi salah satu metode pembayaran yang paling umum baik saat bertransaksi secara online maupun offline sekarang ini. Namun hal tersebut jugalah yang membuat tingkat penipuan kartu kredit ini juga kian meningkat. Oleh sebab itu, banyak lembaga atau perusahaan mulai memfokuskan perhatiannya pada metodologi komputasi terkini untuk menangani masalah penipuan kartu kredit ini. Dengan membuat deteksi penipuan kartu kredit menggunakan metode machine learning, harapannnya dapat mengurangi terjadinya penipuan dengan kartu kredit ini.

Referensi: [Credit card fraud detection using machine learning techniques: A comparative analysis](https://ieeexplore.ieee.org/abstract/document/8123782)

## Business Understanding

### Problem Statements

- Penipuan kartu kredit di tengah perkembangan pembayaran digital yang sedang marak terjadi
- Melakukan perbandingan performa beberapa algoritma klasifikasi untuk menyelesaikan masalah pada data penipuan kartu kredit dengan distribusi miring

### Goals

Menjelaskan tujuan dari pernyataan masalah:

- Membuat deteksi penipuan kartu kredit guna mengurangi kemungkinan terjadinya penipuan kedepannya
- Mendapatkan algoritma klasifikasi yang paling tepat dan efisien untuk menyelesaikan permasalahan penipuan kartu kredit

### Solution statements

- Mengajukan dua algoritma yaitu Naive Bayes dan Logistic Regression, kemudian melakukan perbandingan untuk mencapai model yang diinginkan dan sesuai dengan kebutuhan
- Untuk evaluasi model, model akan diuji dan diukur dengan metrik evaluasi akurasi, precision, recall dan F1 score dengan bantuan library sklearn.

## Data Understanding

Data penipuan kartu kredit ini saya ambil dari situs kaggle. Untuk tautannya dapat diakses di [Credit Card Fraud](https://www.kaggle.com/datasets/dhanushnarayananr/credit-card-fraud/metadata). Selanjutnya, adapun penjelasan fitur sebagai berikut:

### Variabel-variabel pada Restaurant UCI dataset adalah sebagai berikut:

- distance_from_home : mepresentasikan jarak dari rumah ke tempat dimana transaksi dilakukan (0.004874 - 10632.723672)
- distance_from_last_transaction : mempresentasikan jarak dari tempat terakhir dilakukannya transaksi (0.000118 - 11851.104565)
- ratio_to_median_purchase_price : mempresentasikan rasio transaksi dengan median harga beli (0.004399 - 267.802942)
- repeat_retailer : apakah transaksi terjadi dari retail yang sama (0: tidak, 1: iya)
- used_chip : apakah transaksi melalui chip kartu kredit (0: tidak, 1: iya)
- used_pin_number : apakah transaksi tersebut menggunakan nomor pin (0: tidak, 1: iya)
- online_order : apakah transaksi tersebut merupakan pesanan online (0: tidak, 1: iya)
- fraud : apakah transaksi tersebut termasuk penipuan (fraud) (0: tidak, 1: iya)

### Exploratory Data Analysis

#### Menangani Missing Values

Pada data tersebut, saya melakukan pengecekan nilai duplikasi, dan nilai null. Saya juga melakukan pengecekan nilai yang anomali pada data jenis boolean dan hasilnya aman atau sudah bersih. Namun, saya menemukan outlier atau anomali data pada data jenis numerikal kontinyu (distance_from_home, distance_from_last_transaction dan ratio_to_median_purchase_price). Disini saya tidak langsung menghilangkan outlier tersebut dikarenakan terjadi imbalance data pada fitur target yaitu fraud, sehingga apabila saya menghilangkan outlier maka data kelas minoritas fitur fraud akan terhapus semua.

#### Univariate Analysis

Dengan teknik univariate EDA, saya mendapatkan beberapa informasi, antara lain:

- Distribusi data numerikal kontinyu (distance_from_home, distance_from_last_transaction dan ratio_to_median_purchase_price) miring ke kanan atau right-skewed. Hal ini akan berimplikasi pada model sehingga lebih baik dilakukan standarisasi nantinya.
- Distribusi data target fraud tidak seimbang (imbalance data) sehingga ada baiknya dilakukan undersampling. Menurut laman yang saya baca di [Imbalanced Data](https://developers.google.com/machine-learning/data-prep/construct/sampling-splitting/imbalanced-data), derajat ketidakseimbangan kelas minoritasnya pada dataset saya berada pada "moderate" sehingga saya memutuskan untuk melakukan undersampling.

#### Undersampling

Saya akan melakukan "Random Under Sampling" dengan menghilangkan data pada kelas mayoritasnya sehingga distribusi datanya bisa lebih seimbang (Saya mempelajari beberapa teknik ini dari [Dealing with Imbalanced Datasets](https://www.kaggle.com/code/janiobachmann/credit-fraud-dealing-with-imbalanced-datasets/notebook#Test-Data-with-Logistic-Regression:)). Sebelum melakukan undersampling, saya memisahkan dengan dataset aslinya terlebih dahulu. Hal ini dilakukan karena nantinya pada tahap evaluasi model, saya akan menguji model dengan dataset aslinya bukan dataset yang dibuat dengan teknik undersampling. Tujuannya untuk menyesuaikan model dengan kerangka dataset sebelum dilakukan undersample atau oversample dan dapat mendeteksi pola lebih baik pada set pengujian dataset asli.

Setelah melakukan undersampling, distribusi data target fraud sudah seimbang, kemudian barulah saya melakukan pembersihan data lebih lanjut dengan menghilangkan outlier. Untuk proses menghilangkan outlier, saya hanya menghilangkan beberapa data dari kuantil terakhir. Hal ini dikarenakan data dari kuantil terakhir tersebut memiliki nilai yang sangat jauh dibanding kumpulan data lainnya.

#### Multivariate Analysis

Dengan teknik multivariate EDA, saya mendapatkan informasi bahwa fitur repeat_retailer memiliki korelasi sangat rendah dengan fitur target yaitu fraud. Sehingga, fitur repeat_retailer akan saya drop.

## Data Preparation

Pada bagian ini, saya melakukan dua tahap persiapan data, yaitu:

1. Pembagian dataset dengan fungsi train_test_split

Membagi dataset menjadi data latih dan data uji dengan proporsi pembagian 80:20. Tujuannya adalah agar tidak mengotori data uji dengan informasi yang didapat dari data latih. Namun disini terdapat sedikit perbedaan. Untuk data ujinya, saya akan menggunakan data uji dari dataset aslinya yaitu dataset sebelum dilakukan undersampling ataupun oversampling.

2. Standarisasi

Saya melakukan standarisasi pada fitur numerik kontinyu (distance_from_home, distance_from_last_transaction dan ratio_to_median_purchase_price) menggunakan teknik MinMaxScaler dari library sklearn. MinMaxScaler melakukan proses standarisasi fitur dengan mengubah nilai fitur ke rentang tertentu (misalnya diantara nol dan satu).

## Modeling

Pada bagian ini, saya mengembangkan model machine learning dengan dua algoritma, yaitu Naive Bayes dan Logistic Regression. Saya menggunakan dua algoritma tersebut tanpa melakukan improvement apapun. Berikut penjelasan dan hasil lebih lanjut tentang algoritma yang digunakan:

1. Naive Bayes

Kelebihan utama penggunanaan Naive Bayes dalam klasifikasi adalah pengklasifikasiannya bisa dipersonalisasi, yaitu disesuaikan dengan kebutuhan. Sedangkan kelemahannya adalah keberhasilannya sangat bergantung pada pengetahuan awal, sehingga semakin banyak celah yang bisa mengurangi efektivitasnya.

2. Logistic Regression (Model Terbaik)

Kelemahan dari Logistic Regression adalah rentan terhadap underfitting pada dataset yang kelasnya tidak seimbang. Namun kelemahannya ini cukup teratasi sehingga sebagai hasilnya dapat dilihat bahwa algoritma ini telah berhasil membuat model yang terbaik.

## Evaluation

Pada bagian evaluasi ini, saya menggunakan metrik akurasi, precision, recall dan F1 score dengan bantuan library sklearn. Saya juga menampilkan confusion matrix untuk mengetahui darimana hasil metrik akurasi precision recall dan F1 score didapatkan. Pada tahap modeling, didapatkan hasil bahwa algoritma Logistic Regression adalah model yang paling cocok dalam menyelesaikan permasalahan ini. Oleh sebab itu, saya akan membahas lebih lanjut hasil metrik evaluasi pada algortima Logistic Regression.

Confusion Matrix Logistic Regression

![Confusion Matrix Logistic Regression](/assets/images/cf_lr.png)

Penjelasan lebih lanjut:

- True Positif (TP) : kasus dimana transaksi diprediksi fraud (1.0) dan data sebenarnya fraud (1.0) -> 16667
- True Negatif (TN) : kasus dimana transaksi diprediksi tidak fraud (0.0) data sebenarnya tidak fraud (0.0) -> 170089
- False Positif (FP) : kasus dimana transaksi diprediksi fraud (1.0) namun data sebenarnya tidak fraud (0.0) -> 12468
- False Negatif (FN) : kasus dimana transaksi diprediksi tidak fraud (0.0) data sebenarnya fraud (1.0) -> 776

Pengukuran Performa:

- Akurasi = (TP + TN) / (TP+FP+FN+TN)

  Akurasi = (16667 + 170089) / (16667 + 170089 + 12468 + 776) = 0.93

- Presisi = (TP) / (TP+FP)

  Presisi pada kelas 1 = (16667) / 16667 + 12468) = 0.57

  Presisi pada kelas 0 = (170089) / (170089 + 776) = 0.995

- Recall = (TP) / (TP + FN)

  Recall pada kelas 1 = (16667) / (16667 + 776) = 0.9555

  Recall pada kelas 0 = (170089) / (170089 + 12468) = 0.9317

- F1 Score = 2 * (Recall*Presisi) / (Recall+Presisi)

  F1 Score pada kelas 1 = (0.9555 * 0.57) / (0.9555 + 0.57) = 0.72

  F1 Score pada kelas 0 = (0.9317 * 0.995) / (0.9317 * 0.995) = 0.96

Sesuai perhitungan yang telah dilakukan diatas, hasilnya kurang lebih sama dengan classification report dibawah ini.

Classification Report Logistic Regression

![Classification Report Logistic Regression](/assets/images/cr_lr.png)
