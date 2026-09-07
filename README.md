# Perbandingan Model Random Forest & SVM untuk Analisis Sentimen Ulasan Aplikasi Mobile Legends di Play Store

Proyek ini merupakan implementasi **Analisis Sentimen (Sentiment Analysis)** berbasis NLP terhadap ulasan pengguna aplikasi game **Mobile Legends: Bang Bang** di Google Play Store. Proyek ini dikerjakan sebagai bagian dari submission kelas **Dicoding Indonesia — Belajar Fundamental Deep Learning**.

Tujuan utama proyek adalah membandingkan performa beberapa kombinasi **model klasifikasi** dan **teknik ekstraksi fitur** dalam memprediksi sentimen (positif, negatif, netral) dari teks ulasan berbahasa Indonesia.

> **Catatan:** Nama repository menyebut LR (Logistic Regression), RF (Random Forest), dan SVM. Pada notebook versi saat ini (`Proyek_NLP_Sentiment_Analysis.ipynb`), model yang benar-benar diimplementasikan dan dibandingkan adalah **SVM** (dilatih menggunakan `SGDClassifier` dengan `loss="hinge"`) dan **Random Forest**. Bila Logistic Regression ingin ditambahkan, cukup mengikuti pola kode yang sama pada bagian *Modeling*.

## Daftar Isi

- [Ringkasan Proyek](#ringkasan-proyek)
- [Dataset](#dataset)
- [Alur Pengerjaan (Pipeline)](#alur-pengerjaan-pipeline)
- [Struktur Repository](#struktur-repository)
- [Hasil Perbandingan Model](#hasil-perbandingan-model)
- [Instalasi](#instalasi)
- [Cara Menjalankan](#cara-menjalankan)
- [Contoh Inference](#contoh-inference)
- [Model Tersimpan](#model-tersimpan)
- [Teknologi & Library](#teknologi--library)
- [Lisensi](#lisensi)

## Ringkasan Proyek

Ulasan aplikasi di Play Store berisi opini pengguna yang bisa dimanfaatkan untuk memahami persepsi publik terhadap sebuah produk. Pada proyek ini, ribuan ulasan aplikasi Mobile Legends dikumpulkan, dibersihkan, diberi label sentimen secara otomatis menggunakan pendekatan **lexicon-based**, lalu digunakan untuk melatih dan membandingkan beberapa model machine learning klasik dengan dua skema ekstraksi fitur (**TF-IDF** dan **Word2Vec**) serta dua skema pembagian data (**80:20** dan **70:30**).

## Dataset

- **Sumber:** Google Play Store, aplikasi `com.mobile.legends`, diambil menggunakan library `google-play-scraper`.
- **File:** `Ulasan_APK_Mobile_Legends.csv`
- **Jumlah data:** ± 305.000 baris ulasan.
- **Kolom utama:** `reviewId`, `userName`, `content` (teks ulasan), `score` (rating bintang), `thumbsUpCount`, `reviewCreatedVersion`, `at` (tanggal), `appVersion`.
- **Bahasa:** Indonesia.

Proses scraping dataset dapat dilihat pada notebook `Scrapping_dataset.ipynb`.

## Alur Pengerjaan (Pipeline)

Seluruh proses utama ada di `Proyek_NLP_Sentiment_Analysis.ipynb`, dengan tahapan sebagai berikut:

1. **Loading Dataset** — memuat data hasil scraping.
2. **Text Preprocessing**
   - Case folding (mengubah ke huruf kecil)
   - Pembersihan mention, hashtag, RT, URL, angka, dan tanda baca
   - Normalisasi kata slang (slang word)
   - Penghapusan stopword Bahasa Indonesia (NLTK + daftar tambahan kustom)
   - Tokenisasi (`word_tokenize`)
   - Stemming menggunakan **Sastrawi**
3. **Labeling (Lexicon-Based)** — sentimen ditentukan dengan menjumlahkan skor kata terhadap kamus leksikon positif dan negatif Bahasa Indonesia ([sumber leksikon](https://github.com/angelmetanosaa/dataset)), menghasilkan label `positive`, `negative`, atau `neutral`.
4. **Feature Extraction**
   - **TF-IDF** (`TfidfVectorizer`)
   - **Word2Vec** (`gensim`, vector_size=100, window=5)
5. **Modeling** — dilatih pada 4 kombinasi fitur × skema split:
   - SVM (TF-IDF, 80:20) & (70:30)
   - SVM (Word2Vec, 80:20) & (70:30)
   - Random Forest (TF-IDF, 80:20) & (70:30)
   - Random Forest (Word2Vec, 80:20) & (70:30)
6. **Evaluasi** — menggunakan `accuracy_score` dan `classification_report` (precision, recall, F1-score per kelas).
7. **Inference** — pengujian model terbaik (SVM + TF-IDF 80:20) pada kalimat ulasan baru.
8. **Dokumentasi Hasil** — menyimpan ringkasan akurasi seluruh model dan model terlatih ke disk.

## Struktur Repository

```
.
├── Scrapping_dataset.ipynb                     # Notebook scraping ulasan dari Play Store
├── Proyek_NLP_Sentiment_Analysis.ipynb         # Notebook utama: preprocessing, labeling, modeling, evaluasi
├── Ulasan_APK_Mobile_Legends.csv               # Dataset mentah hasil scraping
├── Hasil_Perbandingan_Model_Sentiment_Analysis.csv  # Ringkasan akurasi semua kombinasi model
├── hasil_inference_SVM_TF-IDF_80_20.csv        # Contoh hasil prediksi pada kalimat baru
├── requirements.txt                            # Daftar dependency Python
└── saved_models/                               # Model & vectorizer yang sudah dilatih (.pkl / .model)
```

## Hasil Perbandingan Model

| Model         | Fitur     | Skema Split | Akurasi Train (%) | Akurasi Test (%) |
|---------------|-----------|-------------|--------------------|--------------------|
| SVM           | TF-IDF    | 80:20       | 93,60              | **87,12**          |
| SVM           | TF-IDF    | 70:30       | 94,28              | 86,45              |
| SVM           | Word2Vec  | 80:20       | 76,69              | 73,17              |
| SVM           | Word2Vec  | 70:30       | 76,88              | 73,70              |
| Random Forest | TF-IDF    | 80:20       | 100,00             | 77,66              |
| Random Forest | TF-IDF    | 70:30       | 100,00             | 77,15              |
| Random Forest | Word2Vec  | 80:20       | 100,00             | 71,57              |
| Random Forest | Word2Vec  | 70:30       | 100,00             | 71,39              |

**Kesimpulan:**
- Kombinasi **SVM + TF-IDF (80:20)** memberikan akurasi pengujian terbaik, yaitu **87,12%**.
- **Random Forest** selalu mencapai akurasi training 100% (overfitting terhadap data latih), namun performa pada data uji lebih rendah dibanding SVM.
- **TF-IDF** secara konsisten mengungguli **Word2Vec** untuk kedua model pada kasus ini.

## Instalasi

Clone repository dan install seluruh dependency yang dibutuhkan:

```bash
git clone https://github.com/febryofibonacciamadeo/Perbandingan_Model_LR_RF_SVM_Analisis_Sentimen_Ulasan_APK_Mobile_Legends_Playstore.git
cd Perbandingan_Model_LR_RF_SVM_Analisis_Sentimen_Ulasan_APK_Mobile_Legends_Playstore
pip install -r requirements.txt
```

Beberapa resource NLTK juga perlu diunduh sekali di awal (sudah dipanggil otomatis di dalam notebook):

```python
import nltk
nltk.download("punkt_tab")
nltk.download("stopwords")
```

## Cara Menjalankan

1. **(Opsional) Scraping ulasan baru**
   Jalankan `Scrapping_dataset.ipynb` untuk mengambil ulasan terbaru dari Play Store (membutuhkan koneksi internet).

2. **Menjalankan pipeline analisis sentimen**
   Buka dan jalankan seluruh cell pada `Proyek_NLP_Sentiment_Analysis.ipynb` secara berurutan, mulai dari import library hingga penyimpanan model.

3. **Melihat hasil**
   Ringkasan akurasi seluruh model dapat dilihat pada `Hasil_Perbandingan_Model_Sentiment_Analysis.csv`.

## Contoh Inference

Model terbaik (SVM + TF-IDF 80:20) digunakan untuk memprediksi sentimen kalimat baru, contoh:

| Ulasan | Prediksi Sentimen |
|---|---|
| "Mobile Legends ini sekarang semakin bagus dan seru dimainkan" | negative |
| "Saya suka sekali bermain Mobile Legends" | postive |
| "Game sampah, matchmaking sangat buruk" | negative |

> Catatan: beberapa hasil prediksi di atas (diambil langsung dari `hasil_inference_SVM_TF-IDF_80_20.csv`) tampak kurang sesuai secara intuitif — ini konsisten dengan akurasi model sekitar 87% pada data uji, sehingga masih ada ruang perbaikan pada tahap labeling lexicon-based maupun pemilihan model/fitur.

Cara memuat ulang model untuk prediksi mandiri:

```python
import joblib

svm_model = joblib.load("saved_models/svm_tfidf_80_20.pkl")
tfidf_vectorizer = joblib.load("saved_models/tfidf_vectorizer.pkl")

teks_baru = ["Update terbaru bikin game jadi lebih lancar"]
# Lakukan preprocessing yang sama seperti pada notebook sebelum transform
vector = tfidf_vectorizer.transform(teks_baru)
prediksi = svm_model.predict(vector)
print(prediksi)
```

## Model Tersimpan

Folder `saved_models/` berisi seluruh model dan vectorizer hasil training:

- `svm_tfidf_80_20.pkl`, `svm_tfidf_70_30.pkl`
- `svm_word2vec_80_20.pkl`, `svm_word2vec_70_30.pkl`
- `rf_tfidf_80_20.pkl`, `rf_tfidf_70_30.pkl`
- `rf_word2vec_80_20.pkl`, `rf_word2vec_70_30.pkl`
- `tfidf_vectorizer.pkl`
- `word2vec.model`

## Teknologi & Library

- Python
- [google-play-scraper](https://pypi.org/project/google-play-scraper/) — scraping ulasan Play Store
- [NLTK](https://www.nltk.org/) — tokenisasi & stopword
- [Sastrawi](https://pypi.org/project/Sastrawi/) — stemming Bahasa Indonesia
- [scikit-learn](https://scikit-learn.org/) — TF-IDF, SVM (SGDClassifier), Random Forest, evaluasi
- [Gensim](https://radimrehurek.com/gensim/) — Word2Vec
- [pandas](https://pandas.pydata.org/) & [NumPy](https://numpy.org/) — manipulasi data
- [Matplotlib](https://matplotlib.org/) & [Seaborn](https://seaborn.pydata.org/) — visualisasi
- [Joblib](https://joblib.readthedocs.io/) — serialisasi model

## Lisensi

Belum ada berkas lisensi resmi pada repository ini. Silakan tambahkan berkas `LICENSE` (misalnya MIT License) apabila proyek ini ingin dibagikan secara terbuka dengan ketentuan penggunaan yang jelas.

---

*README ini dibuat berdasarkan isi aktual repository (notebook, dataset, dan hasil eksperimen) per September 2026.*
