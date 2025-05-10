# 📰 Streamlit Interactive News Chatbot

Repositori ini berisi aplikasi **Streamlit** sederhana yang mem‑bundle model GPT OpenAI, vektor embedding, dan *knowledge base* (KB) statis—berupa **JSON** *atau* **TXT**—untuk menjawab pertanyaan pembaca *Kompas* tentang berita/reportase secara interaktif. Jawaban dari chatbot akan lebih akurat apabila menggunakan KB berbasis JSON.

Aplikasi:

1. **Memuat KB lokal**

   * `news.json`  *atau*  `news.txt` di root project.
   * Jika `news.txt`, program otomatis membagi teks menjadi paragraf‑paragraf (dipisah dua baris kosong) dan memperlakukan setiap paragraf sebagai dokumen terpisah untuk pencarian.

2. **Meng‑embedding** setiap dokumen dengan model `text-embedding-ada-002` dan menyimpan vektor di memori (dengan *caching* Streamlit).

3. **Mencari** paragraf paling relevan (cosine‑similarity) untuk tiap pertanyaan pengguna.

4. **Membangun prompt** yang *hanya* berisi paragraf terpilih lalu memanggil `gpt-4.1` agar jawaban tidak keluar dari konteks KB (jika di luar cakupan → bot menjawab minta maaf).

---

## Struktur Berkas

```
.
├─ newsbot.py          # Skrip Streamlit (lihat contoh di README)
├─ requirements.txt    # Dependensi pip
├─ news.json           # KB terstruktur
└─ news.txt            # (opsional) KB tak terstruktur
```

> **Catatan**: Aplikasi akan mencari `news.json` lebih dulu.
> Jika tidak ada, ia akan memakai `news.txt`.

---


## Konfigurasi API Key

1. Ketika akan mendeploy aplikasi di Streamlit, klik ***Advanced Settings***.
2. Di bagian Secrets, isi seperti contoh berikut:

```toml
[openai]
api_key = "ISI_DENGAN_API_KEY_OPENAI_ANDA"
```
3. Jika terlupa mengisikan ketika mendeploy, API Key masih bisa dimasukkan ketika aplikasi sudah terlanjur terdeployed.
4. Klik ***Manage app*** di pojok kanan bawah aplikasi Streamlit yang sudah terbuka.
5. Lalu, akan muncul sidebar di sebelah kanan dan klik menu titik tiga.
6. Klik ***Settings***.
7. Lalu pilih tab **Secrets**. Dan isikan kode di atas, lalu **Save changes**.
---

## Menyiapkan Knowledge Base

### Menyiapkan file JSON

Susun bahan knowledge via Microsoft Excel atau aplikasi spreadsheet lainnya dengan struktur tabel seperti:

| url      | published_at | title  | full_text
| ----------- | ----------- | ----------- | -----------
| {isi link}  | {yyyy-mm-dd}       | Judul | Isi berita
| https://www.kompas.id/artikel/perlambatan-ekonomi-etc   | 2025-05-06 | Ekonomi Melambat, Pengangguran Meningkat | JAKARTA, KOMPAS — Adanya stagnasi pada pertumbuhan konsumsi rumah tangga membuat....        |

lalu save sebagai .xlsx (seperti normal atau default Excel)

* Kunci luar bebas (`Knowledge`, `data`, dsb) asalkan memuat **daftar** objek.
* Di tiap objek, ubi berubah‐ubah: gantilah pemetaan `title` / `content` di kode bila memakai nama lain (mis. `headline`, `body`).


## Menjalankan Aplikasi

```bash
streamlit run newsbot.py
```

* Buka link lokal yang muncul (biasanya [http://localhost:8501](http://localhost:8501)).
* Ketik pertanyaan di kolom chat; bot akan menampilkan jawaban berdasarkan KB.

---

## Penyesuaian Lebih Lanjut

| Bagian Kode                      | Fungsi                              | Cara Modifikasi                                      |
| -------------------------------- | ----------------------------------- | ---------------------------------------------------- |
| `KB_BASE`                        | Nama file KB tanpa ekstensi         | Ganti `news` ke nama lain                            |
| `EMBEDDING_MODEL` / `CHAT_MODEL` | Ganti model OpenAI                  | Edit string model                                    |
| `paragraph split`                | Logika pemenggalan TXT              | Ubah `text.split("\n\n")` (mis. pakai regex heading) |
| `TRUNCATE_CHARS`                 | Batas maksimum karakter yg di-embed | Naikkan/turunkan sesuai biaya & batas token          |

---

## Lisensi

MIT.
Silakan fork & modifikasi sesuai kebutuhan redaksi Kompas, tetap perhatikan syarat penggunaan OpenAI API.
