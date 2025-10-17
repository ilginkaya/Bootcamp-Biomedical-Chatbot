# 🔬 Biyomedikal RAG Bilgi Asistanı (Gemini & LangChain)

## 📌 1. Projenin Amacı (Bootcamp Kriteri)

Bu proje, Akbank GenAI Bootcamp kapsamında geliştirilmiştir. Temel amaç, **Retrieval Augmented Generation (RAG)** temelli bir sohbet robotu geliştirerek bunu bir web arayüzü üzerinden sunmaktır.

* **Çözülen Problem:** Biyomedikal Mühendisliği alanındaki karmaşık, yapılandırılmamış teknik metinlerden (PDF/TXT/MD), hızlı, güvenilir ve **kaynak gösteren** bilgi çekimi sağlamaktır.
* **Teknolojiler:** Gemini LLM, LangChain, ChromaDB.

## 📚 2. Veri Seti Hazırlama (Bootcamp Kriteri)

* **Konu:** Biyomedikal Mühendisliği Temelleri, İmmünoloji, Tıbbi Görüntüleme, Biyoetik ve Cihaz Regülasyonları gibi 14 farklı konuyu kapsayan bilgi tabanı.
* **Hazırlık Metodolojisi:** Hazır bilgi kaynaklarından toplanan bilgiler, projenin amacına uygun olarak **özel hazırlanmış 14 adet TXT/MD dosyası** (`data_docs/` klasöründe) haline getirilerek yapılandırılmıştır.

## ⚙️ 4. Çözüm Mimariniz ve Kullanılan Yöntemler (Bootcamp Kriteri)

| Bileşen | Kullanılan Teknoloji | Amaç |
| :--- | :--- | :--- |
| **LLM (Generation Model)** | Gemini-2.5-flash | Nihai yanıtı üretir. |
| **Embedding Model** | Google `models/text-embedding-004` | Metin parçalarını vektörlere dönüştürür. |
| **Vektör Veritabanı** | ChromaDB | Vektörleri depolar ve alaka düzeyine göre çeker. |
| **RAG Framework** | LangChain | Retrieval (Çekme) ve Generation (Üretme) zincirini yönetir. |

**RAG Süreci:** Kullanıcı sorgusu alınır. ChromaDB'den en alakalı metin parçaları çekilir. Bu parçalar ve sorgu, Gemini'ye gönderilerek nihai, **bağlama dayalı** yanıt oluşturulur.

## 🚀 4. Kodunuzun Çalışma Kılavuzu (Bootcamp Kriteri)

### Ön Koşullar
* Python 3.9+
* [GEMINI API Anahtarı](https://ai.google.dev/gemini-api/docs/api-key)
* Tüm bağımlılıklar `requirements.txt` dosyasında listelenmiştir.

### Kurulum Adımları (Streamlit Cloud Kullanımı)
1.  **Kodun Yüklenmesi:** Proje dosyaları (özellikle `app.py`, `requirements.txt` ve `data_docs` klasörü) GitHub'a yüklenmiştir.
2.  **API Anahtarı Ayarı:** Gemini API Anahtarı, Streamlit Cloud'un **Secrets** menüsüne (veya `.streamlit/secrets.toml` dosyasına) tanımlanmıştır.
3.  **Veritabanı Oluşturma:** Uygulama, `app.py` içerisinde `rag_store` klasörünü bulamazsa, ilk çalıştırmada `data_docs` klasöründen veritabanını **otomatik olarak oluşturacak** şekilde ayarlanmıştır.

### Deploy Linki
https://bootcamp-biomedical-chatbot-he2gvhhkpcdzlz4shr6xfn.streamlit.app/ 

### Lokal Çalıştırma Komutu
```bash
streamlit run app.py
