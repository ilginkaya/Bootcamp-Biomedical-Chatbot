# 🔬 Biyomedikal RAG Bilgi Asistanı (Gemini & LangChain)

## 📌 1. Projenin Amacı

Bu proje, Akbank GenAI Bootcamp kapsamında geliştirilmiştir. Temel amaç, **Retrieval Augmented Generation (RAG)** temelli bir sohbet robotu geliştirerek bunu bir web arayüzü üzerinden sunmaktır.

* **Çözülen Problem:** Biyomedikal Mühendisliği alanındaki karmaşık, yapılandırılmamış teknik metinlerden (PDF/TXT/MD), hızlı, güvenilir ve **kaynak gösteren** bilgi çekimi sağlamaktır.
* **Teknolojiler:** Gemini LLM, LangChain, ChromaDB.

## 📚 2. Veri Seti Hazırlama 

* **Konu:** Biyomedikal Mühendisliği Temelleri, İmmünoloji, Tıbbi Görüntüleme, Biyoetik ve Cihaz Regülasyonları gibi 14 farklı konuyu kapsayan bilgi tabanı.
* **Hazırlık Metodolojisi:** Hazır bilgi kaynaklarından toplanan bilgiler, projenin amacına uygun olarak **özel hazırlanmış 14 adet TXT/MD dosyası** (`data_docs/` klasöründe) haline getirilerek yapılandırılmıştır.

## ⚙️ 3. Çözüm Mimarisi ve Kullanılan Yöntemler 

| Bileşen | Kullanılan Teknoloji | Amaç |
| :--- | :--- | :--- |
| **LLM (Generation Model)** | Gemini-2.5-flash | Nihai yanıtı üretir. |
| **Embedding Model** | Google `models/text-embedding-004` | Metin parçalarını vektörlere dönüştürür. |
| **Vektör Veritabanı** | ChromaDB | Vektörleri depolar ve alaka düzeyine göre çeker. |
| **RAG Framework** | LangChain | Retrieval (Çekme) ve Generation (Üretme) zincirini yönetir. |

**RAG Süreci:** Kullanıcı sorgusu alınır. ChromaDB'den en alakalı metin parçaları çekilir. Bu parçalar ve sorgu, Gemini'ye gönderilerek nihai, **bağlama dayalı** yanıt oluşturulur.



## 🚀 4. Kodun Çalışma Kılavuzu 

### Ön Koşullar
* Python 3.9+
* [GEMINI API Anahtarı](https://ai.google.dev/gemini-api/docs/api-key)
* Tüm bağımlılıklar `requirements.txt` dosyasında listelenmiştir.

### Kurulum Adımları (Streamlit Cloud Kullanımı)
1.  **Kodun Yüklenmesi:** Proje dosyaları (özellikle `app.py`, `requirements.txt` ve `data_docs` klasörü) GitHub'a yüklenmiştir.
2.  **API Anahtarı Ayarı:** Gemini API Anahtarı, Streamlit Cloud'un **Secrets** menüsüne (veya `.streamlit/secrets.toml` dosyasına) tanımlanmıştır.
3.  **Veritabanı Oluşturma:** Uygulama, `app.py` içerisinde `rag_store` klasörünü bulamazsa, ilk çalıştırmada `data_docs` klasöründen veritabanını **otomatik olarak oluşturacak** şekilde ayarlanmıştır.

   ### Kurulum Adımları
1.  **Reposu Klonlama:**
    ```bash
    git clone (https://github.com/ilginkaya/Bootcamp-Biomedical-Chatbot)
    cd biyomedikal-rag-chatbot
    ```
2.  **Sanal Ortam Kurulumu (Virtual Environment):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3.  **Bağımlılıkların Kurulumu:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **API Anahtarı Tanımlama:** Terminal'de oturum için API anahtarı ayarlanır:
    ```bash
    export GEMINI_API_KEY="Kendi Gemini API Anahtarınız"
    ```
5.  **Veritabanı Oluşturma:** Uygulama, `app.py` içerisinde `rag_store` klasörünü bulamazsa ilk çalıştırmada **otomatik olarak oluşturacak** şekilde ayarlanmıştır.



## ✅ 5. Web Arayüzü & Product Kılavuzu (Bootcamp Kriteri)

**Elde Edilen Sonuçlar Özeti:**

* **Başarı:** Proje, Gemini API, LangChain ve ChromaDB entegrasyonunu başarıyla göstererek RAG mimarisini hayata geçirmiştir.
* **Kabiliyet:** Chatbot, sadece yüklenen biyomedikal metinlerden bilgi çekerek doğru ve konuya özgü yanıtlar üretmektedir.
* **Yanıt Süresi (Örnek Latency Notu):** Streamlit Cloud üzerinde ilk çalıştırma ve indeksleme sonrası, tipik bir sorguya yanıt süresi ortalama **3-5 saniyedir** (Model: Gemini-2.5-flash).

### Test Senaryosu Örnekleri (Kabiliyet Kanıtı)

| Soru | İlgili Alan | Beklenen Yanıt Tipi |
| :--- | :--- | :--- |
| "Aksiyon potansiyelini başlatan temel fiziksel mekanizma nedir?" | Biyofizik | İyonların hücre zarı boyunca hareketini açıklayan yanıt. |
| "Biyomedikal araştırmalarda etik kurallardan biri olan Özerklik ne anlama gelir?" | Biyoetik | Hastanın karar verme hakkını açıklayan yanıt. |
| "Fransa'nın başkenti neresidir?" | RAG Sınırlandırma Testi | "Bu konuda elimde yeterli bilgi yok." (RAG izolasyonunun kanıtı). |

### Çalışma Akışı ve Görsel Kılavuz

**Çalışma Akışı:** Kullanıcı, arayüzde sorusunu sorar. Chatbot, otomatik olarak oluşturulan veritabanından bilgi çeker. Cevabın altında, bilginin hangi kaynaktan (hangi TXT/MD dosyasından) alındığı gösterilir.
**Ekran Görüntüsü:**

<img width="1470" height="797" alt="Ekran Resmi 2025-10-17 12 51 47" src="https://github.com/user-attachments/assets/800d7b8b-4eb1-4625-80d4-fbeaf12729b0" />



   

***

### 🔗 Deploy Linki
[Canlı Uygulama Linki](https://bootcamp-biomedical-chatbot-he2gvhhkpcdzlz4shr6xfn.streamlit.app/)

### Uygulamayı Çalıştırma (Web Arayüzü)
```bash
streamlit run app.py
