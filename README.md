# 🔬 Biyomedikal RAG Bilgi Asistanı (Gemini & LangChain)

## 📌 1. Projenin Amacı 

Bu proje, Akbank GenAI Bootcamp kapsamında geliştirilmiştir. Temel amaç, **Retrieval Augmented Generation (RAG)** temelli bir chatbot geliştirerek bunu bir web arayüzü üzerinden sunmaktır.

* **Çözülen Problem:** Biyomedikal Mühendisliği alanındaki karmaşık, yapılandırılmamış teknik metinlerden (PDF/TXT/MD), hızlı, güvenilir ve kaynak gösteren bilgi çekimi sağlamaktır.
* **Teknolojiler:** Gemini LLM, LangChain, ChromaDB.

## 📚 2. Veri Seti Hazırlama 

* **Konu:** Biyomedikal Mühendisliği Temelleri, İmmünoloji, Tıbbi Görüntüleme, Biyoetik ve Cihaz Regülasyonları gibi 14 farklı konuyu kapsayan bilgi tabanı.
* **Hazırlık Metodolojisi:** Hazır bilgi kaynaklarından toplanan bilgiler, projenin amacına uygun olarak **özel hazırlanmış 14 adet TXT/MD dosyası** (`data_docs/` klasöründe) haline getirilerek yapılandırılmıştır.

## 3. Çözüm Mimariniz ve Kullanılan Yöntemler (Bootcamp Kriteri)

| Bileşen | Kullanılan Teknoloji | Amaç |
| :--- | :--- | :--- |
| **LLM (Generation Model)** | Gemini-2.5-flash | Nihai yanıtı üretir. |
| **Embedding Model** | Google `models/text-embedding-004` | Metin parçalarını vektörlere dönüştürür. |
| **Vektör Veritabanı** | ChromaDB | Vektörleri depolar ve alaka düzeyine göre çeker. |
| **RAG Framework** | LangChain | Retrieval (Çekme) ve Generation (Üretme) zincirini yönetir. |

**RAG Süreci:** Kullanıcı sorgusu alınır. ChromaDB'den en alakalı metin parçaları çekilir. Bu parçalar ve sorgu, Gemini'ye gönderilerek nihai, bağlama dayalı yanıt oluşturulur.

## 4. Kodun Çalışma Kılavuzu 

### Ön Koşullar
* Python 3.10 (Streamlit Cloud'da uyumluluk için önerilir.)
* GEMINI API Anahtarı
* Tüm bağımlılıklar `requirements.txt` dosyasında listelenmiştir.

### Kurulum Adımları
1.  **Reposu Klonlama:**
    ```bash
    git clone https://github.com/ilginkaya/Bootcamp-Biomedical-Chatbot.git
    cd Bootcamp-Biomedical-Chatbot
    ```
2.  **Sanal Ortam Kurulumu:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```
3.  **Bağımlılıkların Kurulumu:**
    ```bash
    pip3 install -r requirements.txt
    ```
4.  **API Anahtarı Tanımlama:**
    ```bash
    export GEMINI_API_KEY="[Kendi API Anahtarınız]"
    ```
5.  **Uygulamayı Çalıştırma:**
    ```bash
    python3 app.py
    ```
Local URL'niz ile sayfaya ulaşabilrsiniz.

## 5. Web Arayüzü & Product Kılavuzu 

**Elde Edilen Sonuçlar Özeti:**

* **Başarı:** Proje, Gemini API, LangChain ve ChromaDB entegrasyonunu başarıyla göstererek RAG mimarisini hayata geçirmiştir.
* **Kabiliyet:** Chatbot, sadece yüklenen biyomedikal metinlerden bilgi çekerek doğru ve konuya özgü yanıtlar üretmektedir.

### Çalışma Akışı ve Görsel Kılavuz
Kullanıcı, arayüzde sorusunu sorar. Chatbot, otomatik olarak oluşturulan veritabanından bilgi çeker. Cevabın altında, bilginin hangi kaynaktan (hangi TXT/MD dosyasından) alındığı gösterilir.

**Ekran Görüntüsü (Çalışma Örneği):**
<img width="1098" height="688" alt="Ekran Resmi 2025-10-19 22 09 13" src="https://github.com/user-attachments/assets/84f9f77a-bd65-4fd0-b89e-034cb445d56f" />


### Test Senaryosu Örnekleri
| Soru | İlgili Alan | Beklenen Yanıt Tipi |
| :--- | :--- | :--- |
| "Aksiyon potansiyelini başlatan temel fiziksel mekanizma nedir?" | Biyofizik | İyonların hücre zarı boyunca hareketini açıklayan yanıt. |
| "Biyomedikal araştırmalarda etik kurallardan biri olan Özerklik ne anlama gelir?" | Biyoetik | Hastanın karar verme hakkını açıklayan yanıt. |
| "Fransa'nın başkenti neresidir?" | RAG Sınırlandırma Testi | "Bu konuda elimde yeterli bilgi yok." (RAG izolasyonunun kanıtı). |

***

### 🔗 Deploy Linki
[Canlı Uygulama Linki](https://fcaf7df0b59027a46c.gradio.live)
Running on local URL:  (http://127.0.0.1:7860)
