import os
from pathlib import Path
import gradio as gr # Yeni Gradio kütüphanesi
from typing import List

# LangChain ve Gemini Kütüphaneleri (Tüm uyumlu importlar)
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Sabitler ve Ayarlar ---
LLM_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "models/text-embedding-004"
DB_PATH = "rag_store"
DOCS_PATH = "data_docs"

# API Anahtarını al (Lokalde ve Cloud'da çalışır)
API_KEY = os.getenv("GEMINI_API_KEY")

# --- RAG Zinciri Kurulumu (load_rag_chain) ---
# Gradio'da Streamlit'in @st.cache_resource yerine normal bir fonskiyon kullanıyoruz
def load_rag_chain():
    """RAG zincirini, LLM'i yükler ve veritabanını kontrol/oluşturur."""

    if not API_KEY:
        # Gradio'da hata mesajı yerine None döndürüyoruz
        print("HATA: GEMINI_API_KEY ortam değişkeni ayarlanmadı.")
        return None, None
    
    # 1. Embedding Fonksiyonu
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=API_KEY)

    # 2. Veritabanının varlığını kontrol et ve oluştur (Yalnızca yoksa)
    if not Path(DB_PATH).exists():
        try:
            # Dokümanları yükleme
            loader = DirectoryLoader(
                DOCS_PATH, glob="**/*.txt", loader_kwargs={'encoding': 'utf-8', 'errors': 'ignore'}
            )
            docs = loader.load()
            
            # Parçalara ayırma (Chunking)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(docs)
            
            if not chunks:
                print("HATA: Otomatik indeksleme başarısız oldu.")
                return None, None

            vector_store = Chroma.from_documents(
                documents=chunks, embedding=embeddings, persist_directory=DB_PATH
            )
            print("Veritabanı başarıyla oluşturuldu.")
            
        except Exception as e:
            print(f"FATAL HATA: Otomatik indeksleme sırasında beklenmeyen hata oluştu: {e}")
            return None, None
        
    else:
        # Veritabanı varsa, sadece yükle
        vector_store = Chroma(
            persist_directory=DB_PATH, embedding_function=embeddings
        )

    # 3. RAG Zincirini Kurma
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.2, google_api_key=API_KEY)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    prompt_template = """
    Sen bir Biyomedikal Bilgi Asistanısın. Aşağıdaki biyomedikal metinleri (Context) kullanarak, kullanıcıya Türkçe ve net bir şekilde yanıt ver. 
    Yanıtların teknik, kısa ve direkt olmalıdır. Bağlamda bulamadığın sorulara 'Bu konuda elimde yeterli bilgi yok.' diye yanıt ver.

    Context: {context}
    Soru: {question}
    Yanıt:
    """
    PROMPT = PromptTemplate.from_template(prompt_template)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain, retriever 

# RAG zincirini bir kez başlat
QA_CHAIN, RETRIEVER = load_rag_chain()

# --- Gradio Arayüz Fonksiyonu ---
def predict(query):
    if not QA_CHAIN:
        return "API Anahtarı bulunamadığı için chatbot şu anda çalışmıyor."
    
    try:
        # RAG zincirini çalıştırma
        result = QA_CHAIN({"query": query})
        response = result['result']
        docs = result['source_documents']

        # Kaynak dokümanları formatlama
        sources = "\n\n### 📄 Kaynaklar:\n"
        for i, doc in enumerate(docs):
            file_name = doc.metadata.get('source', 'Bilinmeyen Kaynak')
            sources += f"- **{file_name}**:\n > {doc.page_content[:100]}...\n" 
        
        return response + sources

    except Exception as e:
        return f"Yanıt oluşturulamadı. Lütfen API anahtarınızı kontrol edin. Hata: {e}"

# --- Gradio Arayüz Tanımı ---
if __name__ == "__main__":
    if QA_CHAIN:
        # Gradio arayüzünü tanımla
        iface = gr.Interface(
            fn=predict,
            inputs=gr.Textbox(lines=2, placeholder="Biyomedikal sorunuzu giriniz...", label="Sorunuzu Buraya Yazınız"), 
            outputs="markdown",
            title="🔬 Biyomedikal RAG Bilgi Asistanı",
            description="Gemini, LangChain ve 14 biyomedikal bilgi dosyası ile güçlendirilmiştir."
        )
        iface.launch(share=True) # Public link oluşturmak için share=True
    else:
        print("\nChatbot başlatılamadı. Lütfen API anahtarı ayarınızı kontrol edin.")