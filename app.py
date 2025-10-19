import streamlit as st
import os
from pathlib import Path
from typing import List

# LangChain'in Hata Çözücü YENİ ve UYUMLU İMPORT YAPISI
from langchain_core.prompts import PromptTemplate
from langchain_core.vectorstores import VectorStoreRetriever # Retriever tipini tanımlamak için
from langchain_core.runnables import RunnablePassthrough, RunnableLambda 
from langchain_core.output_parsers import StrOutputParser # Çıktıyı düz metne çevirmek için

from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# --- Sabitler ve Ayarlar ---
LLM_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "models/text-embedding-004" 
DB_PATH = "rag_store"
DOCS_PATH = "data_docs" 

# API Anahtarını Streamlit secrets veya ortam değişkeninden çeker
if "gemini" in st.secrets:
    API_KEY = st.secrets["gemini"]["api_key"]
else:
    API_KEY = os.getenv("GEMINI_API_KEY")

# --- RAG Fonksiyonları ---
@st.cache_resource
def load_rag_chain():
    """RAG zincirini, LLM'i yükler ve veritabanını kontrol/oluşturur."""

    api_key = API_KEY
    
    if not api_key:
        st.error("❌ HATA: GEMINI_API_KEY bulunamadı. Lütfen Streamlit Cloud Secrets ayarlarını kontrol edin.")
        return None 
    
    # 1. Embedding Fonksiyonu
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=api_key)

    # 2. Veritabanının varlığını kontrol et ve yükle
    if not Path(DB_PATH).exists():
        # Veritabanı yoksa otomatik oluşturma (İlk çalıştırmada sadece bir kez olur)
        try:
            loader = DirectoryLoader(
                DOCS_PATH, glob="**/*.txt", loader_kwargs={'encoding': 'utf-8', 'errors': 'ignore'}
            )
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(docs)
            
            if not chunks:
                st.error("HATA: Otomatik indeksleme başarısız oldu. Dokümanlar boş veya okunamıyor.")
                return None

            vector_store = Chroma.from_documents(
                documents=chunks, embedding=embeddings, persist_directory=DB_PATH
            )
            
        except Exception as e:
            st.error(f"FATAL HATA: Otomatik indeksleme sırasında beklenmeyen hata oluştu: {e}")
            return None
        
    else:
        # Veritabanı varsa, sadece yükle
        vector_store = Chroma(
            persist_directory=DB_PATH, embedding_function=embeddings
        )

    # 3. Yeni Runnable RAG Zincirini Kurma (Hatasız Yöntem)
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.2, google_api_key=api_key)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # 3.1. Prompt Template
    prompt_template = """
    Sen bir Biyomedikal Bilgi Asistanısın. Aşağıdaki biyomedikal metinleri (Context) kullanarak, kullanıcıya Türkçe ve net bir şekilde yanıt ver. 
    Yanıtların teknik, kısa ve direkt olmalıdır. Bağlamda bulamadığın sorulara "Bu konuda elimde yeterli bilgi yok." diye yanıt ver.

    Context: {context}
    Soru: {question}
    Yanıt:
    """
    PROMPT = PromptTemplate.from_template(prompt_template)
    
    # 3.2. Zincir Tanımı: RetrievalQA'nın Runnable karşılığı
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()} 
        | PROMPT
        | llm
        | StrOutputParser()
    )
    
    # Not: Bu versiyon sadece yanıt döndürür, kaynakları çekmez. 
    # Kaynakları çekmek için ek RunnableLambda kodları gerekir ki, projeyi basit tutalım.
    
    return rag_chain, retriever # Hem zinciri hem de retriever'ı döndürüyoruz

# --- Streamlit Ana Uygulaması ---
def main():
    st.set_page_config(page_title="Biyomedikal RAG Asistanı", layout="wide")

    # Başlık ve Açıklama
    st.markdown(
        """
        <div style="text-align: center; background-color: #008080; padding: 10px; border-radius: 10px; color: white;">
            <h1>🔬 Biyomedikal RAG Bilgi Asistanı</h1>
            <p>Gemini AI, LangChain ve ChromaDB ile güçlendirilmiştir.</p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    st.markdown("---")

    # RAG sistemini başlat
    # Yeni yapı ile iki değer dönüyor: zincir ve retriever
    qa_chain, retriever = load_rag_chain()
    if not qa_chain:
        st.stop() 

    # Chat Mesajları
    if 'messages' not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Merhaba! Ben biyomedikal bilgi asistanınızım. Dokümanlarımdaki (Tıbbi terimler, cihazlar) konularla ilgili sorular sorun."}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"], avatar="🔬" if message["role"] == "assistant" else "👤"):
            st.markdown(message["content"])

    # Kullanıcıdan Girdi Alma
    if prompt := st.chat_input("Tıbbi bir terimi açıklayabilir misin?"):

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        with st.spinner("🧠 Gemini yanıt oluşturuyor..."):
            # RAG zincirini çalıştır
            try:
                # Runnable zinciri sadece query'yi alır
                response = qa_chain.invoke(prompt) 
                
                # Kaynak dokümanları çekme (Ayrı bir adım olarak)
                docs = retriever.get_relevant_documents(prompt)
                
                # Kaynak dokümanları ekle
                sources = "\n\n### 📄 Kaynak Dokümanlar:\n"
                for i, doc in enumerate(docs):
                    file_name = doc.metadata.get('source', 'Bilinmeyen Kaynak')
                    sources += f"- **{file_name}**:\n > {doc.page_content[:150]}...\n" 
                
                full_response = response + sources

                st.session_state.messages.append({"role": "assistant", "content": full_response})
                with st.chat_message("assistant", avatar="🔬"):
                    st.markdown(full_response)

            except Exception as e:
                st.error(f"Yanıt oluşturulamadı. Hata: {e}")
                st.session_state.messages.append({"role": "assistant", "content": "Üzgünüm, Gemini ile bağlantı kurulamadı veya bir hata oluştu."})


if __name__ == "__main__":
    main()