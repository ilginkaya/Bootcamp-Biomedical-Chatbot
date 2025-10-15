import streamlit as st
import os
from pathlib import Path

# LangChain ve Gemini Kütüphaneleri
from langchain_google_genai import ChatGoogleGenerativeAI 
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import Chroma 

# --- Sabitler ve Ayarlar ---
LLM_MODEL = "gemini-2.5-flash"
EMBEDDING_MODEL = "models/text-embedding-004" 
DB_PATH = "rag_store"

# --- RAG Fonksiyonları ---
@st.cache_resource
def load_rag_chain():
    """RAG zincirini ve LLM'i yükler."""

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("❌ HATA: GEMINI_API_KEY ortam değişkeni bulunamadı. Lütfen Terminal'de ayarlayın.")
        return None 

    # 1. Embedding Fonksiyonu
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=api_key)

    # 2. Veritabanının varlığını kontrol et ve yükle
    if not Path(DB_PATH).exists():
        st.warning("⚠️ RAG veritabanı bulunamadı. Lütfen 'python3 ingest.py' komutunu çalıştırın!")
        return None

    # Vektör Veritabanını yükle (Sistemin kalbi)
    vector_store = Chroma(
        persist_directory=DB_PATH, 
        embedding_function=embeddings
    )

    # 3. RAG Zincirini Kurma (Retriever + Prompt + LLM)
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL, temperature=0.2, google_api_key=api_key)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    prompt_template = """
    Sen bir Biyomedikal Bilgi Asistanısın. Aşağıdaki biyomedikal metinleri (Context) kullanarak, kullanıcıya Türkçe ve net bir şekilde yanıt ver. 
    Yanıtların teknik, kısa ve direkt olmalıdır. Bağlamda bulamadığın sorulara "Bu konuda elimde yeterli bilgi yok." diye yanıt ver.

    Context: {context}
    Soru: {question}
    Yanıt:
    """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    return qa_chain

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
    qa_chain = load_rag_chain()
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
                result = qa_chain({"query": prompt})
                response = result['result']

                # Kaynak dokümanları ekle
                sources = "\n\n### 📄 Kaynak Dokümanlar:\n"
                for i, doc in enumerate(result['source_documents']):
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