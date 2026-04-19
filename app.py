import streamlit as st
import google.generativeai as genai
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import base64
import asyncio
import edge_tts
import re

# =========================
# 1. CONFIGURATION
# =========================
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
model = genai.GenerativeModel("models/gemini-flash-latest")

st.set_page_config(
    page_title="SANAD AI Assistant",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# =========================
# 2. SESSION STATE
# =========================
if "page" not in st.session_state:
    st.session_state.page = "home"

if "chat_type" not in st.session_state:
    st.session_state.chat_type = None

# =========================
# 3. CORE FUNCTIONS (PDF & TTS)
# =========================
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text: text += page_text
    return text

def get_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def create_vector_store(text_chunks):
    embeddings = get_embeddings()
    db = FAISS.from_texts(text_chunks, embedding=embeddings)
    db.save_local("faiss_index")

def load_db():
    embeddings = get_embeddings()
    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

def text_to_audio(text):
    audio_file = "response.mp3"
    clean_text = re.sub(r"[.,:*()\-\n#]", " ", text)
    async def generate():
        communicate = edge_tts.Communicate(clean_text, voice="ar-EG-SalmaNeural")
        await communicate.save(audio_file)
    asyncio.run(generate())
    return audio_file

def autoplay_audio(file_path):
    with open(file_path, "rb") as f:
        audio_bytes = f.read()
        b64 = base64.b64encode(audio_bytes).decode()
    st.markdown(f'<audio autoplay><source src="data:audio/mp3;base64,{b64}" type="audio/mp3"></audio>', unsafe_allow_html=True)

# =========================
# 4. IMPROVED HOME PAGE UI
# =========================
def home_page():
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;700;900&display=swap');

            .stApp {
                background: radial-gradient(circle at 50% 0%, #1e293b 0%, #0f172a 100%);
                font-family: 'Cairo', sans-serif;
                direction: rtl;
            }

            .main-header {
                text-align: center;
                padding: 60px 0 20px 0;
            }
            
            .title-text {
                font-size: 4rem;
                font-weight: 900;
                background: linear-gradient(90deg, #10b981, #34d399, #10b981);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                margin-bottom: 10px;
                filter: drop-shadow(0 0 10px rgba(16, 185, 129, 0.2));
            }

            .tagline {
                color: #94a3b8;
                font-size: 1.3rem;
                margin-bottom: 40px;
            }

            /* Glassmorphism Card Wrapper */
            .card-style {
                background: rgba(30, 41, 59, 0.4);
                border: 1px solid rgba(255, 255, 255, 0.05);
                border-radius: 25px;
                padding: 40px 20px;
                text-align: center;
                backdrop-filter: blur(10px);
                transition: 0.4s ease;
                min-height: 380px;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
            }

            .card-style:hover {
                transform: translateY(-10px);
                border-color: #10b981;
                box-shadow: 0 20px 40px rgba(0,0,0,0.4);
            }

            .icon-circle {
                background: rgba(16, 185, 129, 0.1);
                width: 90px;
                height: 90px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 3.5rem;
                margin-bottom: 20px;
            }

            .card-title {
                font-size: 1.6rem;
                font-weight: 700;
                color: #ffffff;
                margin-bottom: 15px;
            }

            .card-desc {
                font-size: 1rem;
                color: #94a3b8;
                line-height: 1.6;
                margin-bottom: 25px;
            }

            /* Custom Button */
            .stButton > button {
                width: 100%;
                background: linear-gradient(90deg, #10b981, #059669) !important;
                color: white !important;
                border: none !important;
                padding: 12px 0 !important;
                border-radius: 12px !important;
                font-weight: 700 !important;
                transition: 0.3s !important;
            }
            
            .stButton > button:hover {
                transform: scale(1.03) !important;
                box-shadow: 0 8px 20px rgba(16, 185, 129, 0.3) !important;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="main-header">
            <h1 class="title-text">SANAD AI Assistant</h1>
            <p class="tagline">مساعدك الذكي المدعوم بالذكاء الاصطناعي في القطاع الزراعي</p>
        </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3, gap="medium")

    with col1:
        st.markdown('<div class="card-style"><div class="icon-circle">🌱</div><div class="card-title">تمويل المحاصيل</div><div class="card-desc">تحليل متقدم لفرص تمويل المحاصيل الزراعية وتقديم حلول مخصصة للمزارعين.</div>', unsafe_allow_html=True)
        if st.button("دخول القسم", key="btn_agri"):
            st.session_state.chat_type = "agriculture"
            st.session_state.page = "chat"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card-style"><div class="icon-circle">📊</div><div class="card-title">التمويل والقروض</div><div class="card-desc">استكشاف الخيارات الائتمانية والتمويلية المتاحة لدعم المشاريع والنمو المالي.</div>', unsafe_allow_html=True)
        if st.button("دخول القسم", key="btn_finance"):
            st.session_state.chat_type = "general"
            st.session_state.page = "chat"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="card-style"><div class="icon-circle">🐄</div><div class="card-title">الثروة الحيوانية</div><div class="card-desc">دعم شامل لمشاريع الإنتاج الحيواني والداجني عبر تحليلات دقيقة واستشارات فورية.</div>', unsafe_allow_html=True)
        if st.button("دخول القسم", key="btn_livestock"):
            st.session_state.chat_type = "agriculture"
            st.session_state.page = "chat"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# =========================
# 5. CHAT PAGE & SIDEBAR
# =========================
def sidebar():
    with st.sidebar:
        st.header("📂 Documents")
        pdf_docs = st.file_uploader("Upload PDF (Knowledge Base)", accept_multiple_files=True)
        if st.button("Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                    chunks = splitter.split_text(raw_text)
                    create_vector_store(chunks)
                st.success("Done! ✅")
            else:
                st.warning("Upload the file.")

def chat_page():
    st.markdown(f"<h1 style='text-align: left; color: #10b981;'>💬 {st.session_state.chat_type.upper()} Assistant</h1>", unsafe_allow_html=True)
    
    if st.button("⬅️ Back"):
        st.session_state.page = "home"
        st.rerun()

    st.divider()

    # Chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask SANAD..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    db = load_db()
                    docs = db.similarity_search(prompt)
                    context = "\n\n".join([d.page_content for d in docs])
                except:
                    context = "No files have been uploaded for this section."

                sys_msg = "أنت خبير ذكاء اصطناعي في مجال الزراعة والتمويل. أجب باللغة العربية بأسلوب مهني ومختصر."
                full_query = f"{sys_msg}\n\nالسياق: {context}\n\nالسؤال: {prompt}"
                
                response = model.generate_content(full_query)
                st.markdown(response.text)
                
                # TTS
                audio_path = text_to_audio(response.text)
                autoplay_audio(audio_path)
                
                st.session_state.messages.append({"role": "assistant", "content": response.text})

# =========================
# 6. ROUTING
# =========================
if st.session_state.page == "home":
    home_page()
elif st.session_state.page == "chat":
    sidebar()
    chat_page()
