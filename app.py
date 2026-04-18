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
# CONFIG
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
# SESSION STATE
# =========================
if "page" not in st.session_state:
    st.session_state.page = "home"

if "chat_type" not in st.session_state:
    st.session_state.chat_type = None

# =========================
# PDF READER
# =========================
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text

# =========================
# EMBEDDINGS
# =========================
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

# =========================
# FAISS
# =========================
def create_vector_store(text_chunks):
    embeddings = get_embeddings()
    db = FAISS.from_texts(text_chunks, embedding=embeddings)
    db.save_local("faiss_index")

def load_db():
    embeddings = get_embeddings()
    return FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

# =========================
# CLEAN TEXT FOR TTS
# =========================
def clean_text_for_tts(text):
    text = re.sub(r"[.,:*()\-\n#]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# =========================
# TEXT → AUDIO (EGYPTIAN VOICE)
# =========================
def text_to_audio(text):
    audio_file = "response.mp3"
    clean_text = clean_text_for_tts(text)

    async def generate():
        communicate = edge_tts.Communicate(
            clean_text,
            voice="ar-EG-SalmaNeural"
        )
        await communicate.save(audio_file)

    asyncio.run(generate())
    return audio_file

def autoplay_audio(file_path):
    with open(file_path, "rb") as f:
        audio_bytes = f.read()
        b64 = base64.b64encode(audio_bytes).decode()

    st.markdown(f"""
        <audio autoplay controls>
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
    """, unsafe_allow_html=True)

# =========================
# HOME PAGE (UPDATED WITH CLICKABLE CARDS)
# =========================
def home_page():
    # Custom CSS for the home page
    st.markdown("""
        <style>
            /* Main container styling */
            .block-container {
                padding: 2rem 5rem 10rem;
            }
            
            .main-container {
                background-color: #202124;
                color: #E8EAED;
                font-family: sans-serif;
            }

            .main-header {
                display: flex;
                flex-direction: column;
                align-items: center;
                margin-bottom: 3rem;
            }

            .header-top {
                display: flex;
                align-items: center;
                gap: 15px;
                margin-bottom: 0.5rem;
            }

            .logo-emoji {
                font-size: 5rem;
            }

            .header-title {
                font-size: 3rem;
                font-weight: bold;
                background: linear-gradient(90deg, #10b981, #34d399, #10b981);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }

            .subtitle {
                font-size: 1.8rem;
                font-weight: bold;
                text-align: center;
                color: #BDC1C6;
            }

            /* Card container styling - made clickable */
            .department-container {
                border: 2px solid #3C4043;
                border-radius: 12px;
                padding: 30px;
                margin: 10px;
                text-align: center;
                box-shadow: 2px 2px 10px rgba(0,0,0,0.3);
                display: flex;
                flex-direction: column;
                align-items: center;
                transition: all 0.3s ease;
                cursor: pointer;
                background: rgba(30, 41, 59, 0.4);
                backdrop-filter: blur(10px);
                height: 350px;
                justify-content: center;
            }

            .department-container:hover {
                transform: translateY(-8px);
                border-color: #10b981;
                box-shadow: 0 10px 30px rgba(16, 185, 129, 0.2);
                background: rgba(30, 41, 59, 0.6);
            }

            .dept-icon {
                font-size: 6rem;
                margin-bottom: 20px;
            }

            .dept-title {
                font-size: 1.8rem;
                font-weight: bold;
                margin-bottom: 15px;
                text-align: center;
                color: white;
            }

            .dept-description {
                font-size: 1.2rem;
                text-align: center;
                color: #94a3b8;
            }
            
            /* Hide button default styling */
            .stButton > button {
                background: transparent !important;
                border: none !important;
                color: transparent !important;
                box-shadow: none !important;
                padding: 0 !important;
                margin: 0 !important;
                height: auto !important;
                width: 100% !important;
            }
            
            .stButton > button:hover {
                background: transparent !important;
                border: none !important;
                box-shadow: none !important;
            }
            
            /* Column spacing */
            .css-1r6slb0 {
                gap: 1.5rem;
            }
            
            [data-testid="column"] {
                position: relative;
            }
        </style>
    """, unsafe_allow_html=True)

    # Header Section
    st.markdown("""
        <div class="main-header">
            <div class="header-top">
                <span class="logo-emoji">🌾</span>
                <span class="header-title">SANAD AI Assistant</span>
            </div>
            <div class="subtitle">اختر القسم المناسب</div>
        </div>
    """, unsafe_allow_html=True)

    # Grid of Departments with clickable cards
    col1, col2, col3 = st.columns(3)

    # Department 1: Crop Financing
    with col1:
        st.markdown("""
            <div class="department-container">
                <div class="dept-icon">🌱</div>
                <div class="dept-title">قسم تمويل المحاصيل الزراعية</div>
                <div class="dept-description">تقديم حلول تمويلية ذكية للمزارع.</div>
            </div>
        """, unsafe_allow_html=True)
        if st.button("", key="crop_btn", use_container_width=True):
            st.session_state.chat_type = "agriculture"
            st.session_state.page = "chat"
            st.rerun()

    # Department 2: Finance and Loans
    with col2:
        st.markdown("""
            <div class="department-container">
                <div class="dept-icon">📈💰</div>
                <div class="dept-title">قسم التمويل والقروض</div>
                <div class="dept-description">استكشف خيارات القروض والتسهيلات الائتمانية.</div>
            </div>
        """, unsafe_allow_html=True)
        if st.button("", key="finance_btn", use_container_width=True):
            st.session_state.chat_type = "general"
            st.session_state.page = "chat"
            st.rerun()

    # Department 3: Livestock and Poultry
    with col3:
        st.markdown("""
            <div class="department-container">
                <div class="dept-icon">🐄🐔</div>
                <div class="dept-title">قسم الثروة الحيوانية والدواجن</div>
                <div class="dept-description">دعم مخصص لمشاريع الإنتاج الحيواني والداجني.</div>
            </div>
        """, unsafe_allow_html=True)
        if st.button("", key="livestock_btn", use_container_width=True):
            st.session_state.chat_type = "agriculture"
            st.session_state.page = "chat"
            st.rerun()

# =========================
# SIDEBAR PDF
# =========================
def sidebar():
    with st.sidebar:
        st.header("📂 Upload PDFs")

        pdf_docs = st.file_uploader(
            "Upload PDFs",
            accept_multiple_files=True
        )

        if st.button("Process PDFs"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)

                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=100
                    )

                    chunks = splitter.split_text(raw_text)
                    create_vector_store(chunks)

                st.success("Done ✅")
            else:
                st.warning("Please upload files")

# =========================
# CHAT PAGE
# =========================
def chat_page():
    # Custom styling for chat page
    st.markdown("""
        <style>
            .stApp {
                background: radial-gradient(circle at 50% 0%, #1e293b 0%, #0f172a 100%);
            }
        </style>
    """, unsafe_allow_html=True)
    
    # Title based on chat type
    if st.session_state.chat_type == "agriculture":
        chat_title = "🌱 مساعد تمويل المحاصيل الزراعية"
    else:
        chat_title = "💬 مساعد التمويل والقروض"
    
    st.title(chat_title)

    # Back button
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("⬅️ العودة للرئيسية", use_container_width=True):
            st.session_state.page = "home"
            st.rerun()

    st.markdown("---")

    # Question input
    question = st.text_input("💬 اسأل سؤالك هنا:")

    if question:
        with st.spinner("جاري البحث والإجابة..."):
            try:
                # Load vector database if exists
                if os.path.exists("faiss_index"):
                    db = load_db()
                    docs = db.similarity_search(question)
                    context = "\n\n".join([d.page_content for d in docs])
                else:
                    context = ""

                # System prompts
                system_prompt = {
                    "agriculture": "أنت خبير في التمويل الزراعي والمحاصيل. قدم إجابات مفيدة باللغة العربية.",
                    "general": "أنت خبير في التمويل والقروض المصرفية. قدم إجابات مفيدة باللغة العربية."
                }

                prompt = f"""
{system_prompt.get(st.session_state.chat_type, system_prompt["general"])}

السياق المستندات (إن وجد):
{context}

سؤال المستخدم:
{question}

قدم إجابة واضحة ومفيدة باللغة العربية:
"""

                # Generate response
                response = model.generate_content(prompt)
                
                # Display response
                st.success("📝 الإجابة:")
                st.write(response.text)
                
                # Generate and play audio
                audio_file = text_to_audio(response.text)
                autoplay_audio(audio_file)
                
            except Exception as e:
                st.error(f"حدث خطأ: {str(e)}")

# =========================
# ROUTING
# =========================
if st.session_state.page == "home":
    home_page()
elif st.session_state.page == "chat":
    sidebar()
    chat_page()
