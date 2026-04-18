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

st.set_page_config(page_title="SANAD AI", layout="wide")


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
# HOME PAGE (IMPROVED UI)
# =========================
def home_page():
    st.set_page_config(
    page_title="SANAD AI Assistant",
    page_icon="🌾",
    layout="wide",  # Use full screen width
    initial_sidebar_state="collapsed"  # Hide sidebar by default
)
    st.markdown("""
    <style>
        .browser-chrome {
            background-color: #3C4043;
            color: #E8EAED;
            padding: 10px 15px;
            font-family: sans-serif;
            display: flex;
            align-items: center;
            border-top-left-radius: 8px;
            border-top-right-radius: 8px;
        }
        .browser-tab {
            background-color: #202124;
            color: #E8EAED;
            padding: 8px 12px;
            border-radius: 8px 8px 0 0;
            margin-right: 10px;
            font-size: 14px;
            display: flex;
            align-items: center;
        }
        .browser-tab-icon {
            margin-right: 6px;
        }
        .browser-address-bar {
            background-color: #F1F3F4;
            color: #202124;
            padding: 6px 15px;
            border-radius: 20px;
            font-size: 14px;
            width: 100%;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .browser-address-text {
            color: #606468;
        }
        .browser-content {
            background-color: #202124; /* Main browser window background */
            padding: 20px;
            border-bottom-left-radius: 8px;
            border-bottom-right-radius: 8px;
            font-family: sans-serif;
        }
    </style>
""", unsafe_allow_html=True)

# Simulate the Chrome window top section
st.markdown("""
    <div class="browser-chrome">
        <div class="browser-tab">
            <span class="browser-tab-icon">🌾</span>
            <span>SANAD AI - Streamlit</span>
        </div>
        <div style="flex-grow: 1;"></div>
    </div>
    <div class="browser-chrome" style="background-color: #E8EAED; color: #606468; padding: 6px 15px;">
        <div class="browser-address-bar">
            <span>🌾 SANAD-AIAssistant</span>
            <span class="browser-address-text">| https://sanad-ai-assistant.streamlit.app</span>
        </div>
    </div>
""", unsafe_allow_html=True)

# 3. Inject CSS for main app styling and card appearance
# We use custom classes to control margins, borders, padding, and text centering.
st.markdown("""
    <style>
        .block-container {
            padding: 2rem 5rem 10rem; /* Add outer padding */
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
        }

        .subtitle {
            font-size: 1.8rem;
            font-weight: bold;
            text-align: center;
        }

        /* Card container styling */
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
        }

        .dept-description {
            font-size: 1.2rem;
            text-align: center;
            color: #BDC1C6;
        }
        
        /* Ensure columns are evenly spaced */
        .css-1r6slb0 {
            gap: 1.5rem;
        }
    </style>
""", unsafe_allow_html=True)

# 4. Main App Content - Inside simulated browser content
with st.container(css_class="browser-content"):
    
    # 5. Header Section
    with st.container(css_class="main-header"):
        st.markdown("""
            <div class="header-top">
                <span class="logo-emoji">🌾</span>
                <span class="header-title">SANAD AI Assistant</span>
            </div>
            <div class="subtitle">اختر القسم المناسب</div>
        """, unsafe_allow_html=True)

    # 6. Grid of Departments
    with st.container():
        # Create three columns
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

        # Department 2: Finance and Loans
        with col2:
            st.markdown("""
                <div class="department-container">
                    <div class="dept-icon">📈💰</div>
                    <div class="dept-title">قسم التمويل والقروض</div>
                    <div class="dept-description">استكشف خيارات القروض والتسهيلات الائتمانية.</div>
                </div>
            """, unsafe_allow_html=True)

        # Department 3: Livestock and Poultry
        with col3:
            st.markdown("""
                <div class="department-container">
                    <div class="dept-icon">🐄🐔</div>
                    <div class="dept-title">قسم الثروة الحيوانية والدواجن</div>
                    <div class="dept-description">دعم مخصص لمشاريع الإنتاج الحيواني والداجني.</div>
                </div>
            """, unsafe_allow_html=True)


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
    st.title(f"💬 {st.session_state.chat_type.upper()} CHATBOT")

    if st.button("⬅ Back"):
        st.session_state.page = "home"

    question = st.text_input("Ask your question:")

    if question:
        db = load_db()
        docs = db.similarity_search(question)

        context = "\n\n".join([d.page_content for d in docs])

        system_prompt = {
            "agriculture": "You are an agriculture expert AI assistant.",
            "data": "You are a data science expert AI assistant.",
            "general": "You are a helpful AI assistant."
        }

        prompt = f"""
{system_prompt[st.session_state.chat_type]}

Context:
{context}

Question:
{question}

Answer clearly and simply:
"""

        response = model.generate_content(prompt)

        st.success(response.text)

        # AUDIO OUTPUT
        audio_file = text_to_audio(response.text)
        autoplay_audio(audio_file)


# =========================
# ROUTING
# =========================
if st.session_state.page == "home":
    home_page()

elif st.session_state.page == "chat":
    sidebar()
    chat_page()
