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
            if page_text:
                text += page_text
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

# 🔥 UPDATED TTS FUNCTION (PERFECT VERSION)
def text_to_audio(text):
    audio_file = "response.mp3"

    # 1. تنظيف النص من الرموز
    clean_text = re.sub(r"[^\w\s\u0600-\u06FF]", " ", text)
    clean_text = re.sub(r"\s+", " ", clean_text).strip()

    # 2. SSML لصوت طبيعي
    ssml_text = f"""
    <speak version="1.0" xml:lang="ar-EG">
        <voice name="ar-EG-SalmaNeural">
            <prosody rate="-8%" pitch="+2%">
                {clean_text}
            </prosody>
        </voice>
    </speak>
    """

    async def generate():
        communicate = edge_tts.Communicate(
            text=ssml_text,
            voice="ar-EG-SalmaNeural"
        )
        await communicate.save(audio_file)

    asyncio.run(generate())
    return audio_file

def autoplay_audio(file_path):
    with open(file_path, "rb") as f:
        audio_bytes = f.read()
        b64 = base64.b64encode(audio_bytes).decode()

    st.markdown(
        f"""
        <audio autoplay controls>
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
        """,
        unsafe_allow_html=True
    )

# =========================
# 4. HOME PAGE
# =========================
def home_page():
    st.title("🌾 SANAD AI Assistant")
    st.write("مساعدك الذكي في الزراعة والتمويل")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🌱 الزراعة"):
            st.session_state.chat_type = "agriculture"
            st.session_state.page = "chat"
            st.rerun()

    with col2:
        if st.button("📊 التمويل"):
            st.session_state.chat_type = "finance"
            st.session_state.page = "chat"
            st.rerun()

    with col3:
        if st.button("🐄 الإنتاج الحيواني"):
            st.session_state.chat_type = "livestock"
            st.session_state.page = "chat"
            st.rerun()

# =========================
# 5. SIDEBAR
# =========================
def sidebar():
    with st.sidebar:
        st.header("📂 Upload PDF")
        pdf_docs = st.file_uploader("Upload PDF", accept_multiple_files=True)

        if st.button("Process"):
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
                st.warning("Upload files first")

# =========================
# 6. CHAT PAGE
# =========================
def chat_page():
    st.title(f"💬 {st.session_state.chat_type} Assistant")

    if st.button("⬅️ Back"):
        st.session_state.page = "home"
        st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("اسأل سَنَد..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):

                try:
                    db = load_db()
                    docs = db.similarity_search(prompt)
                    context = "\n".join([d.page_content for d in docs])
                except:
                    context = "لا يوجد ملفات مرفوعة."

                # 🔥 تحسين البرومبت للهجة المصرية
                sys_msg = "اتكلم باللهجة المصرية بشكل بسيط وواضح ومفهوم."

                full_prompt = f"""
                {sys_msg}

                السياق:
                {context}

                السؤال:
                {prompt}
                """

                response = model.generate_content(full_prompt)

                st.markdown(response.text)

                # 🔊 تشغيل الصوت
                audio_path = text_to_audio(response.text)
                autoplay_audio(audio_path)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response.text
                })

# =========================
# 7. ROUTING
# =========================
if st.session_state.page == "home":
    home_page()
elif st.session_state.page == "chat":
    sidebar()
    chat_page()
