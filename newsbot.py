import streamlit as st
import json
import os
import time
import pathlib
import numpy as np
from openai import OpenAI

# Set page config harus di awal
st.set_page_config(page_title="📰 Chatbot Interaktif Harian Kompas", layout="centered")

# Custom CSS untuk styling
st.markdown("""
    <style>
    /* Gradient Header */
    .gradient-header {
        background: linear-gradient(90deg, #1E3A8A 0%, #3B82F6 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Logo Animation */
    .logo-container {
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    /* Subtitle Styling */
    .custom-subtitle {
        color: #4B5563;
        font-size: 1.2rem;
        font-weight: 500;
        margin-top: 1rem;
        line-height: 1.5;
    }
    
    /* Chat Bubbles */
    .user-bubble {
        background-color: #E5E7EB;
        padding: 1rem;
        border-radius: 1rem 1rem 0 1rem;
        margin: 0.5rem 0;
        max-width: 80%;
        margin-left: auto;
    }
    
    .assistant-bubble {
        background-color: #3B82F6;
        color: white;
        padding: 1rem;
        border-radius: 1rem 1rem 1rem 0;
        margin: 0.5rem 0;
        max-width: 80%;
    }
    
    /* Typing Indicator */
    .typing-indicator {
        display: inline-block;
        padding: 0.5rem;
    }
    .typing-indicator span {
        display: inline-block;
        width: 8px;
        height: 8px;
        background-color: #3B82F6;
        border-radius: 50%;
        margin: 0 2px;
        animation: typing 1s infinite;
    }
    .typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
    .typing-indicator span:nth-child(3) { animation-delay: 0.4s; }
    @keyframes typing {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-5px); }
    }
    
    /* Send Button */
    .stButton button {
        background-color: #3B82F6;
        color: white;
        border-radius: 20px;
        padding: 0.5rem 1.5rem;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton button:hover {
        background-color: #2563EB;
        transform: translateY(-2px);
    }
    </style>
""", unsafe_allow_html=True)

# ── Configuration ───────────────────────────────────────────────────────────────
KB_BASE         = "news"                    # will look for news.json or news.txt
EMBEDDING_MODEL = "text-embedding-ada-002"
CHAT_MODEL      = "gpt-4.1"
BATCH_SIZE      = 10                        # embeddings batch size
TRUNCATE_CHARS  = 3000                      # max chars per record

# ── Load OpenAI API key ─────────────────────────────────────────────────────────
api_key = (
    st.secrets.get("openai", {}).get("api_key")
    or os.getenv("OPENAI_API_KEY")
)
if not api_key:
    st.error(
        "🚨 Missing OpenAI API key. Define under [openai] api_key in secrets.toml "
        "or set the OPENAI_API_KEY environment variable."
    )
    st.stop()
client = OpenAI(api_key=api_key)

# ── Embedding helper ────────────────────────────────────────────────────────────
@st.cache_resource
def compute_embeddings(texts: list[str]) -> list[list[float]]:
    all_embeds = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        try:
            resp = client.embeddings.create(input=batch, model=EMBEDDING_MODEL)
        except Exception as e:
            st.error(f"Embedding API error: {e}")
            return []
        all_embeds.extend(d.embedding for d in resp.data)
        time.sleep(0.2)
    return all_embeds

# ── Knowledge loader (JSON or TXT with paragraph splitting) ────────────────────
@st.cache_data
def load_knowledge(base_path: str = KB_BASE) -> list[dict]:
    json_path = f"{base_path}.json"
    txt_path  = f"{base_path}.txt"

    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, list):
            entries = raw
        elif isinstance(raw, dict):
            entries = (
                raw.get("Knowledge")
                or raw.get("data")
                or next((v for v in raw.values() if isinstance(v, list)), [])
            )
        else:
            entries = []
    elif os.path.exists(txt_path):
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read()
        # split into paragraphs on blank lines
        paras = [p.strip() for p in text.split("\n\n") if p.strip()]
        entries = [
            {"title": f"Paragraph {i+1}", "content": para}
            for i, para in enumerate(paras)
        ]
    else:
        st.error(f"No knowledge file found at {json_path} or {txt_path}")
        return []

    records = []
    for idx, rec in enumerate(entries):
        if isinstance(rec, dict):
            title   = rec.get("title") or rec.get("headline") or f"Entry {idx}"
            content = rec.get("content") or rec.get("body")     or json.dumps(rec, ensure_ascii=False)
        else:
            title, content = f"Entry {idx}", str(rec)
        records.append({"id": idx, "title": title, "content": content})
    return records

# ── Build vector store ──────────────────────────────────────────────────────────
@st.cache_resource
def build_vector_store(records: list[dict]):
    texts     = [r["content"][:TRUNCATE_CHARS] for r in records]
    embeddings = compute_embeddings(texts)
    if not embeddings:
        st.stop()
    norms     = [np.linalg.norm(e) for e in embeddings]
    return embeddings, norms

# ── Retriever via cosine similarity ─────────────────────────────────────────────
def retrieve(query: str, records, embeddings, norms, k: int = 3) -> list[dict]:
    q_emb  = compute_embeddings([query])[0]
    q_norm = np.linalg.norm(q_emb)
    scores = [
        np.dot(q_emb, e) / (q_norm * n + 1e-8)
        for e, n in zip(embeddings, norms)
    ]
    top_idxs = np.argsort(scores)[-k:][::-1]
    return [records[i] for i in top_idxs]

# Header dengan gradient
st.markdown('<div class="gradient-header">', unsafe_allow_html=True)

# Logo dengan animasi
st.markdown('<div class="logo-container">', unsafe_allow_html=True)
st.image("http://satrio.pw/wp-content/uploads/2025/05/Logo-Ikon-Kompas-85x85-1.png", width=150)
st.markdown('</div>', unsafe_allow_html=True)

# Title dan Subtitle
st.title("Chatbot Interaktif Harian Kompas")
st.markdown("""
    <div class="custom-subtitle">
    Temukan jawaban dari berita terkini Kompas secara interaktif. 
    Tanyakan apapun tentang berita, reportase, atau artikel terbaru kami.
    </div>
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

st.badge(":warning: Versi Pre-alpha", color="orange")

# Load and vectorize knowledge
records = load_knowledge()
if not records:
    st.stop()
embeddings, norms = build_vector_store(records)

# Render chat history dengan styling baru
if "history" not in st.session_state:
    st.session_state.history = []

for msg in st.session_state.history:
    if msg["role"] == "user":
        st.markdown(f'<div class="user-bubble">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="assistant-bubble">{msg["content"]}</div>', unsafe_allow_html=True)

# Input area dengan tombol kirim
col1, col2 = st.columns([4, 1])
with col1:
    user_input = st.text_input("Tanyakan sesuatu tentang berita kami…", key="chat_input")
with col2:
    send_button = st.button("Kirim", use_container_width=True)

if user_input and (send_button or st.session_state.chat_input):
    st.session_state.history.append({"role": "user", "content": user_input})
    
    # Tampilkan typing indicator
    typing_indicator = st.markdown("""
        <div class="typing-indicator">
            <span></span>
            <span></span>
            <span></span>
        </div>
    """, unsafe_allow_html=True)

    # Retrieve relevant paragraphs
    related = retrieve(user_input, records, embeddings, norms)
    context = "\n\n---\n\n".join(f"**{r['title']}**\n{r['content']}" for r in related)

    system_prompt = (
        "Anda adalah asisten yang hanya menjawab berdasarkan basis pengetahuan berikut:\n\n"
        f"{context}\n\n"
        "Jika pertanyaan di luar cakupan, jawab: "
        "'Maaf, saya tidak memiliki informasi tersebut.'"
        "Narasumber ekonom, pakar, ahli, pengajar, pengamat, praktisi juga disebut pakar."
        "di akhir respon, berikan judul berita yang memuat url atau link sumber respons."
    )

    with st.spinner("Memproses…"):
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_input},
            ],
        )
        answer = resp.choices[0].message.content.strip()

    # Hapus typing indicator
    typing_indicator.empty()
    
    st.session_state.history.append({"role": "assistant", "content": answer})
    st.markdown(f'<div class="assistant-bubble">{answer}</div>', unsafe_allow_html=True)
    
    # Reset input
    st.session_state.chat_input = ""
