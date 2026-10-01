"""
MR Analyst - AI-powered data analyst platform
RAG (ChromaDB + SentenceTransformers) + Groq LLM + Streamlit

pip install streamlit groq python-dotenv pandas numpy pdfplumber python-docx chromadb sentence-transformers openpyxl
streamlit run mr_analyst_pro.py
"""

import hashlib
import io
import os
import re
from datetime import datetime

import chromadb
import numpy as np
import pandas as pd
import pdfplumber
import streamlit as st
from docx import Document
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

# =====================================================
# CONFIG  (edit this block)
# =====================================================
APP_NAME = "MR Analyst"
TAGLINE = "Upload your data. Ask anything. Get answers with sources."
MODEL = "openai/gpt-oss-20b"
DB_PATH = "./chroma_db"
COLLECTION = "mr_analyst"

PROFILE = {
    "name": "Your Name",
    "role": "Data Analyst  •  AI Enthusiast",
    "about": "I build data products that turn raw files into decisions.",
    "linkedin": "https://www.linkedin.com/in/your-handle/",
    "github": "https://github.com/your-handle/your-repo",
}

TABULAR = (".csv", ".xlsx", ".xls")
SUPPORTED = ["csv", "xlsx", "xls", "pdf", "docx", "txt"]

st.set_page_config(page_title=APP_NAME, page_icon="🤖", layout="wide")

# =====================================================
# STYLING
# =====================================================
st.markdown(
    """
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 1.5rem; max-width: 1200px;}

.hero {
    padding: 2rem 2.2rem; border-radius: 20px; margin-bottom: 1.2rem;
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #db2777 100%);
    color: white; box-shadow: 0 10px 30px rgba(79,70,229,.25);
}
.hero h1 {margin: 0; font-size: 2.4rem; font-weight: 800; letter-spacing: -.5px;}
.hero p  {margin: .4rem 0 0; font-size: 1.05rem; opacity: .92;}
.pill {
    display: inline-block; padding: .2rem .7rem; margin: .5rem .3rem 0 0;
    border-radius: 999px; background: rgba(255,255,255,.18);
    font-size: .78rem; backdrop-filter: blur(4px);
}
.card {
    padding: 1rem 1.2rem; border-radius: 14px; text-align: center;
    background: rgba(127,127,127,.08); border: 1px solid rgba(127,127,127,.18);
}
.card .num {font-size: 1.8rem; font-weight: 800; color: #7c3aed;}
.card .lbl {font-size: .8rem; opacity: .7; text-transform: uppercase; letter-spacing: .06em;}
.chip {
    display: inline-block; padding: .15rem .6rem; margin: .15rem;
    border-radius: 8px; font-size: .75rem;
    background: rgba(124,58,237,.14); border: 1px solid rgba(124,58,237,.3);
}
.src {
    padding: .7rem .9rem; border-left: 3px solid #7c3aed; border-radius: 6px;
    background: rgba(127,127,127,.07); margin-bottom: .5rem; font-size: .85rem;
}
div.stButton > button {border-radius: 10px; font-weight: 600;}
button[data-baseweb="tab"] {font-size: 1rem; font-weight: 600;}
</style>
""",
    unsafe_allow_html=True,
)


def metric_card(col, value, label):
    col.markdown(
        f'<div class="card"><div class="num">{value}</div><div class="lbl">{label}</div></div>',
        unsafe_allow_html=True,
    )


# =====================================================
# RESOURCES
# =====================================================
load_dotenv()


def get_api_key():
    key = os.getenv("GROQ_API_KEY")
    if key:
        return key
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return None


api_key = get_api_key()
if not api_key:
    st.error("GROQ_API_KEY not found. Add it to a .env file or Streamlit secrets.")
    st.stop()

client = Groq(api_key=api_key)


@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource
def load_collection():
    return chromadb.PersistentClient(path=DB_PATH).get_or_create_collection(COLLECTION)


embedder = load_embedder()
collection = load_collection()

# =====================================================
# SESSION STATE
# =====================================================
ss = st.session_state
ss.setdefault("files", {})  # name -> bytes
ss.setdefault("messages", [])
ss.setdefault("pending", None)

# =====================================================
# EXTRACTION
# =====================================================
@st.cache_data(show_spinner=False)
def load_df(name: str, data: bytes) -> pd.DataFrame:
    buf = io.BytesIO(data)
    return pd.read_csv(buf) if name.lower().endswith(".csv") else pd.read_excel(buf)


def profile_text(name: str, df: pd.DataFrame) -> str:
    cols = "\n".join(f"- {c} ({t})" for c, t in df.dtypes.astype(str).items())
    desc = df.describe(include="all").T.head(40).to_string()
    return (
        f"DATASET PROFILE for {name}\nRows: {len(df)}, Columns: {df.shape[1]}\n"
        f"Columns:\n{cols}\n\nSummary statistics:\n{desc}"
    )


def df_chunks(name: str, df: pd.DataFrame, rows: int = 25) -> list[str]:
    header = ",".join(map(str, df.columns))
    chunks = [profile_text(name, df)]
    for i in range(0, len(df), rows):
        body = df.iloc[i : i + rows].to_csv(index=False, header=False)
        chunks.append(f"Rows {i + 1}-{min(i + rows, len(df))} of {name}\n{header}\n{body}")
    return chunks


def text_chunks(text: str, size: int = 1000, overlap: int = 100) -> list[str]:
    text = text.strip()
    return [text[i : i + size] for i in range(0, len(text), size - overlap)] if text else []


def extract_chunks(name: str, data: bytes) -> list[str]:
    low = name.lower()
    if low.endswith(TABULAR):
        return df_chunks(name, load_df(name, data))
    if low.endswith(".pdf"):
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            text = "\n".join(p.extract_text() or "" for p in pdf.pages)
    elif low.endswith(".docx"):
        text = "\n".join(p.text for p in Document(io.BytesIO(data)).paragraphs)
    else:
        text = data.decode("utf-8", errors="ignore")
    return text_chunks(text)


# =====================================================
# INDEXING + RETRIEVAL
# =====================================================
def index_files(progress_cb=None) -> tuple[int, int]:
    new_chunks, skipped = 0, 0
    total = len(ss.files)
    for n, (name, data) in enumerate(ss.files.items(), 1):
        fhash = hashlib.md5(data).hexdigest()
        if collection.get(where={"file_hash": fhash}, limit=1)["ids"]:
            skipped += 1
        else:
            chunks = extract_chunks(name, data)
            if chunks:
                vectors = embedder.encode(chunks, show_progress_bar=False).tolist()
                for s in range(0, len(chunks), 2000):
                    sl = slice(s, s + 2000)
                    collection.upsert(
                        ids=[f"{fhash}_{i}" for i in range(len(chunks))][sl],
                        documents=chunks[sl],
                        embeddings=vectors[sl],
                        metadatas=[
                            {
                                "filename": name,
                                "file_hash": fhash,
                                "chunk_index": i,
                                "indexed_at": datetime.now().isoformat(timespec="seconds"),
                            }
                            for i in range(len(chunks))
                        ][sl],
                    )
                new_chunks += len(chunks)
        if progress_cb:
            progress_cb(n / total, f"Processed {name}")
    return new_chunks, skipped


def retrieve(question: str, k: int, only_files=None):
    kwargs = {"where": {"filename": {"$in": only_files}}} if only_files else {}
    res = collection.query(
        query_embeddings=[embedder.encode(question).tolist()],
        n_results=min(k, max(collection.count(), 1)),
        **kwargs,
    )
    return list(zip(res["documents"][0], res["metadatas"][0], res["distances"][0]))


def kb_stats():
    metas = collection.get(include=["metadatas"])["metadatas"] if collection.count() else []
    per_file = {}
    for m in metas:
        per_file[m["filename"]] = per_file.get(m["filename"], 0) + 1
    return per_file


# =====================================================
# LLM
# =====================================================
SYSTEM_QA = """You are MR Analyst, a senior data analyst assistant.
Answer using ONLY the provided context from the user's uploaded files.
- Cite the source file name in brackets, e.g. [sales.csv].
- Show calculations and use bullet points or small tables when helpful.
- If the answer is not in the context, say so clearly and suggest what to upload or ask.
- Never invent numbers. Note when the context is a partial sample of a larger dataset."""


def stream_answer(question, context, history, temperature):
    msgs = [{"role": "system", "content": SYSTEM_QA}]
    msgs += [{"role": m["role"], "content": m["content"]} for m in history[-6:]]
    msgs.append({"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"})
    stream = client.chat.completions.create(
        model=MODEL, messages=msgs, temperature=temperature, stream=True
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


def ask_llm(system, user, temperature=0.2):
    r = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=temperature,
    )
    return r.choices[0].message.content


# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:
    st.markdown(f"## 🤖 {APP_NAME}")
    st.caption("✨ AI-Powered Data Analyst Platform")
    st.divider()

    st.markdown(f"**👤 {PROFILE['name']}**")
    st.caption(PROFILE["role"])
    st.info(PROFILE["about"])
    c1, c2 = st.columns(2)
    c1.link_button("💼 LinkedIn", PROFILE["linkedin"], use_container_width=True)
    c2.link_button("🐙 GitHub", PROFILE["github"], use_container_width=True)
    st.divider()

    st.markdown("### ⚙️ Settings")
    top_k = st.slider("Chunks retrieved (top-k)", 1, 10, 4)
    temperature = st.slider("Creativity", 0.0, 1.0, 0.2, 0.05)
    per_file = kb_stats()
    scope = st.multiselect("Restrict search to files", list(per_file), placeholder="All files")
    st.divider()

    st.markdown("### 🛠️ Tech Stack")
    st.markdown(
        "".join(
            f'<span class="chip">{t}</span>'
            for t in ["Python", "Pandas", "RAG", "ChromaDB", "SBERT", "Groq", "Streamlit"]
        ),
        unsafe_allow_html=True,
    )
    st.divider()
    if st.button("🗑️ Clear knowledge base", use_container_width=True):
        collection.delete(ids=collection.get()["ids"]) if collection.count() else None
        ss.files, ss.messages = {}, []
        st.rerun()

# =====================================================
# HERO
# =====================================================
st.markdown(
    f"""
<div class="hero">
  <h1>🤖 {APP_NAME}</h1>
  <p>{TAGLINE}</p>
  <span class="pill">📄 CSV · Excel · PDF · Word · TXT</span>
  <span class="pill">🧠 Retrieval-Augmented Generation</span>
  <span class="pill">⚡ Streaming answers</span>
  <span class="pill">🗄️ Persistent vector store</span>
</div>
""",
    unsafe_allow_html=True,
)

tab_hub, tab_chat, tab_sql, tab_insights = st.tabs(
    ["📁 Data Hub", "💬 Ask MR Analyst", "🧮 SQL Studio", "📈 Insights"]
)

# =====================================================
# TAB 1 - DATA HUB
# =====================================================
with tab_hub:
    uploads = st.file_uploader(
        "Drop your files here", type=SUPPORTED, accept_multiple_files=True
    )
    if uploads:
        for u in uploads:
            ss.files[u.name] = u.getvalue()

    col_a, col_b = st.columns([1, 4])
    if col_a.button("⚡ Process & Index", type="primary", use_container_width=True):
        if not ss.files:
            st.warning("Please upload at least one file.")
        else:
            bar = st.progress(0.0, text="Starting...")
            try:
                added, skipped = index_files(lambda p, t: bar.progress(p, text=t))
                bar.empty()
                st.success(
                    f"Indexed {added} new chunks. Skipped {skipped} unchanged file(s)."
                )
            except Exception as e:
                bar.empty()
                st.error(f"Indexing failed: {e}")
    col_b.caption(
        "Files are split into chunks, embedded, and stored in ChromaDB. "
        "Unchanged files are skipped automatically."
    )

    per_file = kb_stats()
    m1, m2, m3, m4 = st.columns(4)
    metric_card(m1, len(ss.files), "Files loaded")
    metric_card(m2, len(per_file), "Files indexed")
    metric_card(m3, collection.count(), "Chunks in KB")
    tab_files = [n for n in ss.files if n.lower().endswith(TABULAR)]
    metric_card(m4, sum(len(load_df(n, ss.files[n])) for n in tab_files), "Data rows")

    if per_file:
        st.markdown("#### 📚 Knowledge base")
        st.dataframe(
            pd.DataFrame(
                [{"File": f, "Chunks": c} for f, c in per_file.items()]
            ),
            use_container_width=True,
            hide_index=True,
        )

# =====================================================
# TAB 2 - CHAT
# =====================================================
with tab_chat:
    if collection.count() == 0:
        st.info("👈 Upload and index a file in the **Data Hub** tab to start chatting.")

    if not ss.messages and collection.count() > 0:
        st.markdown("**Try asking:**")
        sugg = [
            "Summarize the key points of my data",
            "What are the top 5 values by revenue or count?",
            "Are there any anomalies or missing values?",
        ]
        cols = st.columns(len(sugg))
        for c, s in zip(cols, sugg):
            if c.button(s, use_container_width=True):
                ss.pending = s
                st.rerun()

    for m in ss.messages:
        with st.chat_message(m["role"], avatar="🧑‍💻" if m["role"] == "user" else "🤖"):
            st.markdown(m["content"])
            if m.get("sources"):
                with st.expander(f"📚 Sources ({len(m['sources'])})"):
                    for doc, meta, dist in m["sources"]:
                        st.markdown(
                            f'<div class="src"><b>{meta["filename"]}</b> · chunk '
                            f'{meta["chunk_index"]} · relevance {1 - dist:.2f}</div>',
                            unsafe_allow_html=True,
                        )
                        st.code(doc[:600], language=None)

    prompt = st.chat_input("Ask a question about your files...") or ss.pending
    ss.pending = None

    if prompt:
        if collection.count() == 0:
            st.warning("Knowledge base is empty. Index your files first.")
        else:
            with st.chat_message("user", avatar="🧑‍💻"):
                st.markdown(prompt)
            with st.chat_message("assistant", avatar="🤖"):
                try:
                    with st.spinner("Searching your data..."):
                        hits = retrieve(prompt, top_k, scope or None)
                    context = "\n\n---\n\n".join(
                        f"[{m['filename']}]\n{d}" for d, m, _ in hits
                    )
                    answer = st.write_stream(
                        stream_answer(prompt, context, ss.messages, temperature)
                    )
                    with st.expander(f"📚 Sources ({len(hits)})"):
                        for doc, meta, dist in hits:
                            st.markdown(
                                f'<div class="src"><b>{meta["filename"]}</b> · chunk '
                                f'{meta["chunk_index"]} · relevance {1 - dist:.2f}</div>',
                                unsafe_allow_html=True,
                            )
                            st.code(doc[:600], language=None)
                    ss.messages += [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": answer, "sources": hits},
                    ]
                except Exception as e:
                    st.error(f"Something went wrong: {e}")

    if ss.messages and st.button("🧹 New conversation"):
        ss.messages = []
        st.rerun()

# =====================================================
# TAB 3 - SQL STUDIO
# =====================================================
with tab_sql:
    st.markdown("Turn plain English into **MySQL 8.0** queries based on your dataset's real schema.")
    tab_files = [n for n in ss.files if n.lower().endswith(TABULAR)]
    if not tab_files:
        st.info("Upload a CSV or Excel file in the Data Hub to use SQL Studio.")
    else:
        sel = st.selectbox("Dataset", tab_files, key="sql_file")
        df = load_df(sel, ss.files[sel])
        with st.expander("👀 Schema & sample"):
            st.dataframe(df.head(5), use_container_width=True)
        req = st.text_area(
            "What do you want to find out?",
            placeholder="e.g. Top 10 customers by total sales in 2024, with their average order value",
        )
        if st.button("✨ Generate SQL", type="primary"):
            if not req.strip():
                st.warning("Describe what you need first.")
            else:
                table = re.sub(r"\W+", "_", sel.rsplit(".", 1)[0]).lower()
                schema = "\n".join(f"{c}: {t}" for c, t in df.dtypes.astype(str).items())
                with st.spinner("Writing query..."):
                    out = ask_llm(
                        "You are an expert MySQL 8.0 developer. Use only columns that exist. "
                        "Return the SQL in one ```sql block, then a short explanation. "
                        "If the request can't be answered with the columns, say what is missing.",
                        f"Table: {table}\nSchema:\n{schema}\n\nSample:\n"
                        f"{df.head(5).to_dict(orient='records')}\n\nRequest: {req}",
                        0.1,
                    )
                m = re.search(r"```sql\n(.*?)```", out, re.S)
                if m:
                    st.code(m.group(1).strip(), language="sql")
                    st.markdown(out.replace(m.group(0), ""))
                else:
                    st.markdown(out)

# =====================================================
# TAB 4 - INSIGHTS
# =====================================================
with tab_insights:
    tab_files = [n for n in ss.files if n.lower().endswith(TABULAR)]
    if not tab_files:
        st.info("Upload a CSV or Excel file in the Data Hub to see automatic insights.")
    else:
        sel = st.selectbox("Dataset", tab_files, key="ins_file")
        df = load_df(sel, ss.files[sel])
        num = df.select_dtypes("number").columns.tolist()
        cat = df.select_dtypes(exclude="number").columns.tolist()

        k1, k2, k3, k4 = st.columns(4)
        metric_card(k1, f"{len(df):,}", "Rows")
        metric_card(k2, df.shape[1], "Columns")
        metric_card(k3, int(df.isna().sum().sum()), "Missing cells")
        metric_card(k4, int(df.duplicated().sum()), "Duplicate rows")

        st.markdown("#### 🔍 Preview")
        st.dataframe(df.head(50), use_container_width=True)

        left, right = st.columns(2)
        with left:
            st.markdown("#### 🧩 Missing values")
            miss = df.isna().sum()
            miss = miss[miss > 0]
            if miss.empty:
                st.success("No missing values 🎉")
            else:
                st.bar_chart(miss)
        with right:
            st.markdown("#### 📊 Statistics")
            st.dataframe(df.describe().T, use_container_width=True)

        st.markdown("#### 🎛️ Chart builder")
        c1, c2, c3 = st.columns(3)
        if num:
            metric_col = c1.selectbox("Metric", num)
            if cat:
                group = c2.selectbox("Group by", cat)
                agg = c3.selectbox("Aggregation", ["sum", "mean", "count", "max", "min"])
                st.bar_chart(
                    df.groupby(group)[metric_col].agg(agg).sort_values(ascending=False).head(15)
                )
            else:
                counts, edges = np.histogram(df[metric_col].dropna(), bins=20)
                st.bar_chart(pd.Series(counts, index=edges[:-1].round(2)))
            if len(num) > 1:
                st.markdown("#### 🔗 Correlation")
                st.dataframe(df[num].corr().round(2).style.background_gradient(cmap="RdBu_r"))

        st.markdown("#### 🤖 AI-generated insights")
        if st.button("Generate insights", type="primary"):
            with st.spinner("Analyzing..."):
                st.markdown(
                    ask_llm(
                        "You are a senior data analyst. From the dataset profile, give: "
                        "1) a 2-sentence overview, 2) 4-5 key observations, 3) data quality issues, "
                        "4) 3 recommended analyses. Be specific and concise. Do not invent values.",
                        profile_text(sel, df) + f"\n\nSample rows:\n{df.head(8).to_string()}",
                    )
                )