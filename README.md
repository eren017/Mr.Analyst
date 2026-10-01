# 🤖 MR Analyst

**Upload your data. Ask anything. Get answers with sources.**

MR Analyst is an AI-powered data analyst platform built with **Retrieval-Augmented Generation (RAG)**. Upload CSV, Excel, PDF, Word or text files, and chat with them in plain English. Answers are streamed in real time and come with the exact source chunks used, so you can verify every response.

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?logo=streamlit&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-7c3aed)
![Groq](https://img.shields.io/badge/Groq-LLM-orange)

---

## 📸 Screenshots

### Data Hub
Upload files, index them into the vector store, and track knowledge-base stats.

![Data Hub](assets/screenshots/data-hub.png)

### Ask MR Analyst
Chat with your files. Every answer is grounded in your data and cites its sources.

![Chat](assets/screenshots/chat.png)

### SQL Studio
Turn plain English into MySQL 8.0 queries based on your dataset's real schema.

![SQL Studio](assets/screenshots/sql-studio.png)

### Insights
Automatic data profiling, charts, correlations and AI-generated insights.

![Insights](assets/screenshots/insights.png)

---

## ✨ Features

- **📁 Multi-format ingestion**: CSV, XLSX, XLS, PDF, DOCX and TXT (up to 200 MB per file).
- **🧠 RAG pipeline**: files are chunked, embedded with Sentence-Transformers, and stored in ChromaDB.
- **⚡ Streaming answers**: responses stream token by token from a Groq-hosted LLM.
- **📚 Source citations**: each answer shows the file name, chunk index and relevance score of the retrieved context.
- **🗄️ Persistent vector store**: the index is saved on disk, and unchanged files are skipped automatically (MD5 file hashing).
- **🎯 Scoped search**: restrict retrieval to specific files, and tune top-k and temperature from the sidebar.
- **🧮 SQL Studio**: generates MySQL 8.0 queries from natural language using your dataset's actual columns and types.
- **📈 Insights dashboard**: row/column counts, missing values, duplicates, summary statistics, a chart builder, a correlation matrix and AI-written observations.
- **🎨 Custom dark theme** via Streamlit's `config.toml`.

---

## 🏗️ How It Works

```
 Upload files ──► Extract text / tables ──► Chunk ──► Embed (all-MiniLM-L6-v2)
                                                          │
                                                          ▼
                                                   ChromaDB (persistent)
                                                          │
 User question ──► Embed question ──► Top-k similarity search
                                                          │
                                                          ▼
                              Context + question ──► Groq LLM ──► Streamed answer + sources
```

- **Tabular files** are converted to a dataset profile (schema + summary statistics) plus row chunks of 25 rows each.
- **PDF / DOCX / TXT** files are split into overlapping text chunks (1000 characters, 100 overlap).
- The LLM is instructed to answer **only from the retrieved context** and to cite file names.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| LLM | Groq API (`openai/gpt-oss-20b`) |
| Embeddings | Sentence-Transformers (`all-MiniLM-L6-v2`) |
| Vector DB | ChromaDB (persistent) |
| Data handling | Pandas, NumPy, openpyxl |
| Document parsing | pdfplumber, python-docx |
| Config | python-dotenv |

---

## 🚀 Getting Started

### Prerequisites
- Python **3.11+**
- A free [Groq API key](https://console.groq.com/keys)

### 1. Clone the repository
```bash
git clone https://github.com/eren017/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Create a virtual environment and install dependencies

**Using pip**
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

**Or using uv**
```bash
uv sync
```

### 3. Add your API key
Create a `.env` file in the project root:
```env
GROQ_API_KEY=your_groq_api_key_here
```
> When deploying on Streamlit Cloud, add `GROQ_API_KEY` under **Secrets** instead.

### 4. (Optional) Apply the theme
Place the theme file at `.streamlit/config.toml`:
```toml
[theme]
base = "dark"
primaryColor = "#a855f7"
backgroundColor = "#0b1020"
secondaryBackgroundColor = "#10142b"
textColor = "#e5e7eb"
font = "sans serif"
```

### 5. Run the app
```bash
streamlit run stream.py
```
Open **http://localhost:8501** in your browser.

---

## 📖 Usage

1. **Data Hub**: drag and drop your files and click **⚡ Process & Index**.
2. **Ask MR Analyst**: ask questions such as:
   - *"Summarize the key points of my data"*
   - *"What are the top 5 values by revenue or count?"*
   - *"Are there any anomalies or missing values?"*
3. **SQL Studio**: pick a CSV/Excel dataset and describe the query you need.
4. **Insights**: explore automatic stats and charts, then click **Generate insights** for an AI summary.

---

## 📂 Project Structure

```
.
├── stream.py              # Main Streamlit application
├── .streamlit/
│   └── config.toml        # Dark theme configuration
├── API_test.ipynb         # Notebook for testing LLM API connections
├── chroma_db/             # Persistent vector store (auto-generated)
├── requirements.txt       # pip dependencies
├── pyproject.toml         # uv / project metadata
├── .env                   # API keys (not committed)
└── README.md
```

---

## ⚙️ Configuration

Edit the config block at the top of `stream.py`:

| Variable | Description | Default |
|---|---|---|
| `MODEL` | Groq model used for answers | `openai/gpt-oss-20b` |
| `DB_PATH` | ChromaDB storage folder | `./chroma_db` |
| `COLLECTION` | Vector collection name | `mr_analyst` |
| `PROFILE` | Name, role, links shown in the sidebar | — |

---

## 🔮 Future Improvements

- Run generated SQL directly against the uploaded data
- Conversation export
- Support for more file formats and OCR for scanned PDFs
- Hybrid (keyword + vector) search and re-ranking
- Docker image and one-click cloud deployment

---

## 👤 Author

**Sarthak Pawar**
Final-year Computer Science Engineering student, aspiring Data Analyst / Data Engineer / AI-ML engineer.

- GitHub: [@eren017](https://github.com/eren017)
- LinkedIn: [sarthak-pawar](https://www.linkedin.com/in/sarthak-pawar-53136a201)

---

⭐ If you found this project useful, consider giving it a star!
