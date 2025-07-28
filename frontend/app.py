import streamlit as st
import sys, os
# __file__ is .../frontend/app.py → go up one level into embedding/
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.Utils.RAG import retrieve_text_chunks, retrieve_image_chunks, answer_question
from pathlib import Path
from src.pipeline.precheck import ensure_meta_table, check_duplicate, record_metadata
from src.pipeline.extractor import Extractor
from src.pipeline.cleaner import organize_and_clean_by_section
from src.pipeline.chunker import chunk_markdown
from src.pipeline.loader import load_chunks_to_iris
from src.pipeline.utils import get_pdf_list
from src.pipeline.store_images import ingest_image, ensure_table
import irisnative

# Page config & CSS
st.set_page_config(page_title="📑💹 Financial Portfolio RAG", layout="wide")
st.title("📑💹 Financial Portfolio RAG Chatbot")
st.markdown(
    """
    <style>
      .stExpanderHeader { font-size: 1.1em; font-weight: bold; }
      .citation { color: #555; font-size: 0.9em; }
    </style>
    """, unsafe_allow_html=True
)

if "slug" not in st.session_state:
    st.session_state["slug"] = ""

# --- STEP 1: PDF upload + OCR/chunking/load into IRIS ---
uploaded = st.file_uploader("Upload portfolio PDF", type="pdf")
if uploaded:
    slug = Path(uploaded.name).stem
    out_dir = Path("data") / slug
    out_dir.mkdir(exist_ok=True, parents=True)
    pdf_path = out_dir / uploaded.name
    with open(pdf_path, "wb") as f:
        f.write(uploaded.getbuffer())
    st.success(f"Saved PDF: {pdf_path}")

    # process extraction, cleaning, chunking, loading
    prog = st.progress(0)
    with st.spinner("🔍 Extracting content…"):
        extractor = Extractor(output_root="data")
        stats = extractor.extract(str(pdf_path))
    st.write("Extraction stats:", stats)
    prog.progress(25)

    with st.spinner("🧹 Cleaning & organizing…"):
        organize_and_clean_by_section("data", slug)
    prog.progress(50)

    with st.spinner("✂️ Splitting into chunks…"):
        chunk_markdown(slug)
    prog.progress(75)

    with st.spinner("📥 Loading into IRIS…"):
        conn = irisnative.createConnection("127.0.0.1", 52774, "DEMO", "superuser", "sys")        
        instance = irisnative.createIris(conn)
        rows = instance.classMethodValue(
            "Utils.JSONLoader",      # package.class name
            "LoadChunks",     # method
            slug,                    # pdfSlug argument
            str(out_dir / "chunks.json")                # JSON text argument
        )
        instance.close()
        # inserted_text = load_chunks_to_iris(slug, str(out_dir / "chunks.json"))
        # ensure image table exists and ingest images
        ensure_table()
        img_dir = out_dir / "images"
        inserted_imgs = 0
        for img_path in img_dir.iterdir():
            ingest_image(str(img_path), slug)
            inserted_imgs += 1
    prog.progress(100)
    st.success(f"Loaded {rows} text chunks and {inserted_imgs} images into IRIS.")
    st.session_state.slug = slug


    # 5️⃣ Record metadata for next time

    # if inserted > 0:
    #     record_metadata(slug, file_hash, size_bytes, page_count)

    st.success("🎉 PDF successfully processed and loaded!")

# Sidebar filters
with st.sidebar:
    st.header("📈 Query Settings")
    slug = st.session_state["slug"]
    pdfs = [""] + get_pdf_list()
    pdf = st.selectbox("PDF to query", pdfs, index=0)
    top_k = st.sidebar.slider("Top K contexts", min_value=1, max_value=10, value=3)


st.title("Financial Portfolio Assistant")
question = st.text_input("Enter your clinical question:")

if st.button("Run"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        # retrieve contexts
        text_ctx = retrieve_text_chunks(question, pdf, top_k)
        img_ctx  = retrieve_image_chunks(question, pdf, top_k)
        if not text_ctx and not img_ctx:
            st.error("No relevant context found.")
        else:
            with st.expander("🗒️ Text Context"):
                for i, t in enumerate(text_ctx, 1):
                    st.write(f"[T{i}] {t}")
            with st.expander("🖼️ Image Context"):
                for i, im in enumerate(img_ctx, 1):
                    img_file = Path("data") / slug / "images" / im['path']
                    if img_file.exists():
                        st.image(str(img_file), caption=f"[I{i}] {im['label']}", use_column_width=True)
            with st.spinner("💭 Generating answer…"):
                answer = answer_question(question, text_ctx, img_ctx)
            st.markdown("### 🗨️ Assistant Answer")
            st.markdown(answer)