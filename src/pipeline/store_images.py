from PIL import Image
import torch, json, glob, os
from sqlalchemy import create_engine, text as sql_text
import torch
import clip
from dotenv import load_dotenv
import sys

# 1️⃣ Load CLIP
device    = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14@336px", device=device)


# 2️⃣ Connect to IRIS
load_dotenv()                              
IRIS_CONN_STR = os.getenv("IRIS_CONN_STR")
if not IRIS_CONN_STR:
    print("❌ IRIS_CONN_STR is not set in your .env", file=sys.stderr)
    sys.exit(1)

engine = create_engine(IRIS_CONN_STR)       

# ─── Ensure the Embedding.Documents table exists ────────────────────────
def ensure_table():
    ddl = """
    CREATE TABLE IF NOT EXISTS Embedding.Documents (
      pdf VARCHAR(40),
      modality VARCHAR(16),
      content VARCHAR(512),
      embedding VECTOR(FLOAT, 768)
    )
    """
    with engine.begin() as conn:
        conn.execute(sql_text(ddl))
    print("✅ Table Embedding.Documents is ready.")

def ingest_image(path, pdf_name: str):
    img_t = preprocess(Image.open(path).convert("RGB")) \
               .unsqueeze(0) \
               .to(device)
    with torch.no_grad():
        feats = model.encode_image(img_t)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    vec = feats.cpu().numpy().flatten().tolist()
    name = os.path.basename(path)

    sql = """
      INSERT INTO Embedding.Documents (pdf, modality, content, embedding)
      VALUES (:pdf, :modality, :content, :vec)
    """
    with engine.begin() as conn:
        conn.execute(sql_text(sql), {
          "pdf": pdf_name,
          "modality":"image",
          "content": name,
          "vec": json.dumps(vec)
        })
    print(f"Ingested {name}")

# 3️⃣ Batch-load from a directory
# ─── 5) Main: prompt for PDF name & batch‐loop ──────────────────────────────────
if __name__ == "__main__":
    pdf_name = input("Enter PDF name (e.g. 'UOB_Market_Outlook_2025'): ").strip()
    ensure_table()

    image_dir = "data/sc_global_market_outlook/images"
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}
    paths = [
        p for p in glob.glob(f"{image_dir}/**/*", recursive=True)
        if os.path.splitext(p)[1].lower() in exts
    ]

    print(f"Found {len(paths)} images under {image_dir}. Starting ingestion…")
    for img_path in paths:
        ingest_image(img_path, pdf_name)

    print("🎉 All done!")