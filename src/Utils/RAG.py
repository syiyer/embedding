"""clinical_rag_demo_no_summ_clean.py"""

from __future__ import annotations
import argparse, textwrap, os
import openai
import re
from sqlalchemy import create_engine, text as sql_text
import sys
import torch
import clip
from torch.nn.functional import softmax
import base64


# ----- IRIS connection -----
import os
from dotenv import load_dotenv
load_dotenv()

conn_str = os.getenv("IRIS_CONN_STR")
if not conn_str:
    raise RuntimeError("IRIS_CONN_STR not set in .env")

engine = create_engine(conn_str)

OPENAI_API_KEY= os.getenv("OPENAI_API_KEY")

# 1️⃣ Reload CLIP + IRIS
device    = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14@336px", device=device)

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# 2️⃣ Embed your query text
def embed_text(text):
    tokens = clip.tokenize([text]).to(device)         # batch of 1
    with torch.no_grad():
        feats = model.encode_text(tokens)             # (1,512)
        feats /= feats.norm(dim=-1, keepdim=True)     # L2 normalize
    return feats.cpu().numpy().flatten().tolist()     # to plain Python list

def retrieve_image_chunks(query, slug: str, k=5, temp=100.0):
    vec = embed_text(query)

    sql = f"""
      SELECT DISTINCT TOP {k} pdf, content,
             VECTOR_DOT_PRODUCT(embedding, TO_VECTOR(:v)) AS score
        FROM Embedding.Documents
        WHERE pdf = :slug
       ORDER BY score DESC
    """
    with engine.connect() as conn:
        rows = conn.execute(sql_text(sql), {'v': str(vec), 'slug': slug}).fetchall()

    # extract scores, optionally scale by temp
    scores = torch.tensor([float(r.score) for r in rows])
    logits = scores * temp       # matches the "100.0 * image@text" in the paper
    probs  = softmax(logits, dim=0)

    # print both raw score and probability
    for r, p in zip(rows, probs):
        print(f"{r.pdf:2s} - {r.content:30s}  score={float(r.score):.4f}   prob={100*p.item():.2f}%")

    return [{"label": f"{r.pdf}|{r.content}", "path": r.content, "slug": r.pdf} for r in rows]

# Performing Vector Search
def retrieve_text_chunks(query: str, slug: str, top_k: int = 3) -> list[str]:
    """
    Return up to `top_k` *distinct* chunk_text rows ordered by vector similarity,
    filter out any exact duplicates, and normalize to sentence case.
    """
    # 1) pull DISTINCT from the DB
    if slug:
        sql = sql_text(f"""
            SELECT DISTINCT TOP {top_k} Description
            FROM Embedding.Financial
            WHERE PDF = :slug
            ORDER BY VECTOR_DOT_PRODUCT(
                    DescriptionEmbedding,
                    EMBEDDING(:qtxt, 'bge-base-config')
                    ) DESC
        """)
        with engine.connect() as conn, conn.begin():
            rows = conn.execute(sql, {"qtxt": query, 'slug': slug}).fetchall()
    else:
        sql = sql_text(f"""
            SELECT DISTINCT TOP {top_k} Description
            FROM Embedding.Financial
            ORDER BY VECTOR_DOT_PRODUCT(
                    DescriptionEmbedding,
                    EMBEDDING(:qtxt, 'bge-base-config')
                    ) DESC
        """)

        with engine.connect() as conn, conn.begin():
            rows = conn.execute(sql, {"qtxt": query}).fetchall()

    return [row[0] for row in rows if row[0]]

# ------- Print Context for evidence ---------
def show_context(chunks: list[str]) -> None:
    print("\n=== Context Sent to LLM ===\n")
    for i, txt in enumerate(chunks, 1):
        preview = txt.replace("\n", " ")[:300]
        more = "…" if len(txt) > 300 else ""
        print(f"[{i}] {preview}{more}\n")

def answer_question(question: str, text_ctx: list[str], img_ctx: list[dict]) -> str:
    # 1) Build your content blocks
    content_blocks = []

    # a) text context
    txt = "\n\n".join(f"[T{i+1}] {c}" for i, c in enumerate(text_ctx)) or "No financial text context."
    content_blocks.append({
        "type": "input_text",
        "text": txt
    })

    # b) image contexts
    for i, img_meta in enumerate(img_ctx):
        # img_path = os.path.join("data/images", img_meta["path"])
        slug    = img_meta["slug"]
        img_path = os.path.join("data", slug, "images", img_meta["path"])
        b64 = encode_image(img_path)
        content_blocks.append({
            "type": "input_image",
            "image_url": f"data:image/jpeg;base64,{b64}"
        })

    # c) the actual question (and optional prompt to describe images)
    content_blocks.append({
        "type": "input_text",
        "text": f"Question: {question}"
    })
    client = openai.OpenAI(api_key=OPENAI_API_KEY)
    # 2) Call the OpenAI responses endpoint
    response = client.responses.create(
        model="gpt-4o",
        input=[
            {
                "role": "system",
                "content": (
                    "You are a financial portfolio analysis assistant. "
                    "Use both text snippets and referenced images to answer the user’s question. "
                    "If the context is insufficient, reply with 'Insufficient data'."
                )
            },
            {
                "role": "user",
                "content": content_blocks
            }
        ],
        temperature=0.3,
    )

    # 3) Extract and return the assistant’s reply
    return response.output_text.strip()

# ----- CLI -----
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Financial Portfolio RAG with Images")
    parser.add_argument("--question", required=True, help="User’s portfolio question")
    parser.add_argument("--top-k", type=int, default=3, help="# of items per context type")
    parser.add_argument("--table", default="UOB", help="Embedding table name for text context")
    args = parser.parse_args()

    text_chunks = retrieve_text_chunks(args.question, args.top_k, args.table)
    img_chunks  = retrieve_image_chunks(args.question, args.top_k)
    if not text_chunks and not img_chunks:
        print("No context retrieved.")
        sys.exit(1)

    show_context(text_chunks, img_chunks)
    answer = answer_question(args.question, text_chunks, img_chunks)
    print("\n=== Assistant Response ===\n")
    print(answer)
