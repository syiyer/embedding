import json, os, sys
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from pathlib import Path
from dotenv import load_dotenv

# ─── config ───────────────────────────────────────────────────────────────
load_dotenv()                              
IRIS_CONN_STR = os.getenv("IRIS_CONN_STR")
if not IRIS_CONN_STR:
    print("❌ IRIS_CONN_STR is not set in your .env", file=sys.stderr)
    sys.exit(1)

engine = create_engine(IRIS_CONN_STR)        

# ─── loader function ──────────────────────────────────────────────────────
def load_chunks_to_iris(pdf_slug: str, chunks_json_path: str, batch_size: int = 5) -> int:
    tbl = "Embedding.Financial"

    # ── 1) DDL in its own connection ──────────────────────────────────
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as ddl_conn:
        # autocommit mode by default for connect()
        ddl_conn.execute(text(f"""
          CREATE TABLE IF NOT EXISTS {tbl} (
            Name VARCHAR(200),
            Length INT,
            Description LONGVARCHAR,
            DescriptionEmbedding EMBEDDING('bge-base-config','Description'),
            NameEmbedding        EMBEDDING('bge-base-config','Name'),
            PDF VARCHAR(50)
          )
        """))

        exists = ddl_conn.execute(text(f"""
          SELECT COUNT(*) 
            FROM INFORMATION_SCHEMA.INDEXES
           WHERE TABLE_SCHEMA = 'Embedding'
             AND TABLE_NAME   = 'Financial'
             AND INDEX_NAME   = 'HNSWIndex'
        """)).scalar()

        if exists == 0:
            ddl_conn.execute(text(f"""
              CREATE INDEX HNSWIndex 
                ON TABLE {tbl} (DescriptionEmbedding)
                AS HNSW(M=32, efConstruction=100, Distance='DotProduct')
            """))

    # ── 2) DML (inserts) in a separate transaction ────────────────────
    path = Path(chunks_json_path)
    if not path.exists():
        raise FileNotFoundError(f"chunks.json not found at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    insert_sql = text(f"""
      INSERT INTO {tbl} (Description, Length, Name, PDF)
      VALUES (:text, :tokens, :heading, :pdf)
    """)
    inserted = 0
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        params = []
        for idx, chunk in enumerate(batch, start=i+1):
            text_val = chunk.get("text", "")
            heading  = chunk.get("heading", "")
            tokens   = chunk.get("tokens", 0)
            if not text_val:
                print(f"⚠ Skipping empty chunk #{idx}")
                continue
            params.append({
                "text":    text_val,
                "tokens":  tokens,
                "heading": heading,
                "pdf":     pdf_slug
            })

        if not params:
            continue

        try:
            # executemany-style bulk insert
            with engine.begin() as tx:
                tx.execute(insert_sql, params)
            inserted += len(params)
            print(f"✅ Batch {i//batch_size +1}: inserted {len(params)} rows")
        except SQLAlchemyError as e:
            print(f"❌ Batch {i//batch_size +1} failed: {e}")

    print(f"🎉 Done. Total inserted: {inserted} rows into {tbl}")
    return inserted
    # for idx, chunk in enumerate(chunks, start=1):
    #     text_val = chunk.get("text", "")
    #     heading  = chunk.get("heading", "")
    #     tokens   = chunk.get("tokens", 0)
    #     if not text_val:
    #         print(f"⚠ Skipping empty chunk #{idx}")
    #         continue

    #     try:
    #         with engine.begin() as conn:
    #             conn.execute(insert_sql, {
    #                 "text":    text_val,
    #                 "tokens":  tokens,
    #                 "heading": heading,
    #                 "pdf":     pdf_slug
    #             })
    #             inserted += 1
    #     except SQLAlchemyError as e:
    #         print(f"❌ Insert failed at chunk #{idx}: {e}")

    # print(f"✅ Done. Inserted {inserted} rows into {tbl}")
    # return inserted

# ─── CLI entrypoint ───────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Load chunks.json into IRIS")
    p.add_argument("slug", help="PDF slug (matches your directory name & table suffix)")
    p.add_argument("json_path", help="Path to chunks.json")
    args = p.parse_args()
    load_chunks_to_iris(args.slug, args.json_path)
