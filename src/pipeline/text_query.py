import torch, json
from sqlalchemy import create_engine, text as sql_text
import clip
from torch.nn.functional import softmax

# 1️⃣ Reload CLIP + IRIS
device    = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14@336px", device=device)
engine    = create_engine("iris://superuser:sys@localhost:51774/DEMO")

# 2️⃣ Embed your query text
def embed_text(text):
    tokens = clip.tokenize([text]).to(device)         # batch of 1
    with torch.no_grad():
        feats = model.encode_text(tokens)             # (1,512)
        feats /= feats.norm(dim=-1, keepdim=True)     # L2 normalize
    return feats.cpu().numpy().flatten().tolist()     # to plain Python list



# 4️⃣ Run the nearest-neighbor search
def search_images(query, k=5, temp=100.0):
    vec = embed_text(query)

    sql = f"""
      SELECT pdf, content,
             VECTOR_DOT_PRODUCT(embedding, TO_VECTOR(:v)) AS score
        FROM Embedding.Documents
       ORDER BY score DESC
    """
    with engine.connect() as conn:
        rows = conn.execute(sql_text(sql), {'v': str(vec)}).fetchall()

    # extract scores, optionally scale by temp
    scores = torch.tensor([float(r.score) for r in rows])
    logits = scores * temp       # matches the "100.0 * image@text" in the paper
    probs  = softmax(logits, dim=0)

    # print both raw score and probability
    for r, p in zip(rows, probs):
        print(f"{r.pdf:2s} - {r.content:30s}  score={float(r.score):.4f}   prob={100*p.item():.2f}%")

# 5️⃣ Example usage
if __name__=="__main__":
    search_images("Show me an image of a bar graph", k=10)
