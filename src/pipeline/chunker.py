# ── chunk_markdown_langchain.py ────────────────────────────────────────
"""
Reads a cleaned Markdown file, splits it by heading hierarchy,
then splits those sections into overlapping chunks (~750 words each),
ensures each chunk is ≤ 512 GPT tokens, and saves as a JSON array.
No OpenAI key is needed (token counting uses `tiktoken`).
"""

import json, uuid, re
from pathlib import Path
from tqdm import tqdm  # tqdm provides a progress bar during iteration
import tiktoken
import sys

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

# ---------------------------- CONFIG -----------------------------------

# Markdown heading levels used to split the document
HEADER_LEVELS = [("#", "h1"), ("##", "h2"), ("###", "h3")]

# Approximate word limits and token constraints
WORD_CHUNK       = 750      # size of each text chunk (~600 tokens)
WORD_OVERLAP     = 100      # overlapping words between chunks
TOKEN_HARD_LIMIT = 512      # max allowed GPT tokens per chunk
MIN_TOKENS       = 40       # skip chunks smaller than this

# Token encoder for token counting (matches OpenAI GPT models)
enc = tiktoken.get_encoding("cl100k_base")

# ---------------------------- HELPERS -----------------------------------

def num_tokens(text: str) -> int:
    """Returns token count of a string using tiktoken."""
    return len(enc.encode(text))

def is_noise(chunk: str) -> bool:
    """Filters out chunks that are trivial or look like source/captions."""
    if num_tokens(chunk) < MIN_TOKENS:
        return True
    if re.match(r"^\s*source:\s*", chunk, re.I):
        return True
    if re.match(r"^\s*(figure|table)\s+\d+:", chunk, re.I):
        return True
    return False

def header_path(md: dict) -> str:
    """Builds a full header path string like 'Section > Subsection > ...'"""
    return " > ".join([md.get("h1", ""), md.get("h2", ""), md.get("h3", "")]).strip(" >")


def merge_consecutive_headers(text: str) -> str:
    """
    Scans the document for runs of top-level headers (lines starting with '# '),
    allowing blank lines in between, and collapses each run into one header:

      # A
      # B

      # C

    becomes:

      # A - B - C
    """
    lines = text.splitlines()
    merged = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        # If this line is a top-level header, collect the whole run
        if re.match(r"^#\s+\S+", line):
            headers = []
            header_level = None

            # Gather all headers and skip over blank lines in between
            while i < n and (lines[i].strip() == "" or re.match(r"^#\s+\S+", lines[i])):
                if re.match(r"^#\s+\S+", lines[i]):
                    header_level = "#"  # we only care about one '#' here
                    headers.append(lines[i].lstrip("#").strip())
                i += 1

            # Emit one merged header, then a blank line to separate from body
            merged.append(f"{header_level} {' - '.join(headers)}")
            merged.append("")  # keep one blank line
        else:
            merged.append(line)
            i += 1

    return "\n".join(merged)


# ---------------------------- MAIN PIPELINE -----------------------------

def chunk_markdown(slug: str):
    base_dir = Path("data") / slug
    md_file = base_dir / "organized_cleaned_document.md"
    if not md_file.exists():
        raise FileNotFoundError(f"Markdown not found: {md_file}")

    out_file = base_dir / "chunks.json"

    raw_text = md_file.read_text(encoding="utf-8")
    merged_text = merge_consecutive_headers(raw_text)
    # save merged version for debug
    debug_file = base_dir / "merged_output.md"
    debug_file.write_text(merged_text, encoding="utf-8")
    print(f"✅ Merged markdown written to {debug_file}")
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=HEADER_LEVELS,
        strip_headers=True,
    )
    header_docs = header_splitter.split_text(merged_text)

    # Step 3: Further split each section into word-based chunks
    word_splitter = RecursiveCharacterTextSplitter(
        chunk_size=WORD_CHUNK,
        chunk_overlap=WORD_OVERLAP,
        length_function=lambda txt: len(txt.split()),  # count by words
    )

    all_chunks = []

    # Iterate over header sections with progress bar
    for hdoc in tqdm(header_docs, desc="Header blocks"):
        # Create sub-chunks within this header section
        sub_docs = word_splitter.split_documents([hdoc])

        for sd in sub_docs:
            text = sd.page_content.strip()
            if is_noise(text):
                continue

            # If chunk is too large, split in half until it fits
            while num_tokens(text) > TOKEN_HARD_LIMIT:
                words = text.split()
                midpoint = len(words) // 2

                # NEW  ➜ keep an overlap of WORD_OVERLAP words
                part_words = words[:midpoint]
                text_words = words[midpoint - WORD_OVERLAP:]  # slide back

                part = " ".join(part_words)
                text = " ".join(text_words)

                if not is_noise(part) and num_tokens(part) >= MIN_TOKENS:
                    all_chunks.append(
                        build_chunk(sd.metadata, part)  # ← pass meta
                    )
            # Add the final chunk
            all_chunks.append(build_chunk(sd.metadata, text))


    # Write to JSON
    with open(out_file, "w", encoding="utf-8") as fout:
        json.dump(all_chunks, fout, ensure_ascii=False, indent=2)

    print(f"✅ {len(all_chunks)} chunks written to {out_file}")

# ---------------------------- CHUNK BUILDER -----------------------------

def build_chunk(meta, text):
    """Builds a single chunk dictionary with metadata and token count."""
    return {
        "id": uuid.uuid4().hex,       # Unique ID for this chunk
        "heading": header_path(meta), # Full header path
        "tokens": num_tokens(text),   # Number of GPT tokens
        "text": text                  # Chunk content
    }

# ---------------------------- ENTRY POINT -------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Chunk markdown for a given slug")
    parser.add_argument("slug", help="PDF slug folder under data/")
    args = parser.parse_args()
    chunk_markdown(args.slug)
