"""
Run end-to-end RAG inference from the terminal.

Usage:
    python run_inference.py
    python run_inference.py test_input.json test_output.json
"""

import os
import sys
import json
import time
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILE  = sys.argv[1] if len(sys.argv) > 1 else "test_input.json"
OUTPUT_FILE = sys.argv[2] if len(sys.argv) > 2 else "test_output.json"

if not os.path.exists(INPUT_FILE):
    print(f"ERROR: '{INPUT_FILE}' not found.")
    print("Place the input JSON file in the project folder and try again.")
    sys.exit(1)

from retriever import (
    load_chunks, setup_vector,
    retrieve_vector, MODELS, CHUNK_FILES as RETRIEVER_CHUNK_FILES
)
from generator import (
    load_model, format_context, build_messages, generate_answer, MODEL_NAME
)

print("=" * 60)
print("  Mediterranean Cuisine RAG — Inference Pipeline")
print("=" * 60)
print(f"  Input  : {INPUT_FILE}")
print(f"  Output : {OUTPUT_FILE}")
print(f"  Config : vector + mpnet + section_based + structured")
print("=" * 60)

print("\n[1/3] Loading FAISS index + mpnet...")
chunks = load_chunks(RETRIEVER_CHUNK_FILES["section_based"])
cl = {c["chunk_id"]: c for c in chunks}
vec_setup = setup_vector("mpnet", "section_based")

print("\n[2/3] Loading Qwen LLM...")
llm_model, tokenizer = load_model()

print("\n[3/3] Running inference...")
with open(INPUT_FILE, encoding="utf-8") as f:
    data = json.load(f)
queries = data["queries"]
print(f"  Loaded {len(queries)} queries\n")

results = []
total_start = time.time()

for i, q in enumerate(queries):
    query_id  = q["query_id"]
    query_text = q["query"]

    t0 = time.time()
    hits = retrieve_vector(query_text, vec_setup, k=5)
    retrieval_time = time.time() - t0

    ret_chunks = []
    for hit in hits:
        chunk = cl.get(hit["chunk_id"], {})
        ret_chunks.append({"doc_id": hit["chunk_id"], "text": chunk.get("text", "")})

    context_str = format_context(ret_chunks)
    msgs = build_messages("structured", context_str, query_text)
    response, gen_time = generate_answer(llm_model, tokenizer, msgs)

    formatted_context = [
        {"doc_id": f"{idx:03d}", "text": rc["text"]}
        for idx, rc in enumerate(ret_chunks)
    ]

    results.append({
        "query_id": query_id,
        "query": query_text,
        "response": response,
        "retrieved_context": formatted_context,
    })

    print(f"  [{i+1}/{len(queries)}] Q{query_id} "
          f"(retrieval: {retrieval_time:.2f}s, generation: {gen_time:.1f}s)")
    print(f"           {response[:120]}...")

total_time = time.time() - total_start

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump({"results": results}, f, indent=2, ensure_ascii=False)

print(f"\n{'=' * 60}")
print(f"  Done! {len(results)} queries processed in {total_time:.1f}s")
print(f"  Output saved: {OUTPUT_FILE}")
print(f"{'=' * 60}")
