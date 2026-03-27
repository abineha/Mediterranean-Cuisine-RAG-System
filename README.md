# Mediterranean Cuisine RAG System
**COMP64702: Transforming Text into Meaning**

Retrieval-Augmented Generation (RAG) pipeline for Mediterranean cuisine question-answering. Built from scratch: corpus scraping -> chunking -> embedding -> retrieval -> generation -> evaluation.

---

## Quick Start

### 1. Install dependencies
```bash
pip install streamlit sentence-transformers faiss-cpu transformers rank-bm25 rouge-score bert-score requests beautifulsoup4 numpy
```

### 2. Run the interactive UI
```bash
streamlit run demo_app.py
```
Opens at `http://localhost:8501`

### 3. Open the notebook (pre-executed, all outputs saved)
```
main.ipynb
```

---

## Folder & File Structure

```
.
├── build_corpus.py              # Scrapes Wikipedia, Wikibooks, blog → corpus/
├── chunker.py                   # 5 chunking strategies → chunks*.json
├── embedder.py                  # 4 embedding models → indices/
├── retriever.py                 # Vector / BM25 / Hybrid retrieval
├── generator.py                 # Qwen2.5-0.5B-Instruct + 3 prompt strategies
├── evaluator.py                 # P@5, R@5, MRR, ROUGE-L, BERTScore, Faithfulness
├── demo_app.py                  # Streamlit interactive UI (5 tabs)
├── main.ipynb                   # Master notebook — all cells pre-executed
├── requirements.txt             # Python dependencies (pip install -r requirements.txt)
├── environment.yml              # Conda environment (conda env create -f environment.yml)
├── README.md                    # This file
│
├── corpus/                      # 230 scraped .txt documents
│   └── *.txt                    # Each file: metadata header + full article text
├── corpus_manifest.csv          # Source, URL, word count for all 230 documents
├── corpus_combined.txt          # All 230 docs merged into one file (434,102 words)
│
├── chunks.json                  # Section-based chunks  : 946 chunks, avg 380w  ← BEST
├── chunks_fixed_200.json        # Fixed 200-word chunks : 2340 chunks, avg 201w
├── chunks_fixed_500.json        # Fixed 500-word chunks : 874 chunks,  avg 455w
├── chunks_sentence.json         # Sentence-based chunks : 1330 chunks, avg 275w
├── chunks_paragraph.json        # Paragraph chunks      : 4434 chunks, avg 81w
│
├── indices/                     # FAISS vector indices (9 model × strategy combos, 27 files)
│   ├── faiss_mpnet_section_based.bin     # BEST index
│   ├── faiss_mpnet_fixed_200.bin
│   ├── faiss_mpnet_fixed_500.bin
│   ├── faiss_mpnet_sentence_based.bin
│   ├── faiss_mpnet_paragraph.bin
│   ├── faiss_minilm_section_based.bin
│   ├── faiss_bge_section_based.bin
│   ├── faiss_bgem3_section_based.bin
│   ├── faiss_bgem3_paragraph.bin
│   ├── mapping_*.json                    # chunk_id -> vector position mappings
│   └── meta_*.json                       # model/strategy metadata per index
│
├── retrieval_results/           # Top-5 retrieval output per query (10 experiments)
│   ├── retrieval_vector_mpnet_section_based.json    # BEST (MRR=1.0, R@5=0.983)
│   ├── retrieval_bm25_none_section_based.json
│   ├── retrieval_hybrid_mpnet_section_based.json
│   ├── retrieval_vector_mpnet_fixed_200.json
│   ├── retrieval_hybrid_mpnet_fixed_200.json
│   ├── retrieval_vector_bge_section_based.json
│   ├── retrieval_vector_bgem3_section_based.json
│   ├── retrieval_vector_mpnet_paragraph.json
│   ├── retrieval_vector_bgem3_paragraph.json
│   └── retrieval_hybrid_bgem3_paragraph.json
│
├── generation_results/          # Generated answers for 15 benchmark queries
│   ├── generation_results_structured.json   # BEST (Faithfulness=0.655)
│   ├── generation_results_zero_shot.json
│   └── generation_results_few_shot.json
│
├── evaluation_results/
│   └── evaluation_results.json  # All P@5, R@5, MRR, ROUGE-L, BERTScore, Faithfulness scores
│
├── rag_benchmark_queries.json   # 15 benchmark queries (input)
├── rag_benchmark_answers.json   # [DELIVERABLE 2 | RAG BENCHMARK DATASET] Gold-standard answers for the 15 queries
├── benchmark_output.json        # Full pipeline output on the 15 benchmark queries
│
└── sample_qa/                   # Sample test files for live demo
    ├── mediterranean_sample_qa.json
    ├── east_asia_sample_qa.json
    └── south_asia_sample_qa.json
├── test_input.json         # Example input format for evaluation
├── test_output.json          # Example output format
```

---

## Pipeline Steps (run in order)

Only needed if re-building from scratch. All outputs already exist.

```bash
# 1: Scrape corpus (230 docs, 434,102 words)
python build_corpus.py

# 2: Chunk documents (all 5 strategies)
python chunker.py --strategy all

# 3: Build FAISS embeddings (all 9 model+strategy combos)
python embedder.py --run-all

# 4: Run retrieval experiments (all 10 combos)
python retriever.py --run-all

# 5: Generate answers (all 3 prompt strategies)
python generator.py --run-all

# 6: Evaluate everything
python evaluator.py
```

---

## Demo App — 5 Tabs

Launch with: `streamlit run demo_app.py`

| Tab | What it does |
|-----|-------------|
| **Single Query** | Type any Mediterranean cuisine question, get a RAG answer with retrieved context. Also supports uploading a JSON file of queries. |
| **Benchmark** | Runs all 15 benchmark queries through the pipeline and shows results. |
| **Evaluation** | After running benchmark, computes P@5, R@5, MRR, ROUGE-L, BERTScore, Faithfulness with per-query breakdown. |
| **Exploration Results** | Full comparison table of all 10 retrieval experiments and 3 generation strategies. Best config highlighted. |
| **About** | Architecture overview, key findings, pipeline description. |

**Sidebar controls** (defaults = best config):
- Retrieval Method: `vector` (best MRR=1.0)
- Embedding Model: `all-mpnet-base-v2` (best recall)
- Chunking Strategy: `section_based` (best overall)
- Prompt Strategy: `structured` (best faithfulness)
- Top-K: 5

---

## Live Demo / Test with Custom Queries

Place the test file as `test_input.json` in the **root project folder** (same folder as `demo_app.py` and `main.ipynb`):

```json
{
  "queries": [
    {"query_id": "0", "query": "What is hummus made from?"},
    {"query_id": "1", "query": "Where does paella originate?"}
  ]
}
```

Then run cell 6.4 in `main.ipynb` for results:

---

## Best Pipeline Configuration

| Component | Best Choice | Score |
|-----------|------------|-------|
| Chunking | Section-based (946 chunks, avg 380w) | — |
| Embedding | all-mpnet-base-v2 (768d) | — |
| Retrieval | Vector (FAISS cosine) | MRR=1.000, R@5=0.983 |
| Prompt | Structured | ROUGE-L=0.246, Faithfulness=0.655 |
| LLM | Qwen/Qwen2.5-0.5B-Instruct | — |

---

## Corpus Sources

| Source | Documents | Description |
|--------|-----------|-------------|
| Wikipedia | ~150 | Cuisine overviews, dish pages, ingredient pages |
| Wikibooks (Cookbook) | ~40 | Recipe and technique pages |
| Around the World in 80 Cuisines Blog | ~40 | Regional cuisine articles |
| **Total** | **230** | **434,102 words** |

---

## Evaluation Results Summary

### Retrieval (10 experiments)
| Method | Model | Chunking | P@5 | R@5 | MRR |
|--------|-------|----------|-----|-----|-----|
| **Vector** | **mpnet** | **section_based** | **0.653** | **0.983** | **1.000** |
| Vector | bgem3 | section_based | 0.653 | 0.983 | 1.000 |
| Hybrid RRF | mpnet | section_based | 0.667 | 0.983 | 0.933 |
| Vector | mpnet | fixed_200 | 0.707 | 0.861 | 0.967 |
| BM25 | — | section_based | 0.520 | 0.772 | 0.856 |

### Generation (3 prompt strategies)
| Strategy | ROUGE-L | BERTScore F1 | Faithfulness |
|----------|---------|--------------|--------------|
| **structured** | **0.246** | **0.811** | **0.655** |
| zero_shot | 0.239 | 0.819 | 0.575 |
| few_shot | 0.203 | 0.798 | 0.581 |
