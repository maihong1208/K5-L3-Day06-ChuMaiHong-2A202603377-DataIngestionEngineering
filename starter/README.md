# Day 06 starter: local data-ingestion benchmark

This starter runs three retrieval configurations against the OWASP LLM Top 10
2025 PDF in `data/`. The default setup uses a local multilingual embedding model
and persistent Chroma, so it does not require Docker or an API key. The first
run needs internet access to download the model weights; document and query
text are embedded on this machine.

## Requirements and first run (Windows PowerShell)

Use Python 3.11–3.14. From this directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python -m src.run
```

The default model is
`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, which embeds
Vietnamese questions and English source text in the same vector space. Set
`EMBEDDING_DEVICE` to a supported device if you want to use an accelerator.
The model is configured for up to 256 tokens per input when its backbone allows
it, avoiding truncation of the benchmark's longer chunks.
There is no mock-vector or API fallback: model download or initialization
errors stop the run and must be fixed explicitly.

Chroma persists cosine-distance HNSW indexes in `results/chroma/`. Each run
replaces the selected document's previous chunks in each strategy collection,
so rerunning the same corpus does not accumulate duplicate or obsolete chunks.

## Source data and benchmark

- `data/manifest.json` identifies the source, license, selected physical PDF
  pages, page sections, and normalized character counts.
- `benchmark/questions.json` contains 15 dev and 5 test questions with verified
  gold evidence. Keep the test split unchanged while tuning.
- `src/ingestion.py` extracts the manifest-selected PDF pages and retains
  physical page numbers.

## Strategies

| Strategy | Chunking | Query metadata filter |
|---|---|---|
| A | Fixed 500-character windows, 50-character overlap | None |
| B | Paragraph-aware chunks, maximum 800 characters | None |
| C | Same chunks and vectors as B | Equality filter on `section` when the question supplies one |

All strategies use the same local embedding model, persistent vector store, and
cosine metric. On a run, the pipeline indexes all three strategies, evaluates
dev first, selects by dev Evidence Hit@3 then Hit@1 (ties prefer A, then B,
then C), and evaluates the held-out test split only after that selection is
fixed. Test results are reported for all three predeclared strategies.

## Outputs

`python -m src.run` writes:

- `results/run_summary.json`: model/store settings, chunk statistics, selection,
  token/character counts, timing, and Hit@1 / Evidence Hit@3 by strategy and split.
- `results/traces.json`: full top-3 chunks, scores, metadata, filters, and
  automatic evidence-match labels for each question.
- `results/manual_audit_template.json`: test top-3 results and manual
  metadata/grounding annotations for the current completed run.
- `results/coherence_audit_template.json`: ten predetermined source positions
  with candidate chunks and coherence annotations for the current run.
- `results/chroma/`: persistent vector collections used by the run.

The runner initially creates blank audit fields; the current audit annotations
were reviewed and added after the run. Running `python -m src.run` again
regenerates those audit files, so back them up before rerunning if you need to
preserve the current annotations. Review AI-assisted labels and complete the
five-dimension analysis as described in `../GUIDE.md` and `../BENCHMARK.md`.

## Optional PostgreSQL setup

`docker-compose.yml`, `init.sql`, and `scripts/check_setup.py` are retained for
students who want to start the PostgreSQL + pgvector alternative. The default
`src.run` pipeline currently uses Chroma and does not connect to PostgreSQL.
