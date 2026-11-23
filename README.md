# Hallucination-Aware Hybrid RAG with a Retrieval Transparency Score (RTS v2)

**CSD358 Information Retrieval: Final Project, Shiv Nadar University**
Prithviraj Jhunjhunwala (2310110232) · Diggaj Rupani · Saksham Dhawan · Amey Gautam

Scientific claim verification with **hybrid retrieval**, **generated verdicts with citations** and **sentence-level
verification**. Each sentence the system writes comes with its supporting evidence span and a trust score (RTS v2).
This project extends our mid-term hackathon project (Track T1).

📄 **Report:** [`report/CSD358_Final_Report.pdf`](report/CSD358_Final_Report.pdf) (IEEE two-column) ·
📓 **Code:** [`notebooks/CSD358_Final_Project.ipynb`](notebooks/CSD358_Final_Project.ipynb) (single notebook, executed with outputs) ·
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Prithvi985/CSD358-Hallucination-Aware-RAG/blob/main/notebooks/CSD358_Final_Project.ipynb)

---

## Pipeline

```
claim ─► preprocessing ─► TF-IDF lnc.ltc ┐
                          QL-Dirichlet   ├─► weighted RRF ─► cross-encoder ─► top-5 abstracts
                          MiniLM (dense) │   (k=10)          re-ranking            │
                          BGE-base       ┘                   (BGE-reranker)        ▼
                                                                    Qwen2.5-3B: verdict + cited sentences
                                                                                   │
   final verdict + sentence-level   ◄── verdict correction ◄── RTS v2 per sentence ◄┘
   transparency report                 (claim-level NLI)       span attribution + SBERT similarity
                                                               + DeBERTa NLI + contradiction penalty
                                                               + citation check
```

**RTS v2** for a sentence *s*, using its top-*m* evidence spans *T*:

```
support(s) = max_u∈T [ α·cos(s,u) + (1−α)·P_entail(u,s) ]
contra(s)  = max_u∈T P_contradict(u,s)
RTS_v2(s)  = support(s) · (1 − λ·contra(s))          α = 0.75, λ = 0.25 (tuned on train)
```
Labels: **HIGH** (≥95% grounded on train) · **MED** · **LOW** (≥90% hallucinated on train) · **CONTRADICTED** ·
**CONFLICTING** · **META** (abstention, not scored).

---

## Key results (BEIR SciFact test, 300 claims; every parameter tuned on the train split only)

| | Mid-term | **Final** |
|---|---|---|
| Retrieval NDCG@10 | 0.688 (BM25) / 0.692 (hybrid) | **0.766** (+0.078, p < 0.001) |
| Retrieval Recall@10 | 0.825 | **0.897** |
| Hallucination detection ROC-AUC (677-sentence test benchmark) | 0.660 (RTS v1) | **0.958** (RTS v2) |
| Negations caught | 75.6% | **95.6%** |
| Claim verification macro-F1 | 0.414 (LLM alone) | **0.673** (LLM + RTS v2) |

**Comparison with published work**
* **BEIR** (Thakur et al., 2021): best SciFact system in the paper 0.688 NDCG@10 → ours **0.766**. Our BM25+CE replication: 0.689.
* **SciFact** (Wadden et al., 2020): abstract-level label-only F1 (open setting) VeriSci 51.0 → ours **56.6**.

**Generalisation (no re-tuning), NDCG@10:** NFCorpus 0.326 (BM25) → **0.383**; FiQA 0.240 (BM25) → **0.432**.

**Honest findings.** On 80 hand-labelled real LLM sentences, detection is much harder (RTS v2 ROC-AUC 0.573; RTS v1 0.370).
Most real errors are subtle *unsupported* statements, not contradictions. Answer-level RTS also does not predict verdict
correctness. Both results are discussed in the report.

---

## Repository structure

```
├── README.md
├── requirements.txt
├── notebooks/
│   ├── CSD358_Final_Project.ipynb    # the whole project (Parts 0–6), executed with outputs
│   └── CSD358_Labelling_Tool.ipynb   # standalone tool used for the 80 human labels
├── report/
│   ├── CSD358_Final_Report.pdf
│   └── latex/                        # Overleaf source (main.tex + figures)
├── data/
│   ├── beir_scifact/                 # corpus, queries, qrels/train.tsv, qrels/test.tsv (BEIR format)
│   ├── scifact_original/             # SciFact claims with SUPPORT/CONTRADICT labels + rationales
│   ├── claims_gold.json              # merged gold labels used for evaluation
│   ├── abstract_sentences.json       # sentence-split abstracts (span index)
│   └── split_qids.json               # train / test claim ids
├── results/                          # all result tables (CSV) + generations, RTS scores, human labels (JSON)
├── figures/                          # all figures (PNG + PDF)
```
NFCorpus and FiQA are downloaded automatically by the notebook (Part 5.7). They are not stored here because of their size.

## Notebook outline

| Part | Content |
|---|---|
| 0 | Environment + shared IR library: preprocessing, inverted index, TF-IDF (lnc/ltc + zones), BM25, dense retrieval, RRF, metrics (verified against `trec_eval`) |
| 1 | Data (BEIR SciFact + SciFact labels), exact mid-term reproduction, corrected baseline |
| 2 | Retrieval upgrades: full-abstract dense, BGE, train-split tuning, QL-Dirichlet, RM3, weighted RRF, cross-encoder re-ranking, significance tests, BEIR comparison |
| 3 | Generative claim verification with citations (Qwen2.5-3B-Instruct) |
| 4 | RTS v2: span index, NLI, benchmark, tuning, scoring, verdict correction |
| 5 | Evaluation: detection (test + human labels), claim verification vs. SciFact paper, selective answering, ablations, NFCorpus/FiQA |
| 6 | Interactive Gradio demo |

## How to run

1. Open `notebooks/CSD358_Final_Project.ipynb` in **Google Colab**: File → Upload notebook (or *Open in Colab* from GitHub).
2. **Runtime → Change runtime type → T4 GPU**.
3. **Runtime → Run all** and allow Google Drive access. Results are cached in `MyDrive/CSD358_IR_Final/`.
   * The first full run takes about 2–3 hours (LLM generation, NLI scoring, FiQA encoding). Later runs take about 15 minutes, using the cache.
   * To reuse the data in this repo, copy `data/` into `MyDrive/CSD358_IR_Final/data/` before running.
4. Part 6 prints a public `https://….gradio.live` link to the demo, which has three tabs:
   **Verify a claim** (with *💉 Inject a hallucination*), **Search lab** and **Results**.

Local use: `pip install -r requirements.txt`. The notebook assumes Colab paths; change `ROOT` in Part 0.1.

## Models used
`all-MiniLM-L6-v2` · `BAAI/bge-base-en-v1.5` · `BAAI/bge-reranker-base` · `cross-encoder/ms-marco-MiniLM-L-6-v2` ·
`Qwen/Qwen2.5-3B-Instruct` · `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` · `cross-encoder/nli-deberta-v3-base`

## Data sources
* BEIR SciFact, NFCorpus, FiQA-2018: Thakur et al. (2021), via Hugging Face `BeIR/*`.
* SciFact claims and rationales: Wadden et al. (2020), official release.

## AI-use declaration
AI assistance was used for code structure, debugging and drafting documentation. All code was run, and all results
were produced and checked by the authors on Google Colab. The 80 human labels were made by the authors without seeing
the system's scores.
