
# FinAgent AI - Retrieval Evaluation

## Evaluation Setup

- Dataset: 6 synthetic evaluation questions
- Relevant questions: 5
- Out-of-domain questions: 1
- Documents: 2 fictional banking documents
- Indexed chunks: 15
- Embedding model: all-MiniLM-L6-v2
- Vector database: FAISS
- Chunking strategy: Heading-aware
- Top K: 3
- Similarity threshold: 0.30

## Results

| Metric | Result |
| --- | --- |
| Hit Rate@3 | 100.00% |
| MRR@3 | 1.0000 |
| Average Retrieval Latency | 10.23 ms |
| Out-of-Domain Rejection | 100.00% (1/1) |

## Interpretation

All five relevant questions retrieved the expected
document section in the first position.

The single out-of-domain question was rejected.

Retrieval latency includes query embedding generation
and FAISS search. It excludes LLM generation.

## Limitations

- The evaluation dataset is very small.
- All documents contain fictional banking information.
- The dataset was used during development.
- Results do not represent production performance.
- Out-of-domain rejection was tested on only one query.

## Next Steps

- Expand evaluation with an independent test dataset.
- Measure end-to-end RAG answer quality.
- Add LangGraph-based agent workflows.
- Test prompt injection and data privacy controls.
