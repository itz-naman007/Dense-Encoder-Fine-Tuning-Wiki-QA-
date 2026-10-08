# Domain-Specific Dense Encoder Fine-Tuning (Wiki QA)

Training pipeline to fine-tune a lightweight Sentence Transformer ([`BAAI/bge-small-en-v1.5`](https://huggingface.co/BAAI/bge-small-en-v1.5)) for open-domain question answering on the `wiki_qa` dataset.

The model is optimized with `MultipleNegativesRankingLoss` (MNRL) to improve document retrieval for QA tasks.

**Fine-tuned model:** [nmngpt0/bge-qa-lilsmall01](https://huggingface.co/nmngpt0/bge-qa-lilsmall01)

## Results

Fine-tuning on deduplicated query-document pairs gave a 10 percentage point improvement in top-5 retrieval accuracy:

| Metric       | Pre-Fine-Tune (Baseline) | Post-Fine-Tune |
| :----------- | :----------------------- | :------------- |
| **Recall@5** | 0.7900                   | **0.8900**     |

## Architecture and Key Fixes

- **Base model:** `BAAI/bge-small-en-v1.5`
- **Loss function:** `MultipleNegativesRankingLoss`
- **Dataset:** `wiki_qa`
- **Preventing vector collapse:** Early runs showed representation collapse (Recall dropping to 0.0000). MNRL relies on in-batch negatives, so duplicate contexts paired with different queries produce conflicting gradients. The pipeline now deduplicates documents with Pandas before building the training dataset.
- **Modern API support:** Uses `SentenceTransformerTrainer` with Hugging Face `datasets`, and replaces the deprecated `warmup_ratio` argument with explicit `warmup_steps`.

## Installation

```bash
pip install sentence-transformers datasets pandas huggingface_hub
```

## Usage

Load the model from the Hugging Face Hub:

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("nmngpt0/bge-qa-lilsmall01")

queries = ["how are glacier caves formed?"]
docs = ["A glacier cave is a cave formed within the ice of a glacier."]

q_emb = model.encode(queries)
d_emb = model.encode(docs)

print(model.similarity(q_emb, d_emb))
```
