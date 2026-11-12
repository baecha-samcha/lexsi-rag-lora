# Lexsi RAG — When to Search, What to Search, and When to Abstain

Lexsi AI Safety and Alignment Hackathon, Track 3 (Fine-tuning for RAG).
LoRA fine-tuning that teaches Cohere Tiny Aya Global to decide
(1) whether to search, (2) what to search for, and (3) when to abstain
for lack of evidence. Trained with AgentTune SFT.

## Links

- Article: https://docs.google.com/document/d/1OMCwBMJnIsP3m8y8kk6EhIoPnXnHaY2tdj3SA8-grSo/edit
- Adapter: https://huggingface.co/baecha/tiny-aya-rag-lora
- Dataset: https://huggingface.co/datasets/baecha/lexsi-rag-data
- Collection: https://huggingface.co/collections/baecha/lexsi-rag-when-to-search-what-to-search-when-to-abstain-6af5a1b617d1cf9f6dc3ca2e
- Base model: https://huggingface.co/CohereLabs/tiny-aya-global (CC-BY-NC)

## Results (held-out test domain, n=150)

| Metric | Base | Adapter |
|---|---|---|
| Format valid (stage 2) | 0.811 | 1.0 |
| Answer accuracy (stage 2) | 0.143 | 0.548 |
| Citation accuracy | 0.0 | 1.0 |
| Abstain on unanswerable | 0.208 | 0.958 |
| False abstain | 0.048 | 0.429 |
| Stage-1 direct-answer accuracy | 0.0 | 1.0 |

Limitation: on the unseen domain the model is over-cautious, not wrong —
false abstention 0.429, while citation accuracy on answered items is 1.0.
The test set was evaluated exactly once; the model was kept as-is and
never retrained on test metrics.

## Files

- `gen_data.py` — pilot data generator (2 domains, 100 train / 100 eval)
- `gen_data_full.py` — full data generator (6 domains, 1,324 train / 100 val / 150 test)
- `app.py` — model loading, greedy generation, AgentTune training, evaluation (also runs as a Gradio Space)
- `score.py` — stage-1 / stage-2 scorer (run it directly for a self-check: gold answers score 1.0)
- `run_pilot.py` — pilot runner: baseline → train → evaluate, uploads to Hugging Face
- `run_full.py` — full training runner: same steps on `data_full/`
- `eval_test.py` — held-out test evaluation, once only

Data is not in this repo; it lives in the public dataset above.
`data/` and `data_full/` are generated locally or downloaded from the dataset.

## Reproduction

Requirements: Python ≥ 3.12, GPU for training/evaluation (L4 used).
See `requirements.txt` / `pyproject.toml`.
Tiny Aya Global is gated (CC-BY-NC) — accept its terms on the model page
and use a token with access first.

1. Get the data (or regenerate it with `python3 gen_data_full.py`):

```bash
hf download baecha/lexsi-rag-data --repo-type dataset --local-dir data_full
```

2. Full training (HF Jobs L4, ~54 min, ~$0.74):

```bash
hf jobs run --flavor l4x1 --timeout 3h -s HF_TOKEN -v .:/code ghcr.io/astral-sh/uv:python3.12-bookworm -- bash -c "cp -r /code /tmp/project && cd /tmp/project && uv run run_full.py"
```

3. Held-out test evaluation (1 h timeout). Note: the test set was already
evaluated once for the submission; re-running it counts as a second use.

```bash
hf jobs run --flavor l4x1 --timeout 1h -s HF_TOKEN -v .:/code ghcr.io/astral-sh/uv:python3.12-bookworm -- bash -c "cp -r /code /tmp/project && cd /tmp/project && uv run eval_test.py"
```

## License

CC-BY-NC-4.0 — see `LICENSE`. The base model is CC-BY-NC; only the
LoRA adapter is distributed, never the merged weights.
