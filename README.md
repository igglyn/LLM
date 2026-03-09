# llm_lab

`llm_lab` is a modular experiment framework for sequence modeling research.

The core components are intentionally swappable:
- codec
- patcher
- mixer
- head
- memory

Experiments are meant to be run by changing configuration values and wiring, rather than rewriting core code paths.

## Data formats

Training supports two raw-text dataset modes via `data.dataset_type`:

- `bytes_txt` (default): reads `.txt` files from `data.path`.
- `jsonl_text`: reads teacher-generated JSONL from `data.jsonl_path` (file or directory of `.jsonl`).

For `jsonl_text`, text extraction uses this priority per JSON object:

1. `data.jsonl_text_field` (if set)
2. `completion_text`
3. `text`
4. fallback to `prompt_text + "\n" + completion_text` when both are present.

All extracted text is encoded to UTF-8 bytes with replacement on invalid code points, then chunked into fixed `seq_len` byte tensors exactly like the byte text dataset path.

## JSONL ingestion bring-up

You can sanity-check teacher JSONL ingestion without starting training:

```bash
python scripts/validate_jsonl_ingest.py --config path/to/config.toml
```

The script expects `data.dataset_type = "jsonl_text"`, builds `JsonlTextDataset`, pulls 2 batches, and prints:
- byte length
- sample decoded preview
- batch shape/dtype

It fails clearly if no usable text fields are found in JSONL records.
