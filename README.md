# LLM Monorepo

A Python monorepo with **one shared XML config source-of-truth** for both distillation and training paths.

## Why one shared XML config

Both apps (`distill` and `train`) read the same config parse tree and the same resolved model defaults.
This keeps dataset, distillation, and model structure consistent across workflows and avoids drift.

## Package boundaries

- `shared/config`
  - XML schema dataclasses (`specs.py`)
  - parser (`parser.py`)
  - resolver (`resolver.py`)
- `distill/runtime`
  - source extraction, mixture build, teacher backends, StageA runtime
- `train/runtime`
  - model runtime builder and smoke/summary execution path

Boundary rule:
- `distill` **must not** depend on `train.runtime`
- `train` **must not** depend on `distill.runtime`
- both depend on `shared.config`

## What belongs in XML vs local config

### Put in XML
- dataset logical structure (entries, source kind, split mapping, filters)
- distillation structure (teachers, stages, stage modes)
- model structure (defaults, patchers, trunk, block composition)
- train config structure (train steps, optimizer, scheduler definitions)

### Keep out of XML (use `.env`, CLI args, sidecar local config)
- machine-local paths for outputs/artifacts
- secrets/tokens/credentials
- host-specific runtime overrides
- temporary experiment-local knobs that should not become global schema

## How apps consume the same config differently

- `distill`
  - consumes dataset + distillation sections at runtime
  - runs `extract`, `mix`, `stage-a`
- `train`
  - consumes resolved model section
  - builds compositional runtime model and supports `build`, `package`, `run`, `summary`, `smoke`

## Commands

```bash
# distill
python -m distill extract --config examples/config.example.xml --output-dir /tmp/extracted
python -m distill mix --config examples/config.example.xml --input /tmp/extracted --output /tmp/mixed.jsonl
python -m distill stage-a --config examples/config.example.xml --input /tmp/mixed.jsonl --output /tmp/stage_a

# train
python -m train build --config examples/config.example.xml --dataset-file /tmp/stage_a/stage_a.jsonl --output-dir /tmp/model --token-mapping-file /tmp/stage_a/token_mapping.jsonl
python -m train package --config examples/config.example.xml --dataset-file /tmp/stage_a/stage_a.jsonl --output-dir /tmp/model_manifest
python -m train run --config examples/config.example.xml --model-file /tmp/model/model.json --text "hello"
python -m train summary --model-file /tmp/model/model.json
python -m train smoke --config examples/config.example.xml

# tests
pytest
```


## Hugging Face examples

A full Hugging Face-based example config is available at `examples/config.hf.example.xml`.
It demonstrates both:
- dataset extraction via `<Source type="huggingface" ... />` (including optional `config="..."` dataset field)
- teacher model runner via `<Backend type="huggingface"> ... </Backend>`

Example commands:

```bash
# distill with Hugging Face dataset + model backend
python -m distill extract --config examples/config.hf.example.xml --output-dir /tmp/hf_extracted
python -m distill mix --config examples/config.hf.example.xml --input /tmp/hf_extracted --output /tmp/hf_mixed.jsonl
python -m distill stage-a --config examples/config.hf.example.xml --input /tmp/hf_mixed.jsonl --output /tmp/hf_stage_a

# train runtime bring-up on the same config
python -m train build --config examples/config.hf.example.xml --dataset-file /tmp/hf_stage_a/stage_a.jsonl --output-dir /tmp/hf_model --token-mapping-file /tmp/hf_stage_a/token_mapping.jsonl
python -m train package --config examples/config.hf.example.xml --dataset-file /tmp/hf_stage_a/stage_a.jsonl --output-dir /tmp/hf_model_manifest
python -m train run --config examples/config.hf.example.xml --model-file /tmp/hf_model/model.json --text "hello"
python -m train summary --model-file /tmp/hf_model/model.json
python -m train smoke --config examples/config.hf.example.xml
```

The same StageA output directory convention applies (`stage_a.jsonl` + `token_mapping.jsonl`).

## Extension hooks

- Add new XML block types in parser + specs + runtime composition helpers.
- Add teacher backends in `distill/runtime/teacher_runtime.py` backend factory.
- Add scheduler handling in train runtime (`RuntimeTrainConfig`) without coupling to parser internals.
