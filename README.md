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
- train config structure (train mode, optimizer, scheduler definitions)

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
  - builds compositional runtime model and supports `build`, `summary`, `smoke`

## Commands

```bash
# distill
python -m distill extract --config examples/config.example.xml --output /tmp/extracted.jsonl
python -m distill mix --config examples/config.example.xml --input /tmp/extracted.jsonl --output /tmp/mixed.jsonl
python -m distill stage-a --config examples/config.example.xml --input /tmp/mixed.jsonl --output /tmp/stage_a.jsonl

# train
python -m train build --config examples/config.example.xml
python -m train summary --config examples/config.example.xml
python -m train smoke --config examples/config.example.xml

# tests
pytest
```

## Extension hooks

- Add new XML block types in parser + specs + runtime composition helpers.
- Add teacher backends in `distill/runtime/teacher_runtime.py` backend factory.
- Add scheduler handling in train runtime (`RuntimeTrainConfig`) without coupling to parser internals.
