# llm_lab

`llm_lab` is a modular experiment framework for sequence modeling research.

The core components are intentionally swappable:
- codec
- patcher
- mixer
- head
- memory

Experiments are meant to be run by changing configuration values and wiring, rather than rewriting core code paths.
