# Supervised Fine-tuning

This directory contains the ZGCM-1 supervised fine-tuning code,
configurations, launch scripts, and stage notes.

The run closure of this directory is independent of the other top-level
directories; external data, model assets, and the runtime environment are
declared through this directory's configuration.

## Relationship to pretrain's Megatron-LM

`sft/` is a separate fork of Megatron-LM, not a copy of `pretrain/Megatron-LM`.
Both trees contain the same set of files, including the SFT modules such as
`megatron/training/datasets/sft_dataset.py` and `train_rl.py`, but the core
files listed below have different contents. The two trees are therefore not
interchangeable. The fork base commit is not recorded in this repository.

Core files that differ from `pretrain/Megatron-LM`:

- `megatron/core/datasets/gpt_dataset.py`
- `megatron/core/dist_checkpointing/strategies/torch.py`
- `megatron/core/extensions/transformer_engine.py`
- `megatron/core/inference/text_generation_server/run_mcore_engine.py`
- `megatron/core/models/common/language_module/language_module.py`
- `megatron/core/models/gpt/experimental_attention_variant_module_specs.py`
- `megatron/core/optimizer/__init__.py`
- `megatron/core/pipeline_parallel/schedules.py`
- `megatron/core/tokenizers/text/libraries/sft_tokenizer.py`
- `megatron/core/transformer/attention.py`
- `megatron/core/transformer/experimental_attention_variant/dsa.py`
- `megatron/core/transformer/multi_latent_attention.py`
- `megatron/core/transformer/transformer_block.py`
- `megatron/core/transformer/transformer_config.py`
- `megatron/core/utils.py`
- `megatron/training/arguments.py`
- `megatron/training/datasets/data_samplers.py`
- `megatron/training/datasets/sft_dataset.py`
- `megatron/training/training.py`
- `megatron/training/utils.py`
