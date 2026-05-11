# NLP Project: Speculative Decoding Drafter Distillation

This project generates distillation data from a large target model, fine-tunes a
small draft model with KL distillation, and evaluates speculative decoding
acceptance against the target model.

The default setup uses:

- Target model: `Qwen/Qwen3-8B`
- Base drafter: `Qwen/Qwen3-0.6B`
- Domains: code, math, translation
- Dataset format: JSONL rows containing prompts, teacher responses, target token
  ids, and target top-k log probabilities

Run commands from the repository root.

## Repository Files

### `dataset-generation/dataset-generation.py`

Generates distillation JSONL files for code, math, and translation.

Classes:

- `DomainLimits`: per-domain prompt, response, and token-budget limits.
- `CheckpointManager`: saves and restores generation state, prompt caches, and
  raw output paths.
- `TokenCounter`: wraps a Hugging Face tokenizer for length filtering, with a
  rough fallback counter.

Functions:

- `stable_hash`: builds deterministic SHA-256 hashes for deduplication.
- `normalize_text`: trims and normalizes whitespace.
- `get_nested`: reads normal or nested fields from dataset rows.
- `load_hf_dataset`: loads Hugging Face datasets, including streaming datasets.
- `shuffled_iter`: returns a shuffled iterator for streaming or in-memory data.
- `clean_docstring`: strips Python triple-quote markers from docstrings.
- `is_bad_csn_function`: filters unusable CodeSearchNet Python functions.
- `extract_function_head`: extracts a function signature block.
- `choose_partial_body`: cuts a Python function body to create a completion
  prompt.
- `build_code_prompt`: creates a code teacher prompt.
- `load_code_prompts`: loads and filters CodeSearchNet prompts.
- `is_good_math_problem`: filters math source problems.
- `is_good_translation_source`: filters translation source sentences.
- `load_simple_text_prompts`: loads math prompts from configured datasets.
- `load_translation_prompts`: loads balanced English-French and French-English
  prompts.
- `load_domain_prompts`: dispatches prompt loading by domain.
- `serialize_generation_logprobs`: converts vLLM logprob objects to JSON-safe
  top-k rows.
- `run_teacher_inference`: runs target-model generation with vLLM and writes raw
  JSONL rows.
- `check_text`: returns a stripped response for filtering checks.
- `has_repeated_line_loop`: detects repeated-line degeneration.
- `has_artifact_marker`: detects obvious generated artifact markers.
- `is_bad_response`: rejects unusable or externally truncated responses.
- `clean_response`: truncates accepted responses at domain-specific stop
  patterns.
- `filter_outputs`: writes final filtered distillation JSONL files.
- `print_dataset_stats`: prints dataset size and length statistics.
- `main`: parses CLI arguments and runs prompt loading, inference, filtering,
  and stats.

### `dataset-generation/setup.sh`

Installs the dataset-generation runtime on a RunPod-style GPU machine. It sets
Hugging Face, Torch, and pip caches under `/workspace`, installs `vllm==0.19.0`,
installs PyTorch with CUDA 12.8 wheels, installs dataset utilities, and verifies
the environment.


### `finetuning/prepare_eval_split.py`

Splits final distillation data into deterministic train and evaluation sets.

Functions:

- `split_domain`: shuffles one domain JSONL file and writes train/eval JSONL
  files.
- `main`: parses CLI arguments and processes all available domains.

### `finetuning/finetune_draft.py`

Fine-tunes the draft model with KL loss against stored teacher top-k token
distributions.

Classes:

- `KLDistillationDataset`: tokenizes prompts and attaches target top-k
  distributions.
- `KLCollator`: pads input ids, labels, attention masks, and top-k tensors.
- `KLTrainer`: Hugging Face `Trainer` subclass that computes KL loss over target
  top-k distributions, with CE fallback when needed.

Functions:

- `resolve_domains`: expands `all` into the supported domain list.
- `load_data`: loads and samples train split rows by domain.
- `parse_top_logprobs`: extracts target token ids and top-logprob entries from a
  JSONL row.
- `find_latest_checkpoint`: finds the newest `checkpoint-*` directory.
- `main`: parses CLI arguments, builds tokenizer/model/dataset/trainer, resumes
  from checkpoint when available, and saves `final/`.

### `finetuning/measure_acceptance.py`

Measures vLLM native speculative decoding with either the base drafter,
fine-tuned drafter, or domain-routed drafters.

Functions:

- `_stop_sequences_for_domain`: returns stop sequences for each domain.
- `_load_spec_llm`: creates a vLLM speculative decoding engine.
- `_unload_llm`: frees GPU memory after a run.
- `load_eval_prompts`: loads held-out prompts for one domain.
- `clean_fasttext_text`: normalizes prompt text for fastText.
- `classifier_predict`: predicts domains with a fastText classifier.
- `_get_nested_attr`: reads nested attributes or dict keys.
- `_first_number`: extracts metric counters from vLLM outputs.
- `extract_request_metrics`: records generated, accepted, draft token, and finish
  metrics for one request.
- `run_vllm_speculative`: runs speculative decoding and returns metrics plus
  runtime.
- `_fmt`: formats metric values.
- `_delta`: formats metric deltas.
- `compare_and_print`: compares two result JSON files.
- `aggregate`: computes per-domain and overall acceptance statistics.
- `main`: parses CLI arguments, runs measurement, writes JSON, or compares
  previous results.

### `finetuning/train_domain_classifier.py`

Trains a fastText prompt-domain classifier.

Functions:

- `clean_fasttext_text`: normalizes prompts into one-line fastText input.
- `find_domain_file`: finds domain JSONL files in train split or distillation
  directories.
- `load_prompts`: loads non-empty prompt strings from JSONL.
- `write_fasttext_file`: writes `__label__<domain>` fastText training rows.
- `main`: trains and saves the fastText classifier.

### `finetuning/test_domain_classifier.py`

Evaluates the fastText prompt-domain classifier.

Functions:

- `clean_fasttext_text`: normalizes prompts into one-line fastText input.
- `find_domain_file`: finds domain JSONL files in eval split or distillation
  directories.
- `load_prompts`: loads non-empty prompt strings from JSONL.
- `main`: loads the classifier, evaluates per-domain and overall accuracy, and
  prints a confusion matrix.

### `finetuning/setup.sh`

Installs the fine-tuning and evaluation runtime. It installs `vllm==0.19.0`,
PyTorch CUDA 12.8 wheels, Transformers, Accelerate, PEFT, bitsandbytes,
datasets, Hugging Face transfer helpers, sentencepiece, and fastText.

### `finetuning/WORKFLOW.md`

Detailed workflow for splitting data, measuring the base drafter, fine-tuning,
resuming training, measuring the fine-tuned drafter, and comparing results.

## Data Files

The expected generated data layout is:

```text
datasets/distillation_data/
  code_distillation.jsonl
  math_distillation.jsonl
  translation_distillation.jsonl
  *_distillation_raw.jsonl

datasets/train_split/
  code_train.jsonl
  math_train.jsonl
  translation_train.jsonl

datasets/eval_split/
  code_eval.jsonl
  math_eval.jsonl
  translation_eval.jsonl
```

Checkpoints are written under `checkpoints/`.

## Software Requirements

Recommended GPU environment:

- OS/container: Ubuntu 22.04, RunPod `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
- GPU: NVIDIA A100 80GB recommended for `Qwen/Qwen3-8B` target evaluation
- Python: 3.11
- CUDA runtime/wheels: CUDA 12.8 PyTorch wheels
- PyTorch: installed from `https://download.pytorch.org/whl/cu128`
- vLLM: `0.19.0`
- Transformers: `>=4.45`
- Accelerate: `>=1.0`
- PEFT: `>=0.12` for LoRA training
- bitsandbytes: latest pip version
- datasets: `>=3.6.0`
- huggingface_hub: `>=0.30.0`
- hf_transfer: `>=0.1.8`
- sentencepiece: `>=0.2.0`
- fastText: `>=0.9.3`
- torchvision and torchaudio matching the installed PyTorch CUDA build

You also need a Hugging Face account/token with access to the Qwen models:

```bash
huggingface-cli login
```

Install the main runtime:

```bash
bash finetuning/setup.sh
```

For only dataset generation, this also works:

```bash
bash dataset-generation/setup.sh
```

## Commands

### 1. Generate a small test dataset

```bash
python dataset-generation/dataset-generation.py \
  --samples_per_domain 10 \
  --model_name Qwen/Qwen3-8B \
  --batch_size 8 \
  --output_dir datasets/test_distillation_data \
  --domains all \
  --checkpoint_every 5
```

### 2. Generate full distillation data

```bash
python dataset-generation/dataset-generation.py \
  --samples_per_domain 80000 \
  --model_name Qwen/Qwen3-8B \
  --batch_size 64 \
  --output_dir datasets/distillation_data \
  --domains all \
  --dtype bfloat16 \
  --gpu_memory_utilization 0.85 \
  --max_model_len 2048 \
  --temperature 0.0 \
  --top_logprobs 64 \
  --checkpoint_every 500 \
  --oversample_factor 1.2
```

To restart generation instead of resuming:

```bash
python dataset-generation/dataset-generation.py \
  --samples_per_domain 80000 \
  --output_dir datasets/distillation_data \
  --domains all \
  --fresh
```

### 3. Create train/eval splits

```bash
python finetuning/prepare_eval_split.py \
  --data_dir datasets/distillation_data \
  --output_dir datasets \
  --eval_pct 10 \
  --seed 1337
```

### 4. Train the baseline domain classifier

This trains a fastText prompt classifier used for optional domain-routed draft
model selection.

```bash
python finetuning/train_domain_classifier.py \
  --data_dir datasets/train_split \
  --output_model checkpoints/domain_classifier/domain_classifier.bin \
  --samples_per_domain 500 \
  --domains code math translation \
  --epoch 25 \
  --lr 0.5 \
  --word_ngrams 2 \
  --seed 42
```

Test it:

```bash
python finetuning/test_domain_classifier.py \
  --model checkpoints/domain_classifier/domain_classifier.bin \
  --data_dir datasets/eval_split \
  --samples_per_domain 100 \
  --domains code math translation \
  --seed 123
```

### 5. Test the base speculative decoding baseline

This is the main baseline: unmodified `Qwen/Qwen3-0.6B` drafting for
`Qwen/Qwen3-8B`.

```bash
python finetuning/measure_acceptance.py \
  --drafter Qwen/Qwen3-0.6B \
  --target Qwen/Qwen3-8B \
  --eval_dir datasets/eval_split \
  --domains code math translation \
  --num_speculative_tokens 5 \
  --max_eval_per_domain 500 \
  --output_json results/base.json
```

### 6. Train a fine-tuned drafter

Single-domain example:

```bash
python finetuning/finetune_draft.py \
  --domains code \
  --num_samples 50000 \
  --train_dir datasets/train_split \
  --output_dir checkpoints/code_50k \
  --model_name Qwen/Qwen3-0.6B \
  --epochs 1 \
  --batch_size 8 \
  --grad_accum 4 \
  --lr 2e-5 \
  --max_seq_len 1024 \
  --top_k 64 \
  --save_steps 500 \
  --seed 42
```

All-domain example:

```bash
python finetuning/finetune_draft.py \
  --domains all \
  --num_samples 30000 \
  --train_dir datasets/train_split \
  --output_dir checkpoints/all_30k \
  --model_name Qwen/Qwen3-0.6B \
  --epochs 1 \
  --batch_size 8 \
  --grad_accum 4 \
  --lr 2e-5 \
  --max_seq_len 1024 \
  --top_k 64 \
  --save_steps 500 \
  --seed 42
```

LoRA example:

```bash
python finetuning/finetune_draft.py \
  --domains code \
  --num_samples 50000 \
  --train_dir datasets/train_split \
  --output_dir checkpoints/code_50k_lora \
  --model_name Qwen/Qwen3-0.6B \
  --epochs 1 \
  --batch_size 8 \
  --grad_accum 4 \
  --lr 2e-5 \
  --max_seq_len 1024 \
  --top_k 64 \
  --save_steps 500 \
  --seed 42 \
  --lora
```

The training script automatically resumes from the latest
`checkpoint-*` directory in `--output_dir`. Add `--fresh` only when you want to
delete the existing output directory and restart.

### 7. Test the fine-tuned drafter

```bash
python finetuning/measure_acceptance.py \
  --drafter checkpoints/code_50k/final \
  --target Qwen/Qwen3-8B \
  --eval_dir datasets/eval_split \
  --domains code math translation \
  --num_speculative_tokens 5 \
  --max_eval_per_domain 500 \
  --output_json results/code_50k.json
```

### 8. Test domain-routed drafters

Use the prompt classifier to choose a drafter by predicted domain:

```bash
python finetuning/measure_acceptance.py \
  --drafter Qwen/Qwen3-0.6B \
  --target Qwen/Qwen3-8B \
  --eval_dir datasets/eval_split \
  --domains code math translation \
  --num_speculative_tokens 5 \
  --max_eval_per_domain 500 \
  --prompt_classifier checkpoints/domain_classifier/domain_classifier.bin \
  --code_drafter checkpoints/code_50k/final \
  --math_drafter Qwen/Qwen3-0.6B \
  --translation_drafter Qwen/Qwen3-0.6B \
  --output_json results/routed.json
```

### 9. Compare test results

```bash
python finetuning/measure_acceptance.py \
  --compare results/base.json results/code_50k.json
```

Or compare the baseline against a routed system:

```bash
python finetuning/measure_acceptance.py \
  --compare results/base.json results/routed.json
```

## Running Long Jobs

Use `tmux` or another terminal multiplexer for GPU jobs:

```bash
tmux new -s nlp-project
```

Detach with `Ctrl+B`, then `D`. Reattach with:

```bash
tmux attach -t nlp-project
```

## Expected Outputs

- Dataset generation writes `<domain>_distillation_raw.jsonl` and
  `<domain>_distillation.jsonl`.
- Split preparation writes `datasets/train_split/*_train.jsonl` and
  `datasets/eval_split/*_eval.jsonl`.
- Fine-tuning writes checkpoints under `checkpoints/<run_name>/` and the final
  model under `checkpoints/<run_name>/final/`.
- Acceptance measurement writes JSON summaries such as `results/base.json`,
  `results/code_50k.json`, and `results/routed.json`.

