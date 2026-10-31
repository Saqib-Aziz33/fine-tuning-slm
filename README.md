# fine-tuning

A minimal end-to-end example of fine-tuning a Small Language Model (SLM) with **LoRA**.
The base model is [`HuggingFaceTB/SmolLM2-135M-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct),
trained on a small chat dataset (`src/fine_tunning/train.jsonl`) and run entirely on CPU.

## Requirements

- [uv](https://docs.astral.sh/uv/) (all Python dependencies are managed by uv)
- Python 3.14+ (uv can install it for you)
- An internet connection the first time, to download the base model and tokenizer

## 1. Install dependencies

From the project root:

```bash
uv sync
```

This creates a `.venv` and installs everything listed in `pyproject.toml`
(`torch`, `transformers`, `datasets`, `peft`, `accelerate`, `trl`).

## 2. (Optional) Check the base model before training

```bash
uv run python src/fine_tunning/pre-tune-test.py
```

This prints the untouched base model's answers to a few test prompts, so you have a
baseline to compare against after fine-tuning.

## 3. Train the SLM

```bash
uv run python src/fine_tunning/finetune.py
```

What the script does:

1. Loads `SmolLM2-135M-Instruct` in `float32` and forces CPU usage.
2. Wraps it with a LoRA adapter (`r=16`, `lora_alpha=32`, `target_modules="all-linear"`).
3. Reads `src/fine_tunning/train.jsonl` and applies the model's chat template
   (inputs are truncated to 256 tokens).
4. Trains for 30 epochs with a batch size of 2 and gradient accumulation of 2.

Training logs are printed every 5 steps. When it finishes, the LoRA adapter is saved to
`my-finetuned-lora/` in the project root (only the small adapter weights are saved, not
the full base model). The `results/` and `my-finetuned-lora/` directories are ignored by git.

To change the dataset, edit `src/fine_tunning/train.jsonl`. Each line must be a JSON
object with a `messages` list, e.g.:

```json
{"messages": [{"role": "user", "content": "What is the capital of France?"}, {"role": "assistant", "content": "The capital of France is Paris."}]}
```

## 4. Test the fine-tuned model

```bash
uv run python src/fine_tunning/post-tune-test.py
```

The script loads the base model plus the adapter from `my-finetuned-lora/`, then runs the
same prompts and prints, for each one, the **base model response** (adapter disabled) next
to the **fine-tuned model response** (adapter enabled). This makes it easy to see what the
fine-tuning actually changed.

> Make sure step 3 completed successfully first — `post-tune-test.py` requires the
> `my-finetuned-lora/` directory to exist.

## Project layout

```
.
├── pyproject.toml                     # project metadata and dependencies (managed by uv)
├── src/fine_tunning/
│   ├── finetune.py                    # LoRA fine-tuning script
│   ├── pre-tune-test.py               # baseline prompts against the base model
│   ├── post-tune-test.py              # before/after comparison with the adapter
│   └── train.jsonl                    # chat-format training data
├── my-finetuned-lora/                 # generated LoRA adapter (after training)
└── results/                           # generated training output (after training)
```
