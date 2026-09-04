import gc
import json
import os
import random
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from src.evaluation.prompts import build_prompt

MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
TRAIN_PATH = "data/training/architect_train.jsonl"
VAL_PATH = "data/training/architect_validation.jsonl"
MAX_SEQ_LENGTH = 1536
WANDB_PROJECT = "architect-ai"


class MPSCacheClearCallback(TrainerCallback):
    """Clears PyTorch's MPS caching allocator (and runs a Python gc pass) after every
    training step and after every evaluation. Root cause of a real incident: an
    earlier run's swap usage climbed to 25.6/26.6GB and RAM to ~77MB free over ~16
    steps, causing severe swap-thrashing (each step went from ~40s to 15-25 *minutes*)
    - not a sleep/wake artifact (verified no further Wake events, caffeinate was
    active). MPS's allocator doesn't reliably return freed blocks to the OS on its
    own, especially across train/eval calls with differing sequence lengths, so this
    must be forced explicitly.
    """

    def on_step_end(self, args, state, control, **kwargs):
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()

    def on_evaluate(self, args, state, control, **kwargs):
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()


def _load_env_file(path: str = ".env") -> None:
    """Loads KEY=VALUE lines from .env into os.environ (only if not already set) -
    wandb/huggingface_hub read their tokens from the environment, and .env is not
    auto-loaded by the shell in a background job.
    """
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if value:
                os.environ.setdefault(key, value)

# QLoRA: bitsandbytes >=0.50 ships a real MPS backend (kernels-community/bitsandbytes-mps
# hub kernels on macOS 26+, pure-PyTorch fallback otherwise) - verified working on this
# machine (Apple M1 Pro, macOS 26.2). Earlier assumption that bitsandbytes was CUDA-only
# was outdated as of that release; corrected here. 4-bit NF4 quantized frozen base +
# LoRA adapters, per the standard QLoRA recipe (Dettmers et al.).
BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

LORA_CONFIG = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)


def load_sample(path: str, n: int, seed: int) -> list[dict]:
    with open(path) as f:
        rows = [json.loads(line) for line in f]
    rng = random.Random(seed)
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    chosen = sorted(indices[:n])
    return [rows[i] for i in chosen]


def build_training_example(tokenizer, row: dict, max_length: int = MAX_SEQ_LENGTH) -> dict:
    """Builds one (input_ids, labels, attention_mask) triple. Loss is masked (-100)
    over the prompt (system+user+generation-prompt tokens) - the model only learns
    to predict the target JSON completion, not to reproduce the prompt.
    """
    messages = build_prompt(row["input"])
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    target_text = json.dumps(row["target"]) + tokenizer.eos_token

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target_text, add_special_tokens=False)["input_ids"]

    input_ids = (prompt_ids + target_ids)[:max_length]
    labels = ([-100] * len(prompt_ids) + target_ids)[:max_length]

    return {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids)}


def _collate(batch: list[dict], pad_token_id: int) -> dict:
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attention_mask = [], [], []
    for b in batch:
        pad = max_len - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_token_id] * pad)
        labels.append(b["labels"] + [-100] * pad)
        attention_mask.append(b["attention_mask"] + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(attention_mask),
    }


def prepare_datasets(tokenizer, n_train: int, n_val: int, seed: int) -> tuple[Dataset, Dataset]:
    train_rows = load_sample(TRAIN_PATH, n_train, seed)
    val_rows = load_sample(VAL_PATH, n_val, seed)

    train_examples = [build_training_example(tokenizer, r) for r in train_rows]
    val_examples = [build_training_example(tokenizer, r) for r in val_rows]

    return Dataset.from_list(train_examples), Dataset.from_list(val_examples)


def load_base_model_for_training():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=BNB_CONFIG,
        device_map={"": device},
    )
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LORA_CONFIG)
    model.print_trainable_parameters()
    return tokenizer, model, device


def run_training(
    n_train: int,
    n_val: int,
    output_dir: str,
    seed: int = 42,
    num_train_epochs: float = 1.0,
    learning_rate: float = 2e-4,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    eval_steps: int | None = None,
    max_steps: int = -1,
    use_wandb: bool = True,
    run_name: str | None = None,
) -> dict:
    tokenizer, model, device = load_base_model_for_training()
    train_ds, val_ds = prepare_datasets(tokenizer, n_train, n_val, seed)

    if eval_steps is None:
        eval_steps = max(1, len(train_ds) // per_device_train_batch_size // gradient_accumulation_steps // 5)

    _load_env_file()
    wandb_enabled = use_wandb and bool(os.environ.get("WANDB_API_KEY"))
    if use_wandb and not wandb_enabled:
        print("WANDB_API_KEY not found in environment/.env - continuing without W&B logging.")
    if wandb_enabled:
        import wandb

        run_name = run_name or f"lora-n{n_train}-r{LORA_CONFIG.r}-lr{learning_rate}"
        wandb.init(
            project=WANDB_PROJECT,
            name=run_name,
            config={
                "model": MODEL_ID,
                "n_train": n_train,
                "n_val": n_val,
                "seed": seed,
                "num_train_epochs": num_train_epochs,
                "learning_rate": learning_rate,
                "per_device_train_batch_size": per_device_train_batch_size,
                "gradient_accumulation_steps": gradient_accumulation_steps,
                "lora_r": LORA_CONFIG.r,
                "lora_alpha": LORA_CONFIG.lora_alpha,
                "lora_dropout": LORA_CONFIG.lora_dropout,
                "lora_target_modules": list(LORA_CONFIG.target_modules),
                "max_seq_length": MAX_SEQ_LENGTH,
            },
        )

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        max_steps=max_steps,
        learning_rate=learning_rate,
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="no",
        bf16=False,  # weights already bf16 via dtype=; avoid Trainer's amp path on MPS
        report_to="wandb" if wandb_enabled else "none",
        run_name=run_name if wandb_enabled else None,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=lambda batch: _collate(batch, tokenizer.pad_token_id),
        callbacks=[MPSCacheClearCallback()],
    )

    t0 = time.time()
    train_result = trainer.train()
    elapsed = time.time() - t0

    wandb_run_url = None
    if wandb_enabled:
        import wandb

        wandb_run_url = wandb.run.url if wandb.run else None
        wandb.finish()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    log_history = trainer.state.log_history
    train_losses = [e["loss"] for e in log_history if "loss" in e]
    eval_losses = [e["eval_loss"] for e in log_history if "eval_loss" in e]

    return {
        "model": MODEL_ID,
        "method": "QLoRA (4-bit NF4 quantized base via bitsandbytes MPS backend, double quant, bf16 compute dtype)",
        "wandb_run_url": wandb_run_url,
        "lora_config": {
            "r": LORA_CONFIG.r,
            "lora_alpha": LORA_CONFIG.lora_alpha,
            "lora_dropout": LORA_CONFIG.lora_dropout,
            "target_modules": list(LORA_CONFIG.target_modules),
        },
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "seed": seed,
        "num_train_epochs": num_train_epochs,
        "learning_rate": learning_rate,
        "per_device_train_batch_size": per_device_train_batch_size,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "effective_batch_size": per_device_train_batch_size * gradient_accumulation_steps,
        "device": device,
        "total_steps": trainer.state.global_step,
        "training_seconds": round(elapsed, 1),
        "training_hours": round(elapsed / 3600, 3),
        "final_train_loss": train_losses[-1] if train_losses else None,
        "final_eval_loss": eval_losses[-1] if eval_losses else None,
        "train_loss_history": train_losses,
        "eval_loss_history": eval_losses,
        "output_dir": output_dir,
    }


if __name__ == "__main__":
    import sys

    n_train = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    n_val = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "models/architect-lora-pilot"

    result = run_training(n_train, n_val, output_dir)
    Path("data/reports/lora_pilot_training_report.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if "history" not in k}, indent=2))
