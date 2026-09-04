import gc
import json
import os
import random
import resource
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
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


def _select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class MPSCacheClearCallback(TrainerCallback):
    """Clears PyTorch's MPS caching allocator (and runs a Python gc pass) after every
    training step and after every evaluation. Root cause of a real incident: an
    earlier run's swap usage climbed to 25.6/26.6GB and RAM to ~77MB free over ~16
    steps, causing severe swap-thrashing (each step went from ~40s to 15-25 *minutes*)
    - not a sleep/wake artifact (verified no further Wake events, caffeinate was
    active). MPS's allocator doesn't reliably return freed blocks to the OS on its
    own, especially across train/eval calls with differing sequence lengths, so this
    must be forced explicitly. No-op on CUDA/CPU (guarded internally).
    """

    def on_step_end(self, args, state, control, **kwargs):
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()

    def on_evaluate(self, args, state, control, **kwargs):
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        gc.collect()


class MemoryTrackingCallback(TrainerCallback):
    """Tracks peak VRAM (CUDA) and peak process RSS (portable, via resource.getrusage -
    works on macOS/Linux without extra dependencies) across the run. Read via
    .peak_cuda_mb / .peak_rss_mb after training.
    """

    def __init__(self):
        self.peak_cuda_bytes = 0
        self.peak_rss_kb = 0

    def _sample(self):
        if torch.cuda.is_available():
            self.peak_cuda_bytes = max(self.peak_cuda_bytes, torch.cuda.max_memory_allocated())
        self.peak_rss_kb = max(self.peak_rss_kb, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

    def on_step_end(self, args, state, control, **kwargs):
        self._sample()

    def on_evaluate(self, args, state, control, **kwargs):
        self._sample()

    @property
    def peak_cuda_mb(self) -> float | None:
        return round(self.peak_cuda_bytes / 1e6, 1) if torch.cuda.is_available() else None

    @property
    def peak_rss_mb(self) -> float:
        # ru_maxrss is KB on Linux, bytes on macOS - normalize via a platform check.
        import platform

        kb = self.peak_rss_kb / 1024 if platform.system() == "Darwin" else self.peak_rss_kb
        return round(kb / 1024, 1)


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
# hub kernels on macOS 26+, pure-PyTorch fallback otherwise) - verified working on Apple
# Silicon. On CUDA (e.g. a rented cloud GPU) bitsandbytes is the original, most mature
# platform for this - same config works unchanged. 4-bit NF4 quantized frozen base +
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


def load_sample(path: str, n: int | None, seed: int) -> list[dict]:
    """n=None returns every row in the file (still seed-shuffled for a reproducible
    order), used for full-dataset training rather than a pilot subsample.
    """
    with open(path) as f:
        rows = [json.loads(line) for line in f]
    rng = random.Random(seed)
    indices = list(range(len(rows)))
    rng.shuffle(indices)
    if n is not None:
        indices = sorted(indices[:n])
    return [rows[i] for i in indices]


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


def prepare_datasets(tokenizer, n_train: int | None, n_val: int, seed: int) -> tuple[Dataset, Dataset]:
    train_rows = load_sample(TRAIN_PATH, n_train, seed)
    val_rows = load_sample(VAL_PATH, n_val, seed)

    train_examples = [build_training_example(tokenizer, r) for r in train_rows]
    val_examples = [build_training_example(tokenizer, r) for r in val_rows]

    return Dataset.from_list(train_examples), Dataset.from_list(val_examples)


def load_base_model_for_training(resume_adapter_path: str | None = None):
    """resume_adapter_path: if given, loads that adapter's WEIGHTS onto the fresh
    quantized base and continues training it (is_trainable=True). This does NOT
    resume optimizer/scheduler state - see run_training()'s docstring/report field
    for why, and check RESUME_INFO in the returned report before assuming otherwise.
    """
    device = _select_device()
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

    if resume_adapter_path:
        model = PeftModel.from_pretrained(model, resume_adapter_path, is_trainable=True)
    else:
        model = get_peft_model(model, LORA_CONFIG)
    model.print_trainable_parameters()
    return tokenizer, model, device


def detect_overfitting(eval_loss_history: list[float], train_loss_history: list[float]) -> dict:
    """Simple, explainable heuristic (Task 8): flags overfitting if eval_loss's final
    value is clearly above its own minimum while train_loss kept improving. Not a
    substitute for looking at the curves yourself - this just gives a first-pass flag.
    """
    if len(eval_loss_history) < 2:
        return {"overfitting_detected": False, "reason": "fewer than 2 eval checkpoints - not enough signal"}

    min_eval = min(eval_loss_history)
    min_eval_idx = eval_loss_history.index(min_eval)
    final_eval = eval_loss_history[-1]
    is_final_checkpoint_worse = min_eval_idx < len(eval_loss_history) - 1
    relative_degradation = (final_eval - min_eval) / min_eval if min_eval > 0 else 0

    train_still_improving = len(train_loss_history) >= 2 and train_loss_history[-1] < train_loss_history[0]

    flagged = is_final_checkpoint_worse and relative_degradation > 0.10 and train_still_improving
    return {
        "overfitting_detected": flagged,
        "min_eval_loss": round(min_eval, 5),
        "min_eval_loss_at_checkpoint": min_eval_idx + 1,
        "final_eval_loss": round(final_eval, 5),
        "relative_degradation_from_min": round(relative_degradation, 4),
        "train_loss_still_improving": train_still_improving,
        "reason": (
            f"eval_loss degraded {relative_degradation:.1%} from its minimum (checkpoint {min_eval_idx + 1}) "
            "while train_loss kept improving"
            if flagged
            else "no clear overfitting signal"
        ),
    }


def run_training(
    n_train: int | None,
    n_val: int,
    output_dir: str,
    seed: int = 42,
    num_train_epochs: float = 1.0,
    learning_rate: float = 2e-4,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 8,
    eval_steps: int | None = None,
    save_steps: int | None = None,
    save_total_limit: int = 5,
    max_steps: int = -1,
    use_wandb: bool = True,
    run_name: str | None = None,
    resume_adapter_path: str | None = None,
    resume_from_checkpoint: bool | str = False,
) -> dict:
    """n_train=None trains on the ENTIRE train split (Milestone 10's full run) rather
    than a seeded subsample (used for the 150-example pilot).

    resume_adapter_path: continue training from an existing adapter's WEIGHTS (e.g.
    the 150-example pilot). IMPORTANT: this does NOT resume optimizer/scheduler state
    - the 150-pilot was run with save_strategy="no", so no optimizer state exists to
    resume in the first place. A fresh AdamW optimizer and LR schedule are created;
    only the LoRA weights carry forward. Reported explicitly in RESUME_INFO below, per
    the milestone brief's requirement to report this clearly rather than silently.

    resume_from_checkpoint: True (auto-detect latest checkpoint in output_dir) or a
    specific checkpoint path, to resume an INTERRUPTED run of this same call (this DOES
    restore optimizer/scheduler/RNG state, unlike resume_adapter_path above - Trainer's
    own checkpoint mechanism).

    Uses load_best_model_at_end=True (metric: eval_loss) - the returned/saved model is
    the best-validation checkpoint, not necessarily the final step (Task 7).
    """
    tokenizer, model, device = load_base_model_for_training(resume_adapter_path)
    train_ds, val_ds = prepare_datasets(tokenizer, n_train, n_val, seed)

    if eval_steps is None:
        eval_steps = max(1, len(train_ds) // per_device_train_batch_size // gradient_accumulation_steps // 20)
    if save_steps is None:
        save_steps = eval_steps  # load_best_model_at_end requires save cadence to align with eval cadence

    _load_env_file()
    wandb_enabled = use_wandb and bool(os.environ.get("WANDB_API_KEY"))
    if use_wandb and not wandb_enabled:
        print("WANDB_API_KEY not found in environment/.env - continuing without W&B logging.")
    if wandb_enabled:
        import wandb

        run_name = run_name or f"lora-n{len(train_ds)}-r{LORA_CONFIG.r}-lr{learning_rate}"
        wandb.init(
            project=WANDB_PROJECT,
            name=run_name,
            config={
                "model": MODEL_ID,
                "n_train": len(train_ds),
                "n_val": len(val_ds),
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
                "resumed_adapter_from": resume_adapter_path,
                "device": device,
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
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=(device == "cuda"),  # bf16 autocast is CUDA-safe; weights are already bf16-compute via BNB_CONFIG
        report_to="wandb" if wandb_enabled else "none",
        run_name=run_name if wandb_enabled else None,
        remove_unused_columns=False,
    )

    memory_cb = MemoryTrackingCallback()
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=lambda batch: _collate(batch, tokenizer.pad_token_id),
        callbacks=[MPSCacheClearCallback(), memory_cb],
    )

    t0 = time.time()
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    elapsed = time.time() - t0

    wandb_run_url = None
    if wandb_enabled:
        import wandb

        wandb_run_url = wandb.run.url if wandb.run else None
        wandb.finish()

    # trainer.model is now the BEST checkpoint (load_best_model_at_end=True) - this is
    # what gets saved, not necessarily the final training step.
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    log_history = trainer.state.log_history
    train_losses = [e["loss"] for e in log_history if "loss" in e]
    eval_losses = [e["eval_loss"] for e in log_history if "eval_loss" in e]
    learning_rates = [e["learning_rate"] for e in log_history if "learning_rate" in e]
    grad_norms = [e["grad_norm"] for e in log_history if "grad_norm" in e]

    overfitting = detect_overfitting(eval_losses, train_losses)

    return {
        "model": MODEL_ID,
        "method": "QLoRA (4-bit NF4 quantized base, double quant, bf16 compute dtype)",
        "device": device,
        "wandb_run_url": wandb_run_url,
        "resume_info": {
            "resumed_adapter_from": resume_adapter_path,
            "adapter_weights_resumed": resume_adapter_path is not None,
            "optimizer_scheduler_state_resumed": False if resume_adapter_path else None,
            "note": (
                "Adapter WEIGHTS were loaded from resume_adapter_path and continued "
                "training; optimizer (AdamW momentum/variance) and LR scheduler state "
                "were RESET (fresh), not resumed - the source run was saved with "
                "save_strategy='no' so no optimizer state exists to resume from."
                if resume_adapter_path
                else "Trained from a fresh LoRA adapter (get_peft_model), no prior adapter loaded."
            ),
        },
        "checkpoint_resume_used": bool(resume_from_checkpoint),
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
        "eval_steps": eval_steps,
        "save_steps": save_steps,
        "save_total_limit": save_total_limit,
        "total_steps": trainer.state.global_step,
        "training_seconds": round(elapsed, 1),
        "training_hours": round(elapsed / 3600, 3),
        "examples_per_second": round(len(train_ds) * num_train_epochs / elapsed, 3) if elapsed else None,
        "peak_cuda_memory_mb": memory_cb.peak_cuda_mb,
        "peak_process_rss_mb": memory_cb.peak_rss_mb,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric_eval_loss": trainer.state.best_metric,
        "final_train_loss": train_losses[-1] if train_losses else None,
        "final_eval_loss": eval_losses[-1] if eval_losses else None,
        "overfitting_check": overfitting,
        "train_loss_history": train_losses,
        "eval_loss_history": eval_losses,
        "learning_rate_history": learning_rates,
        "grad_norm_history": grad_norms,
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
