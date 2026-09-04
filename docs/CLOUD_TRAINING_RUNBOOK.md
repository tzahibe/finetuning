# Milestone 10 — Full Training on a Cloud GPU (RunPod)

Local M1 Pro/MPS measured rate: ~41.6s/example → full 54,044-example TRAIN
split would take ~26 days. Not practical locally. This runbook moves the
full training run to a rented CUDA GPU.

## 1. Provider: RunPod

Recommended over Vast.ai/Lambda for this job: simple web UI, ~5 min from
signup to a running pod, Secure Cloud tier has a 99% uptime SLA, RTX 4090
(24GB) at roughly $0.34-0.69/hr. A 7B QLoRA fine-tune like this typically
costs **$3-20 total** end to end.

1. Sign up at https://runpod.io, add a payment method.
2. **Deploy a Pod** → GPU: **RTX 4090** (24GB is comfortably enough for a
   4-bit-quantized 7B base + LoRA adapters; no need for anything bigger/pricier).
3. Template: search for **"RunPod PyTorch"** (has CUDA + PyTorch preinstalled) -
   avoids a slow from-scratch CUDA toolkit install.
4. Storage: 30-40GB container disk is enough (base model ~15GB download,
   dataset, checkpoints).
5. Deploy, then open the pod's **Web Terminal** (or connect via SSH - RunPod
   shows the SSH command on the pod's page once it's running).

## 2. Set up the code on the pod

```bash
git clone https://github.com/tzahibe/finetuning.git
cd finetuning
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Create `.env` (never commit this - it's gitignored):
```bash
cat > .env << 'EOF'
HF_TOKEN=<your token, write-scoped>
HF_REPO_ID=tzachi2222/architect-ai-boomi
WANDB_API_KEY=<your wandb key>
EOF
```

Pull the training data (the dataset is on the Hub, already pushed - see
`data/training/README.md`) and the 150-example pilot adapter (to resume
from it, per Milestone 10 requirement #2):

```bash
# dataset files already live under data/training/*.jsonl in the repo? check first -
# if data/training/*.jsonl are gitignored (they are, ~360MB), pull from HF instead:
python -c "
from datasets import load_dataset
ds = load_dataset('tzachi2222/architect-ai-boomi')
for split in ds:
    ds[split].to_json(f'data/training/architect_{split}.jsonl', orient='records', lines=True)
    # re-flatten input/target back into the {'input':...,'target':...} shape if needed
"

# pull the 150-example pilot adapter to resume from
python -c "
from huggingface_hub import snapshot_download
snapshot_download('tzachi2222/architect-ai-qlora-150', local_dir='models/architect-lora-pilot', token=open('.env').read().split('HF_TOKEN=')[1].split()[0])
"
```

(If the JSON re-flattening from the HF dataset doesn't exactly match
`{"input": {...}, "target": {...}}`, it's simpler to `scp`/`rsync` your local
`data/training/architect_{train,validation,test}.jsonl` files directly to
the pod instead - they're already in the right shape.)

## 3. Verify before committing to the full run

**Do this every time, on new hardware - it caught two real bugs locally
(a transformers API quirk, a severe MPS memory leak) that only showed up
under actual execution, not code review:**

```bash
python -c "
from src.training.train_lora import run_training
result = run_training(
    n_train=50, n_val=10,
    output_dir='/tmp/cuda_smoke_test',
    gradient_accumulation_steps=1,
    max_steps=8,
    eval_steps=4,
    use_wandb=False,
)
import json
print(json.dumps({k:v for k,v in result.items() if 'history' not in k}, indent=2))
"
```

Check: no errors, loss decreasing, note the real `examples_per_second` /
per-step timing this GPU gives you - use it to sanity-check the full-run
ETA before launching it (should be far faster than the ~41.6s/example seen
on MPS; even a few seconds/example brings the full run under a day).

## 4. Launch the full run

```bash
python -c "
from src.training.train_lora import run_training
import json
result = run_training(
    n_train=None,  # entire TRAIN split (54,044 examples)
    n_val=500,     # a validation subset is enough for monitoring - the full
                   # 6,756-example validation split would itself take a long
                   # time to eval every checkpoint; increase later if wanted
    output_dir='models/architect-lora-full',
    num_train_epochs=1.0,
    learning_rate=2e-4,
    per_device_train_batch_size=4,       # a 24GB GPU can likely go higher than
    gradient_accumulation_steps=4,        # the MPS pilot's batch=1 - raise if
                                           # the smoke test's VRAM headroom allows
    save_total_limit=5,
    run_name='qlora-full-train',
    resume_adapter_path='models/architect-lora-pilot',  # continue from the 150-pilot
)
Path('data/reports/lora_full_training_report.json').write_text(json.dumps(result, indent=2))
print(json.dumps({k:v for k,v in result.items() if 'history' not in k}, indent=2))
" 2>&1 | tee training_log.txt
```

Run this inside `tmux` or `screen` so it survives an SSH disconnect:
```bash
tmux new -s training
# ... run the command above inside tmux ...
# detach: Ctrl+B then D
# reattach later: tmux attach -t training
```

Monitor live at https://wandb.ai/tzachi2222-/architect-ai (run name
`qlora-full-train`).

## 5. If interrupted

Checkpoints save every `eval_steps` (default: total_steps/20) under
`models/architect-lora-full/checkpoint-<N>`, keeping the 5 most recent plus
the best-eval-loss one. Resume with:

```python
result = run_training(..., resume_from_checkpoint=True)  # auto-detects the latest checkpoint
```

## 6. When training finishes

Copy results back for the comparison report:
```bash
scp -r <pod>:finetuning/models/architect-lora-full ./models/
scp <pod>:finetuning/data/reports/lora_full_training_report.json ./data/reports/
```

Then continue exactly as after the 150-example pilot: run the same 20 fixed
TEST examples through `models/architect-lora-full`, compare BASE vs 150 vs
FULL, per Milestone 10's requirements #9-13. **Do not merge/publish the
model automatically** - stop and return the comparison report first.

## 7. Cost control

- **Stop the pod** (not just disconnect) the moment training finishes or if
  you're pausing - RunPod bills by the second while running.
- RunPod's Secure Cloud pods can be stopped and later resumed without
  losing the container disk (checkpoints stay); Community Cloud pods may
  not offer this - check before choosing an instance type.
