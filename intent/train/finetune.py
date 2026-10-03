# Copyright 2026 the Docudis contributors. Licensed under Apache-2.0.
"""LoRA fine-tune of Gemma 4 E2B-it on out/train.jsonl (build.py), with Unsloth.

    python intent/train/finetune.py [--epochs 3]

Loss is on the assistant reply only (TRL prompt-completion format). Writes the
adapter to out/lora and the merged 16-bit model to out/merged, which
export.py turns into a GGUF.
"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from unsloth import FastModel  # noqa: I001  (must be imported before trl/transformers)
from datasets import Dataset
from trl import SFTConfig, SFTTrainer

HERE = Path(__file__).parent
OUT = HERE / "out"
BASE = "unsloth/gemma-4-E2B-it"


def load(split):
    # Gemma 4's processor is multimodal and wants content as a list of parts.
    rows = []
    for line in (OUT / f"{split}.jsonl").read_text("utf-8").splitlines():
        messages = [
            {"role": m["role"], "content": [{"type": "text", "text": m["content"]}]}
            for m in json.loads(line)["messages"]
        ]
        rows.append({"prompt": messages[:-1], "completion": messages[-1:]})
    return Dataset.from_list(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    args = ap.parse_args()

    model, tokenizer = FastModel.from_pretrained(BASE, max_seq_length=512, load_in_4bit=True)
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=args.rank,
        lora_alpha=args.rank,
        lora_dropout=0,
        bias="none",
        random_state=0,
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=load("train"),
        args=SFTConfig(
            output_dir=str(OUT / "checkpoints"),
            num_train_epochs=args.epochs,
            # 4 x 4: a batch of 8 runs out of the 3080's 10 GB in the cross entropy
            # over Gemma's 262k-token vocabulary.
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=args.lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            weight_decay=0.01,
            logging_steps=10,
            # Evaluating between epochs ran out of memory on the 3080; dev and
            # eval are scored after training with run.py / score.py instead.
            eval_strategy="no",
            save_strategy="no",
            max_length=512,
            completion_only_loss=True,
            seed=0,
            report_to="none",
        ),
    )
    trainer.train()
    model.save_pretrained(str(OUT / "lora"))
    tokenizer.save_pretrained(str(OUT / "lora"))
    model.save_pretrained_merged(str(OUT / "merged"), tokenizer, save_method="merged_16bit")


if __name__ == "__main__":
    main()
