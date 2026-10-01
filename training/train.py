"""Fine-tunes the on-device NER model on training/out/train.jsonl.

    python training/train.py [--out training/out/model] [--epochs 3] [--lr 2e-5] [--batch 16]

Base: Davlan/xlm-roberta-base-ner-hrl (AFL-3.0), full fine-tune, fp16, sequence length 256.
The label order is the one the app expects and must not change:
O, B-DATE, I-DATE, B-PER, I-PER, B-ORG, I-ORG, B-LOC, I-LOC (assets/models/xlmr_ner_hrl/model.json).
Records are cut into windows of 256 tokens with an overlap of 32, the same as the app; only the first
sub-word of a word carries the label. Selection is by entity-level recall on the dev split, not token accuracy.
Needs a CUDA torch plus transformers, datasets and seqeval in a separate conda environment (several GB):
ask the user before installing. Export with training/export_onnx.py.
"""
import argparse
import json
import os

import numpy as np
import torch
from datasets import Dataset
from seqeval.metrics import classification_report, f1_score, recall_score
from transformers import (AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification,
                          Trainer, TrainingArguments)

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = 'Davlan/xlm-roberta-base-ner-hrl'
LABELS = ['O', 'B-DATE', 'I-DATE', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']
LABEL_ID = {l: i for i, l in enumerate(LABELS)}
MAX_LEN, STRIDE = 256, 32


def load(path):
    return [json.loads(line) for line in open(path, encoding='utf-8')]


def encode(records, tokenizer):
    """One example per window; a label goes on the first sub-word of each word."""
    out = {'input_ids': [], 'attention_mask': [], 'labels': []}
    for rec in records:
        enc = tokenizer(rec['text'], return_offsets_mapping=True, truncation=True, max_length=MAX_LEN,
                        return_overflowing_tokens=True, stride=STRIDE)
        for ids, mask, offsets in zip(enc['input_ids'], enc['attention_mask'], enc['offset_mapping']):
            labels = []
            for i, (start, end) in enumerate(offsets):
                if start == end:  # special token
                    labels.append(-100)
                    continue
                span = next((s for s in rec['spans'] if s['start'] < end and s['end'] > start), None)
                if span is None:
                    labels.append(LABEL_ID['O'])
                    continue
                previous = offsets[i - 1] if i else (0, 0)
                inside = previous[0] < span['end'] and previous[1] > span['start'] and previous[0] != previous[1]
                labels.append(LABEL_ID[('I-' if inside else 'B-') + span['label']])
            out['input_ids'].append(ids)
            out['attention_mask'].append(mask)
            out['labels'].append(labels)
    return Dataset.from_dict(out)


def tag_sequences(pred):
    logits, labels = pred[0], pred[1]
    guesses = np.argmax(logits, axis=-1)
    true, said = [], []
    for g_row, l_row in zip(guesses, labels):
        true.append([LABELS[l] for g, l in zip(g_row, l_row) if l != -100])
        said.append([LABELS[g] for g, l in zip(g_row, l_row) if l != -100])
    return true, said


def metrics_fn(pred):
    """Entity level, not token level: a name half hidden is a leak, a token accuracy of 99 % says nothing."""
    true, said = tag_sequences(pred)
    return {'entity_recall': recall_score(true, said), 'entity_f1': f1_score(true, said)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(HERE, 'out'))
    ap.add_argument('--out', default=os.path.join(HERE, 'out', 'model'))
    ap.add_argument('--epochs', type=float, default=3)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--accum', type=int, default=1, help='gradient accumulation; --batch 8 --accum 2 if 10 GB is tight')
    ap.add_argument('--base', default=BASE)
    ap.add_argument('--seed', type=int, default=1)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForTokenClassification.from_pretrained(
        args.base, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id=LABEL_ID)
    if list(model.config.id2label.values()) != LABELS:
        raise SystemExit(f'the base model has a different label order: {model.config.id2label}')

    train = encode(load(os.path.join(args.data, 'train.jsonl')), tokenizer)
    dev = encode(load(os.path.join(args.data, 'dev.jsonl')), tokenizer)
    print(f'{len(train)} training windows, {len(dev)} dev windows')

    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir=args.out, learning_rate=args.lr, num_train_epochs=args.epochs,
                               per_device_train_batch_size=args.batch, per_device_eval_batch_size=args.batch * 2,
                               gradient_accumulation_steps=args.accum, seed=args.seed,
                               warmup_ratio=0.1, weight_decay=0.01, fp16=torch.cuda.is_available(),
                               eval_strategy='epoch', save_strategy='epoch', load_best_model_at_end=True,
                               metric_for_best_model='entity_recall', greater_is_better=True,
                               logging_steps=50, save_total_limit=2, report_to=[], dataloader_num_workers=2),
        train_dataset=train, eval_dataset=dev, data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=metrics_fn)
    trainer.train()
    result = trainer.evaluate()
    prediction = trainer.predict(dev)
    true, said = tag_sequences((prediction.predictions, prediction.label_ids))
    report = classification_report(true, said, digits=3)
    print(report)
    open(os.path.join(args.out, 'dev_report.txt'), 'w', encoding='utf-8', newline='\n').write(report)
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    json.dump(result, open(os.path.join(args.out, 'dev_metrics.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'saved {args.out}')


if __name__ == '__main__':
    main()
