# Copyright 2026 the Docudis contributors. Licensed under Apache-2.0.
"""Exports the fine-tuned model to the ONNX int8 file the app loads.

    python training/export_onnx.py [--model training/out/model] [--out models/xlmr_ner_docudis]

Runs `optimum-cli export onnx --task token-classification`, quantizes it dynamically to int8, writes
model.json and tokenizer.json next to it, and checks that the graph keeps the names the app uses
(inputs input_ids / attention_mask, output logits) and the label order of the current model. Both the
fp32 and the int8 file are benchmarked separately (docudis-android tool/run_all_benchmarks.sh with --model), and the
int8 graph must be loaded once on the phone before the app switches to it.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS = ['O', 'B-DATE', 'I-DATE', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=os.path.join(ROOT, 'training', 'out', 'model'))
    ap.add_argument('--out', default=os.path.join(ROOT, 'models', 'xlmr_ner_docudis'))
    ap.add_argument('--keep-fp32', action='store_true')
    args = ap.parse_args()
    config = json.load(open(os.path.join(args.model, 'config.json'), encoding='utf-8'))
    order = [config['id2label'][str(i)] for i in range(len(config['id2label']))]
    if order != LABELS:
        raise SystemExit(f'label order changed: {order}')

    os.makedirs(args.out, exist_ok=True)
    tmp = os.path.join(args.out, 'onnx_fp32')
    subprocess.run([sys.executable, '-m', 'optimum.commands.optimum_cli', 'export', 'onnx',
                    '--model', args.model, '--task', 'token-classification', tmp], check=True)

    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantized = os.path.join(args.out, 'model_quantized.onnx')
    quantize_dynamic(os.path.join(tmp, 'model.onnx'), quantized, weight_type=QuantType.QInt8)

    import onnxruntime as ort
    session = ort.InferenceSession(quantized, providers=['CPUExecutionProvider'])
    inputs = {i.name for i in session.get_inputs()}
    outputs = {o.name for o in session.get_outputs()}
    if not {'input_ids', 'attention_mask'} <= inputs or 'logits' not in outputs:
        raise SystemExit(f'the graph does not have the names the app uses: {inputs} -> {outputs}')

    # save_pretrained bakes in whatever truncation the training script used, and the app's Dart tokenizer
    # refuses a stride ("Hugging Face truncation stride is unsupported: 32"). The app windows the text itself.
    tokenizer = json.load(open(os.path.join(args.model, 'tokenizer.json'), encoding='utf-8'))
    tokenizer['truncation'] = None
    tokenizer['padding'] = None
    json.dump(tokenizer, open(os.path.join(args.out, 'tokenizer.json'), 'w', encoding='utf-8', newline='\n'),
              ensure_ascii=False)
    descriptor = {
        'name': 'xlm-roberta-base-ner-docudis',
        'source': 'Fine-tune of Davlan/xlm-roberta-base-ner-hrl (AFL-3.0) on training/out/train.jsonl; '
                  'see training/GUIDE.md and docs/ner-finetune-scope.md',
        'model': 'model_quantized.onnx',
        'tokenizer': {'kind': 'sentencepiece', 'file': 'tokenizer.json'},
        'labels': LABELS,
        'labelMap': {'PER': 'PERSON', 'ORG': 'COMPANY', 'LOC': 'ADDRESS', 'DATE': 'DATE'},
        'maxTokens': 256, 'stride': 32, 'threshold': 0.5, 'padId': 1,
        'inputs': {'ids': 'input_ids', 'mask': 'attention_mask'}, 'output': 'logits',
    }
    json.dump(descriptor, open(os.path.join(args.out, 'model.json'), 'w', encoding='utf-8', newline='\n'),
              ensure_ascii=False, indent=2)
    if not args.keep_fp32:
        shutil.rmtree(tmp, ignore_errors=True)
    size = os.path.getsize(quantized) / 1e6
    print(f'written {args.out} ({size:.0f} MB int8)')
    print('next: in docudis-android, tool/run_all_benchmarks.sh with model=' + os.path.abspath(args.out).replace("\\", "/") +
          ', then the phone benchmark')


if __name__ == '__main__':
    main()
