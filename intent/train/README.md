# Intent fine-tuning

- `BRIEF.md`: what the case writers were asked for (tags, shares, isolation).
- `raw/*.jsonl`: training cases by language, in the eval case format. Each
  file was written by one agent and reviewed and fixed by another, against
  `../spec.md`. None of the writers saw `../eval/`.
- `build.py`: validates `raw/`, drops duplicates and cases close to an eval
  or few-shot instruction (difflib ratio >= 0.8), writes the targets in
  canonical form, and splits 10% per language into `out/dev.jsonl`.
- `finetune.py`: Unsloth QLoRA on `unsloth/gemma-4-E2B-it`, loss on the
  reply only; writes `out/lora` and `out/merged`.
- `export.py`: `out/merged` to a Q4_K_M GGUF with llama.cpp's converter.

The model is trained with the one-line `../system_prompt.txt`, not the full
spec, so evaluate it with `run.py --short-prompt`.

```sh
python intent/train/build.py
<venv>/python intent/train/finetune.py
<venv>/python intent/train/export.py --llama-src <llama.cpp checkout> --llama-bin <llama.cpp binaries>
llama-server -m intent/train/out/intent-Q4_K_M.gguf --port 8080 -ngl 99 -c 2048 --jinja
python intent/eval/run.py --label intent-ft --short-prompt
```

The training venv needs a CUDA build of torch plus `unsloth` and, on
Windows, `triton-windows`. Installing `unsloth` can replace torch with the
CPU build; reinstall the CUDA wheel of the same version afterwards.
