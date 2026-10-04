# Copyright 2026 the Docudis contributors. Licensed under Apache-2.0.
"""Converts out/merged (finetune.py) to a GGUF with llama.cpp's tools.

    python intent/train/export.py --llama-src <llama.cpp checkout> --llama-bin <dir with llama-quantize> [--quant mixq8]

The default, mixq8, is Q8_0 with the per-layer embeddings at Q4_K and the token
embeddings at Q8_0 (3.6 GB). The per-layer embeddings are half the model and,
being lookups, lose little at Q4; on the dev set it scores 83.5% against 84.5%
for f16 and 81% for Q4_K_M (3.25 GB), with about half Q4_K_M's leaks. Any
llama-quantize type (Q4_K_M, Q8_0, ...) can be given instead.
"""

import argparse
import subprocess
import sys
from pathlib import Path

OUT = Path(__file__).parent / "out"
MIXQ8 = ["--tensor-type", "per_layer_token_embd=q4_k", "--token-embedding-type", "q8_0"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama-src", type=Path, required=True)
    ap.add_argument("--llama-bin", type=Path, required=True)
    ap.add_argument("--quant", default="mixq8")
    args = ap.parse_args()

    f16 = OUT / "intent-f16.gguf"
    final = OUT / f"intent-{args.quant}.gguf"
    subprocess.run(
        [sys.executable, str(args.llama_src / "convert_hf_to_gguf.py"), str(OUT / "merged"),
         "--outfile", str(f16), "--outtype", "f16"],
        check=True,
    )
    quantize = [str(args.llama_bin / "llama-quantize")]
    if args.quant == "mixq8":
        quantize += [*MIXQ8, str(f16), str(final), "Q8_0"]
    else:
        quantize += [str(f16), str(final), args.quant]
    subprocess.run(quantize, check=True)
    f16.unlink()
    print(final)


if __name__ == "__main__":
    main()
