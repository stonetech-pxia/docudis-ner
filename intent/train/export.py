# Copyright 2026 the Docudis contributors. Licensed under Apache-2.0.
"""Converts out/merged (finetune.py) to a Q4_K_M GGUF with llama.cpp's tools.

    python intent/train/export.py --llama-src <llama.cpp checkout> --llama-bin <dir with llama-quantize>
"""

import argparse
import subprocess
import sys
from pathlib import Path

OUT = Path(__file__).parent / "out"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama-src", type=Path, required=True)
    ap.add_argument("--llama-bin", type=Path, required=True)
    ap.add_argument("--quant", default="Q4_K_M")
    args = ap.parse_args()

    f16 = OUT / "intent-f16.gguf"
    final = OUT / f"intent-{args.quant}.gguf"
    subprocess.run(
        [sys.executable, str(args.llama_src / "convert_hf_to_gguf.py"), str(OUT / "merged"),
         "--outfile", str(f16), "--outtype", "f16"],
        check=True,
    )
    subprocess.run([str(args.llama_bin / "llama-quantize"), str(f16), str(final), args.quant], check=True)
    f16.unlink()
    print(final)


if __name__ == "__main__":
    main()
