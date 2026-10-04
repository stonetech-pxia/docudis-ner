# Intent test set (frozen)

300 instructions in the eval case format, for measuring a model once its
training is settled. `../eval/cases.jsonl` has guided two rounds of training
data and is now a dev set; this one has not. Round 2 (mixq8) scores 84.7%
here, 89.0% with the host's keyword post-processing.

- Written by three agents (zh + mixed, en + de + it, fr + es) that read only
  `../spec.md` and `../schema.json`: not the dev set, not the training data,
  not any error analysis. The mix follows ordinary use (plain requests,
  document context, literal terms, some unsupported, injection and empty
  requests, a quarter informal) rather than known weaknesses.
- Each file was then reviewed and fixed by another agent under the same
  isolation.
- `cases.jsonl` is `raw/*.jsonl` concatenated. `../train/build.py` drops any
  training case close to one of these instructions.

Do not change a case because a model gets it wrong, and do not write training
data from its failures. Score it with:

```sh
python intent/eval/run.py --label <name> --short-prompt --cases intent/test/cases.jsonl
python intent/eval/score.py intent/eval/results/<name>-short-test.jsonl --cases intent/test/cases.jsonl
```
