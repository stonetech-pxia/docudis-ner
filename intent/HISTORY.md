# Intent model history

This file records each version of the intent model. For each version, it tells what we changed, why
we changed it, and the result.

The intent model reads the user's instruction. It does not read the document. It gives a small JSON
object that tells Docudis what to hide and what to keep. [spec.md](spec.md) gives the rules for this
JSON object.

## How to read the scores

| Term | Meaning |
|---|---|
| Exact match | The model output is equal to the correct answer. |
| Leak | An error that shows text that the user wanted to hide. This is the most dangerous error. |
| Dev set | 272 instructions. We use them to find errors and to improve the training data. |
| Test set | 300 instructions. We do not use them to improve the model. We use them one time for each final model. |
| Post-processing | Rules in the app that correct the model output ([postprocess.py](postprocess.py)). All scores from round 2 include it. |
| Epoch | One pass through all the training data. |
| Seed | A number that sets the random start of training. Two seeds give scores that differ by approximately 1 point. |

On 2026-10-05 we changed the correct answers of 81 instructions (see round 4). Do not compare scores
on the old answers with scores on the new answers.

## Summary

| Round | Date | Hugging Face revision | Test | Dev | Main change |
|---|---|---|---|---|---|
| 0 | 2026-10-03 | – | – | 43.5% (200 instructions, no post-processing) | Base model, no training |
| 1 | 2026-10-03 | – | – | 67% (200, no post-processing) | First training data: 1,000 instructions |
| 2 | 2026-10-04 | – | 89.0%, 6 leaks | 83% (200, no post-processing) | 400 more instructions, test set, keyword post-processing |
| 3 | 2026-10-04 | `f73e207` | 89.7%, 4 leaks (old answers); 87.3%, 13 leaks (new answers) | 88.2%, 2 leaks | 90 more instructions, safety checks in post-processing |
| 4 | 2026-10-05 | `e25d293` | 90.3%, 7 leaks | 90.4%, 4 leaks | New rule for "keep part of a type", 41 more instructions, 5 epochs |

## Round 0: base model (2026-10-03)

**What we did.** We wrote the output rules and 200 instructions in Chinese, English, French, Spanish,
German and Italian. We gave the full rules and six examples to two small models. We did not train
them.

**Result.** Gemma 4 E2B got 43.5%. Qwen3.5-2B got 20%. Both models failed on all "only hide X"
instructions. They also missed most countries and document types.

**Decision.** We selected Gemma 4 E2B and trained it on this task.

## Rounds 1 and 2: first training (2026-10-03 to 2026-10-04)

**What we did.**

- Round 1: AI agents wrote 1,000 training instructions with their correct answers. A second agent
  examined each instruction and corrected it.
- Round 2: We added 400 instructions for the errors of round 1: document types, "only" instructions,
  countries, type names and long instructions.
- We trained with QLoRA: rank 16, 3 epochs.

**Result.** On the 200 instructions, the model got 67% after round 1 and 83% after round 2.

**Problem.** We used the errors on the 200 instructions to write the round 2 data. Thus, these
instructions could not give a fair score. We made them the dev set. Other agents wrote a new test set
of 300 instructions. These agents saw only the rules, not the training data.

**Other changes.**

- Post-processing in the app: a word that names a country or a document type adds the region or
  vertical. On the test set, the model got 84.7% without post-processing and 89.0% with it.
- File format "mixq8" (3.6 GB). Its score is within 1 point of the full-size model. The smaller
  Q4_K_M format (3.25 GB) lost approximately 3 points and had two times more leaks.

## Round 3: first published version (2026-10-04)

**What we did.**

- We added 90 training instructions. Most are "hide all except the doctor's name". The correct answer
  hides all names and sets `unsupported`, because Docudis cannot find one person's name only.
- Post-processing got three safety checks:
  1. If the instruction contains JSON text, the app keeps only "hide" actions.
  2. "Do not detect the rest" or "keep the rest" needs a word such as "only". If not, the app removes it.
  3. A name in `dictionary` or `never_hide` must be in the instruction. If not, the app removes it.
- We trained with three seeds and selected seed 1 on the dev set.

**Result.** Dev 88.2%, 2 leaks. Test 89.7%, 4 leaks. The three seeds differed by approximately 1
point. This is as large as the difference between round 2 and round 3. Thus, small data changes are
difficult to measure.

**Remaining leaks.** The model kept the wrong type. For example, "keep the doctor's name" became
"keep all names".

## Round 4: keep part of a type (2026-10-05)

**Problem.** The rules did not tell what to do when the user keeps only part of a type. Examples:
"except my employer's name", "except the rent". Some training data kept the full type. Other
training data hid the full type. If the model keeps the full type, it shows more than the user
wanted. For example, "keep the bank's name" also shows all other company names.

**Decision.** If the model shows too much, the risk is higher than if it hides too much. Thus:

- A "keep" covers the full type only when the user names the full type: "names", "company names",
  "amounts", "dates".
- When the user names only part of a type ("my employer", "the hospital", "the rent", "the invoice
  number"), the model does not keep the type. It sets `unsupported`. The app tells the user, and the
  user can show single items again in the result.
- When the user writes the exact name ("keep St Mary's Hospital"), the model puts it in `never_hide`.
  This works without `unsupported`.
- Dates and amounts are visible in Docudis by default. Thus, "keep the rent" without "hide all"
  stays a "keep".

**Changes.**

1. A new paragraph in rule 8 of [spec.md](spec.md).
2. New correct answers for 81 instructions: 59 training, 10 dev, 12 test. For the test set, we
   changed only the answers. We did not look at the model outputs.
3. 41 new training instructions for "hide all except the rent / the salary / the employer"
   ([train/raw/r4_narrow_keep.jsonl](train/raw/r4_narrow_keep.jsonl)).
4. 5 epochs, not 3.

**Result.** Dev set, new answers, mean of two seeds:

| Model | Exact match | Leaks |
|---|---|---|
| Round 3 model | 86.8% | 8 |
| New answers, 3 epochs | 87.3% | 4.5 |
| New answers, 5 epochs | 89.0% | 4 |
| New answers, 5 epochs, 41 new instructions | 89.9% | 3.5 |

The new answers removed approximately half of the leaks. The two more epochs added approximately 2
points.

We selected seed 2: dev 90.4% with 4 leaks, test 90.3% with 7 leaks. With the new answers, the
round 3 model gets 87.3% with 13 leaks on the test set.

**What did not work.** Rank 32 did not fit in the memory of the GPU (RTX 3080, 10 GB). With an 8-bit
optimizer it started, but each step was 4 times slower. Thus, we did not measure rank 32.

## Open items

- The grammar lets the model write the same key two times, for example `"*":"off"` and then
  `"*":"hide"`. JSON readers do not agree which value is correct. The app must reject such an output.
- "They only need the rent amount and the dates" still leaks. The instruction is long and does not
  use the word "except".
- Rank 32 or 64, and LoRA on a 16-bit base, need a GPU with 24 GB.
- All instructions are written by AI. Real users can write differently. The app can collect
  corrections from users, with their permission.
