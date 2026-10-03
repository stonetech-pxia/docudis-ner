# Brief for writing training cases

The cases fine-tune a small on-device LLM that turns a user's anonymisation
instruction into an intent JSON. The model sees only the instruction, never
the document. `../spec.md` is the contract; `../schema.json` the format.

## Isolation

Do not open anything under `intent/eval/` (the held-out evaluation set and its
few-shot examples). Training cases that copy or paraphrase it make the
evaluation worthless.

## Format

One JSON object per line, UTF-8, no blank lines:

    {"id": "t-zh-001", "lang": "zh", "instruction": "...", "expect": {...}, "tags": ["..."]}

`note` is allowed when an expect needs a justification. Check the file with
`python intent/eval/validate.py <your file>` until it prints "N cases OK".

## Content

- Users are consumers and small businesses in France, Spain, Belgium,
  Switzerland, the UK, the US, Germany, Italy, and Chinese speakers. The
  documents are letters, invoices, payslips, medical records, leases,
  contracts, bank statements, CVs, court and registry notices, insurance
  claims, utility bills, server logs, emails, chat exports.
- Instructions should sound like real people typing or dictating into an
  app: short commands, polite requests, run-on sentences, typos, missing
  accents, backstory before the request, code-switching. No templated
  near-duplicates: vary verbs, word order, register and length.
- Invent all names, companies, projects and identifiers; make them fit the
  locale. Never use real people.
- Follow `spec.md` exactly. Output only what the instruction states or
  clearly implies.

## Tags and minimum share of your cases

A case carries every tag that applies. Shares are minimums; they overlap.

| Tag | Share | What it tests |
|---|---|---|
| type_mapping | 20% | synonyms and local terms per rule 4 (NIR, DNI, NIE, codice fiscale, Steuer-ID, IBAN vs card, policy / invoice / employee numbers -> NUMBER, hospital -> COMPANY, "contact details") |
| vertical | 15% | document domain per rule 7, including "classify the document, not the recipient" and "contract in a domain gets both" |
| combo | 15% | three or more fields at once |
| colloquial | 15% | informal, typos, dictation |
| region | 12% | rule 6, including generic terms that must NOT set a region |
| unsupported | 12% | rule 8, mostly mixed with supported parts; include scoped hides that need the broadest covering hide |
| only | 10% | rule 3 "only X" -> `"*": "off"`, plus some "only" that stays within one type (no `"*"`) |
| negation | 10% | "don't hide", double negatives, "no need to", "don't forget to" |
| long | 10% | two or more sentences of context |
| never_hide | 8% | literal terms to leave visible, including "except <name>" where the name goes to never_hide |
| dictionary | 8% | literal extra terms to hide, including the user's own name |
| dates | 8% | DATE vs BIRTH_DATE |
| except | 6% | "hide everything except X" |
| single_hide / single_keep / single_off | as needed | one type, one action; at least 3% each |
| injection | 4% | commands aimed at the model (rule 10) |
| all | 3% | "hide everything" / "don't hide anything" |
| noop | 3% | nothing actionable (rule 9) |

## Report

When done: per-tag counts computed with a script, and any case where the spec
did not settle the expect (id and the reading you chose).
