# Training material: labels, formats, rules

Material for fine-tuning the on-device NER model (XLM-R, labels `PER`, `ORG`, `LOC`, `DATE`, everything else `O`)
on the text consumers paste into AI assistants: letters, invoices, payslips, medical letters, leases, bank and
insurance letters, CVs, customer-service e-mails, chat messages, forms. Languages: English, French, Spanish.

**Writers of training material never read docudis-android's `benchmark/`, `docs/benchmark/`, `tool/generate_synthetic_cases.py` or
anything they produce.** Those are the test sets. Training text that copies them makes the test worthless.

## 1. What gets which label

| Label | Label it | Leave it `O` |
|---|---|---|
| `PER` | People, in every written form: `Oluwaseun Adebayo-Hart`, `ADEBAYO-HART Oluwaseun`, `Adebayo-Hart, Oluwaseun`, `O. Adebayo-Hart`, `seun`, a first name or surname alone, nicknames, ALL CAPS registry order `FERREIRO CASTIÑEIRA MARÍA LUZ`. | Titles and honorifics (`Mr`, `Mme`, `Dña.`, `Dr`, `Maître`), job titles, roles, departments, the names inside an e-mail address. Common words that are also names when used as words (`will you`, `a rose`, `un petit souci`, `el sofá blanco`). |
| `ORG` | Companies, firms, shops, practices, clinics, hospitals, schools, universities, banks, insurers, associations, clubs, utilities, with or without legal form: `Wrenfield Glazing Ltd`, `Harrowgate Physio Clinic`, `Kay'Styl 21`, `SARL Menuiserie Le Goff`, `ATMR`, firms named after people (`Villanueva Oduya & Partners`). The legal form is part of the value (`Ltd`, `S.L.`, `Inc.`). | Public bodies: tax and social-security agencies, courts, registries, ministries, regulators, ombudsmen, town halls, official gazettes (`HMRC`, `URSSAF`, `CAF de Quimper`, `Agencia Tributaria`, `Tribunal de Commerce`, `Registro Mercantil`). Departments and teams (`Credit Control`, `Service Contentieux`, `Departamento de Recursos Humanos`). Product, plan, tariff and brand names used as products (`Visa`, `Forfait Sérénité`). Generic words (`the Company`, `la société`, `el banco`). |
| `LOC` | A postal address as **one span per line**: `Unit 4, 27 Quarry Lane, Wakefield WF1 3PQ`, `8 impasse des Tanneurs, 29000 Quimper`, `Avda. de Portugal, 112, 5º izda., 37006 Salamanca`. A floor / flat / unit on its own line. Towns, regions and countries standing alone (`Quimper`, `Co. Mayo`, `Castilla y León`), places of birth, "Fait à Quimper", and **the place inside a local public body's name** (`Quimper` in `CAF de Quimper`, `Salamanca` in `Juzgado de Primera Instancia de Salamanca`). | The rest of that body's name. Nationalities, languages, demonyms. Generic places (`the office`, `la clinique`). The jurisdiction in a company's statement of registration (`Registered in England and Wales No. 09417736`): it names where the company was incorporated, not where anyone is. |
| `DATE` | Dates with a day and a month, any format, including dates of birth: `2 March 2025`, `02/03/2025`, `le 9 juin 1979`→`9 juin 1979`, `11 de febrero de 2023`, `3rd oct`, `1er juillet`. | Weekdays, times, bare years, month + year, durations, ages, pay periods written as month + year. |

Phone numbers, e-mails, URLs, IBANs, card numbers, identifiers, reference numbers and amounts are **`O`** (rules
handle them). They must still appear in the text, next to names, so the model learns where a name stops.

Span boundaries: no leading or trailing spaces or punctuation, except the period of an abbreviation that is part of
the name (`Inc.`, `S.L.`, `5º izda.`). A span never contains a line break: if a name or address wraps, each line's
part is its own span.

## 2. Inline markup (documents and negatives)

Write the text as it would be pasted, and wrap each entity: `[[PER|Oluwaseun Adebayo-Hart]]`, `[[ORG|Wrenfield Glazing Ltd]]`,
`[[LOC|Wakefield]]`, `[[DATE|02/03/2025]]`. Nothing else may use `[[` or `]]`. Chat timestamps like `[03/05/2026, 09:14]`
are fine (single brackets). Check: `python training/markup.py --check <files>`.

## 3. Templates

A template is a document with slots that `training/build.py` fills from the inventories. First lines:

```
#country: GB
#doctype: invoice
```

`country` ∈ `GB IE US FR BE CH ES`; `doctype` ∈ `invoice payslip bank insurance lease medical letter cv support chat form`.
Lines starting with `#` are comments. Text outside slots is copied as is; fixed entities in the text use the inline markup.

Slots are `{{KIND}}`, `{{KIND#n}}` or `{{KIND#n:form}}`. `#n` binds a slot to one entity within the document, so
`{{PER#1:full}}` and later `{{PER#1:given}}` are the same person.

| Slot | Forms | Label |
|---|---|---|
| `PER#n` | `full` (build.py varies it: "Given Surname", "SURNAME Given", "Surname, Given", caps), `given`, `surname`, `initial_surname` (`O. Adebayo-Hart`), `surname_given` (registry order), `signature` (as signed) | PER |
| `TITLE#n` | honorific matching `PER#n`'s gender (`Mr`, `Mme`, `Dña.`) | O |
| `ORG#n` | `legal`, `name` (bare), `acronym`; add a sector: `{{ORG#1:legal@trades}}`. Sectors: `trades retail health finance insurance property employer school utility telecom hospitality transport legal` | ORG |
| `LOC#n` | `line` (full one-line address), `street` (number + street), `unit` (flat / floor / door), `postcode_town`, `town`, `region`, `country`, `po_box` | LOC |
| `DATE` | `long`, `short`, `numeric`, `birth`, `day_month`, `iso` | DATE |
| `PUBLIC#n` | a public body, with the town of `LOC#n` inside it when the body is local (`CAF de Quimper`) | O, town LOC |
| `DEPT`, `JOB` | a department / a job title | O |
| `PHONE`, `MOBILE`, `EMAIL:PER#n`, `EMAIL:ORG#n`, `URL:ORG#n` | contact details, derived from the bound entity | O |
| `IBAN`, `BIC`, `CARD4`, `ACCOUNT`, `SORTCODE` | bank details, check digits valid | O |
| `NATID`, `TAXID`, `COMPANYID`, `VAT`, `PLATE`, `REF`, `NUM` | identifiers for the document's country, a document reference, a small number | O |
| `AMOUNT`, `AMOUNT:big` | money in the country's format | O |

Write several people, companies and addresses per template, in the places real documents put them: letterheads,
field blocks (`Name: …` one per line), flattened tables, signature blocks (name, then job or department on the next
line), headers and footers, quoted e-mails, prose. Check: `python training/markup.py --check <files>`.

## 4. Inventories

`training/inventories/<COUNTRY>.json`, one per country, invented or public-domain facts only:

```json
{
 "country": "GB", "lang": "en",
 "given_names": [{"name": "Oluwaseun", "gender": "m", "origin": "west_african"}],
 "surnames": [{"name": "Adebayo-Hart", "origin": "west_african"}],
 "titles": {"m": ["Mr"], "f": ["Ms", "Mrs", "Miss"], "x": ["Mx"]},
 "legal_forms": ["Ltd", "Limited", "LLP", "PLC"],
 "org_names": {"trades": ["Wrenfield Glazing"], "health": ["Harrowgate Physio Clinic"]},
 "acronyms": ["WGS", "ATMR"],
 "streets": ["Quarry Lane", "Tanner's Yard"],
 "units": ["Flat {n}", "Suite {n}", "{n}rd Floor"],
 "places": [{"town": "Wakefield", "postcode": "WF1 3PQ", "region": "West Yorkshire"}],
 "po_box": ["PO Box {n}"],
 "public_bodies": [{"name": "HMRC", "local": false}, {"name": "{town} City Council", "local": true}],
 "departments": ["Credit Control", "Customer Relations"],
 "jobs": ["Practice Manager", "Head of Operations"],
 "email_domains": ["btinternet.com", "gmail.com"]
}
```

Sizes: given names ≥ 200 (both genders, at least a third from immigrant and regional backgrounds common in that
country), surnames ≥ 300, each `org_names` sector ≥ 25 (mixed: descriptive, person-named, invented words, with
apostrophes, digits, hyphens), acronyms ≥ 40, streets ≥ 250, places ≥ 150 real towns with a real postcode of that town,
public bodies ≥ 40, departments ≥ 40, jobs ≥ 80.
