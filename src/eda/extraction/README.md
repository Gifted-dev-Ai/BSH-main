# `src/eda/extraction` — free-text → structured fields

Mines the German shop-floor free-text columns of the incident log
(`Problemdes` + `Moduldescr` + `Errdescr`) into four structured, **bilingual**
columns and writes them to [`data/processed`](../../../data/processed):

| Output column      | Meaning                              | Example |
| ------------------ | ------------------------------------ | ------- |
| `component_de_en`  | Affected module / part               | `Vereinzeler (singulator)` |
| `phenomenon_de_en` | Defect / symptom                     | `verklemmt / blockiert (jammed / blocked)` |
| `action_de_en`     | Corrective action / reaction taken   | `neu eingelegt (re-inserted)` |
| `reason_de_en`     | Trigger / root cause (often empty)   | `nach Wartung (after maintenance)` |

Each cell holds canonical `German (English)` labels; a row can carry several
labels per category, joined with `; `.

## Run

```bash
python -m src.eda.extraction              # -> data/processed/incident_extracted.csv
python -m src.eda.extraction --coverage   # also print the most frequent labels
```

Outputs in `data/processed/`:
- `incident_extracted.csv` — key columns + the four `*_de_en` columns (UTF-8 **with BOM**, so Excel renders the umlauts).
- `incident_extraction_unmatched.csv` — rows where nothing matched (your worklist for extending the lexicon).

## How it works (and how to extend it)

The core is a **deterministic lexicon / rule-based extractor** — chosen over an ML
model because the data is small (178 rows), highly domain-specific, full of typos,
and needs to be auditable for maintenance/quality use.

| File           | Role |
| -------------- | ---- |
| `lexicon.py`   | The curated taxonomy. Each `Term` maps regex `patterns` → a canonical `de`/`en` label, grouped into the four categories. **This is what you edit.** |
| `normalize.py` | Accent-folds + lower-cases + repairs known typos, so patterns are written in plain ASCII (`duse` matches `Düse`). |
| `extractor.py` | `Extractor` interface + `LexiconExtractor` (and `ChainExtractor` for fallbacks). |
| `__main__.py`  | CLI: read raw → extract → write `data/processed` + coverage report. |

**To extend coverage:** open `incident_extraction_unmatched.csv`, find the
unhandled phrase, and add one `Term` (or a pattern to an existing `Term`) in
`lexicon.py`. Re-run. Current coverage: **100 %** of rows match ≥1 category
(component 88 %, phenomenon 82 %, action 49 %, reason 6 % — action/reason are
genuinely absent in most entries).

**Swapping methods:** every extractor implements the same `Extractor.extract()`
interface. To add an LLM- or spaCy-based pass for the long narrative rows, write a
new class with that method and wrap it: `ChainExtractor([LexiconExtractor(), LlmExtractor()])`
fills only the categories the lexicon left empty — no pipeline changes needed.
