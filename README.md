# HormoneLab

Open, honestly-evaluated AI infrastructure for women's hormonal health.

Built for Hack-Nation's 6th Global AI Hackathon — Challenge 05: Foundation Models for Women's Hormonal Health
Enabled by OpenAI · Trained on real NHANES (CDC) data · License: MIT

HormoneLab is a set of reproducible models that predict hormone-related outcomes — menopause status, infertility history, hormone therapy use, menstrual irregularity — directly from real government health data, plus a natural-language interface that turns a plain-English symptom description into a prediction and an explanation, in real time.

No inflated accuracy claims. Every result here — including the ones that didn't work — is reported honestly, with the naive baseline shown alongside it.

## The problem

Women's hormonal health remains one of the least represented domains in AI and biomedical research, despite affecting the majority of the population. Conditions like PCOS, endometriosis, and menopause-related disease often take years to diagnose because clinical decisions rely on a single lab snapshot rather than any shared, tested infrastructure. There is no equivalent of ImageNet or AlphaFold for hormonal health — every researcher starts from zero.

## The solution

HormoneLab contributes a working example of what that infrastructure could look like: a documented pipeline that goes from raw NHANES survey and lab data, through label derivation, training, and honest evaluation, to a usable natural-language interface — published openly so the next researcher doesn't have to rebuild any of it.

Describe your situation in plain English. HormoneLab extracts the relevant hormone values, runs a trained model, and explains the result — with a clear "this is not a diagnosis" disclaimer on every response.

**Scope note:** HormoneLab reads lab results you already have (FSH, AMH, LH, or estradiol from a blood test) — it does not diagnose from symptoms alone. If you don't have hormone lab values yet, the honest next step is asking a healthcare provider for a hormone panel.

## See it in action

Type: *"I am 55, my FSH came back at 60, my AMH is very low around 0.05, and I have never taken hormone therapy."*

Get back: `post-menopausal, 97% confidence`, with a plain-language explanation and the extracted values available on request — not forced on you by default.

## Six real experiments, reported honestly

Rather than one model tuned to look good, HormoneLab tests six different hormone-outcome relationships against real NHANES 2017–March 2020 data, and reports what actually worked.

| Model | Population | Best result | Finding |
|---|---|---|---|
| Menopause status | All women | 97.8% accuracy | Barely beats guessing by age alone — the real value is in the honest edge-case limitation, not the headline number |
| Hormone therapy detection | All women | 77.6% recall | Strongest result — a direct, mechanistic hormone relationship |
| Infertility history | Reproductive-age | 61.2% recall | Real partial signal; adding more features did not improve it |
| Menstrual irregularity | Under 40 | 60.0% recall (with insulin/glucose) | Metabolic markers meaningfully improved a weak baseline — evidence for a PCOS-related signal |
| Age-at-menopause (timing) | Post-menopausal | No better than guessing the mean | A single hormone snapshot cannot predict future timing — tested with both linear and non-linear models |
| Premature menopause | Post-menopausal | Not viable | Only 5 true cases in this NHANES cycle — a documented data-volume limit, not a modeling failure |

Full methodology, exact NHANES variable names, and label derivation logic for every model are in `docs/benchmark_methodology.md`.

## Why this matters

Prediction quality across these six experiments tracks cleanly with how mechanistically direct the hormone-outcome relationship is: strong where hormones directly cause the outcome (HRT detection), weaker where the relationship is indirect or multi-causal (infertility, irregularity), and near zero where the task requires information a single snapshot simply doesn't contain (future timing). That pattern — not any single accuracy number — is the real, evidence-backed contribution here.

## How it works

```
free-text symptom description
        │
        ▼
  GPT-4o-mini (extraction)  ──►  structured fields: age, FSH, LH, AMH, estradiol, HRT use
        │
        ▼
  trained model (local, scikit-learn)  ──►  prediction + confidence
        │
        ▼
  GPT-4o-mini (explanation)  ──►  plain-language summary + disclaimer
        │
        ▼
  JSON response  ──►  shown as a conversational reply in the frontend
```

If no OpenAI key is set, the extraction step falls back to a transparent keyword matcher, so the pipeline runs end-to-end either way.

## Project structure

```
womens-hormonal-health/
├── data/                          NHANES source files + cleaned datasets (see data/README.md)
├── docs/
│   └── benchmark_methodology.md   task definitions, exact variable names, label logic, results
├── frontend/
│   └── index.html                 single-file frontend, talks to the backend directly
├── src/
│   ├── train.py                   menopause status model
│   ├── train_fertility.py         infertility history model
│   ├── train_hrt_detection.py     hormone therapy detection model
│   ├── train_irregularity.py      menstrual irregularity model
│   ├── train_menopause_age.py     age-at-menopause regression (negative result)
│   ├── train_premature.py         premature menopause (negative result, documented)
│   ├── app.py                     FastAPI backend — extraction, prediction, explanation
│   └── model*.pkl                 trained model checkpoints, committed for reuse
├── requirements.txt
├── LICENSE
└── README.md
```

## Setup

Install dependencies:

```powershell
pip install -r requirements.txt
```

Download the NHANES source files (see `data/README.md` for exact links), then train any model:

```powershell
python -m src.train                  # menopause status
python -m src.train_fertility        # infertility history
python -m src.train_hrt_detection    # hormone therapy detection
python -m src.train_irregularity     # menstrual irregularity
```

Set your OpenAI key (optional — the app runs without it, using a keyword-based fallback):

```powershell
$env:OPENAI_API_KEY="sk-..."
```

Run the backend:

```powershell
python -m uvicorn src.app:app --reload
```

Open the frontend — double-click `frontend/index.html`, or visit `http://localhost:8000/docs` to test the API directly.

## Tech stack

| Layer | What we used |
|---|---|
| Data | NHANES 2017–March 2020 (CDC), via `pyreadstat` |
| Models | scikit-learn — logistic regression (classification), linear/random forest regression (timing) |
| Natural language | OpenAI GPT-4o-mini — extraction + explanation, with a keyword-based offline fallback |
| Backend | Python, FastAPI, Pydantic, CORS-enabled |
| Frontend | Single-file HTML/CSS/JS — no build step, no framework, talks directly to the FastAPI backend |

## Limitations

This is a research prototype, not a diagnostic tool. Every prediction should be confirmed by a healthcare provider. It requires at least one real hormone lab value to produce a prediction — it will not guess hormonal status from symptoms alone, and says so explicitly when the input is insufficient. Findings are based on US NHANES survey data and may not generalize globally. Full limitations for each model are documented in `docs/benchmark_methodology.md`.

## License

MIT — see `LICENSE`. Models, code, and cleaned datasets are freely reusable.
