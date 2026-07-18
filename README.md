# Menopause Status Predictor — Hack-Nation 6th Global AI Hackathon

**Challenge:** 05 — Foundation Models for Women's Hormonal Health (OpenAI × Hack-Nation)
**Track:** 03 — Application Infrastructure (built on a Track 02–style trained model)

## Problem

Hormonal conditions like the menopause transition often take years to recognize
because there is no shared, tested AI infrastructure for women's hormonal health —
unlike fields such as imaging or genomics. Clinical decisions rely on isolated
lab snapshots rather than pattern recognition across signals that are already
being collected in routine health data.

## What this project does

1. Trains a reproducible baseline model that predicts menopause status
   (pre- vs. post-menopausal) from a small set of routine health features,
   using the NHANES dataset.
2. Wraps that model in a natural-language interface: a user describes their
   symptoms in plain English, GPT extracts structured features, the model
   predicts a status with a confidence score, and GPT explains the result
   in plain language — with a clear "this is not a diagnosis" disclaimer.
3. Publishes the model, evaluation results, and methodology as a reusable
   benchmark for future researchers.

## Repository structure

```
womens-hormonal-health/
├── data/               # NHANES source files + processed dataset (see data/README.md)
├── src/
│   ├── train.py        # data loading, cleaning, training, evaluation
│   ├── model.pkl        # trained model checkpoint (committed for reuse)
│   └── app.py           # FastAPI app: text -> extraction -> model -> explanation
├── notebooks/           # exploratory work, evaluation plots
├── docs/
│   └── benchmark_methodology.md   # task definition, split, metrics — for reuse by others
├── requirements.txt
├── LICENSE
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
export OPENAI_API_KEY="your-key-here"
```

## Reproducing the model

```bash
python src/train.py
```

Outputs `src/model.pkl` and prints evaluation metrics (accuracy vs. naive
baseline, precision/recall, subgroup breakdown).

## Running the app

```bash
uvicorn src.app:app --reload
```

Then open `http://localhost:8000/docs` and try the `/predict` endpoint with
a free-text symptom description.

## Results

_Fill in after training: accuracy, comparison to naive baseline, subgroup
performance, feature importance findings._

## Limitations

- NHANES is a US-only, self-reported survey; findings may not generalize
  globally.
- This is a research prototype, not a diagnostic tool. Every prediction is
  accompanied by a disclaimer to consult a healthcare provider.
- Subgroup sample sizes vary; results for small subgroups should be treated
  as low-confidence.

## Reach & quality of life

_Fill in: how many women could this kind of tool reach, and how would it
meaningfully improve their experience getting a hormonal-health signal
earlier?_

## License

MIT — see LICENSE. Model checkpoints and code are freely reusable.
