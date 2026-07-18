# Benchmark Methodology

This document defines the task as a reusable benchmark so future teams can
train their own models and compare fairly against this baseline.

## Task

**Binary classification**: predict whether a respondent is pre-menopausal
or post-menopausal, using indirect signals (age, FSH lab value, hormone
therapy history, smoking status) rather than the direct self-report question
used to derive the label itself.

## Data

- Source: NHANES 2017–March 2020 cycle (`P_RHQ`, `P_DEMO`, thyroid lab file)
- Population: female respondents with non-missing values for all selected
  features and the target label
- Sample size after cleaning: _fill in_

## Label derivation

Derived from two real NHANES questions (P_RHQ):
- `RHQ031` — "Have you had at least one menstrual period in the past 12 months?" (1=Yes, 2=No)
- `RHD043` — asked only if RHQ031=2 — reason for no period (1=Pregnancy, 2=Breastfeeding, 3=Hysterectomy, 7=Menopause/Change of life, 9=Other)

```
RHQ031 == 1                      -> pre-menopausal
RHQ031 == 2 and RHD043 == 7      -> post-menopausal
otherwise                        -> excluded (pregnancy, breastfeeding,
                                     hysterectomy, or unclear reason are
                                     not natural menopause status)
```

## Features

| Feature | Source variable | Type |
|---|---|---|
| age | DEMO (RIDAGEYR) | continuous |
| fsh_level | P_TST (LBXFSH) | continuous |
| hormone_therapy_use | P_RHQ (RHQ540) | binary |

## Split

- Train/test split: 80/20
- Random seed: 42 (fixed for reproducibility)
- Stratified by target label to preserve class balance

## Metrics

- Accuracy (vs. naive age-threshold baseline)
- Precision, recall (per class)
- Confusion matrix
- Subgroup accuracy by race/ethnicity and age band (sample-size caveated)

## Baseline result

_Fill in after training: naive baseline accuracy vs. logistic regression
accuracy._

## How to compare a new model against this benchmark

1. Use the same train/test split (seed=42, same feature columns)
2. Report the same metrics above
3. State any additional features used, and why

## Known limitations

- Binary framing collapses the peri-menopausal transition, which is
  clinically the hardest and most ambiguous stage to define.
- Self-reported data; NHANES is US-only and may not generalize globally.
- Small subgroup sizes for some demographic breakdowns.
