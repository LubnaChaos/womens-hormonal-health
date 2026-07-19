import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

RANDOM_SEED = 42
DATA_DIR = "data"
MODEL_PATH = "src/model.pkl"

FEATURE_COLUMNS = ["age", "fsh_level", "lh_level", "amh_level", "estradiol_level", "hormone_therapy_use"]
TARGET_COLUMN = "menopause_status"


def _read_xpt(path):
    """Read a NHANES .xpt file. Uses pyreadstat (robust for NHANES) if
    available, else falls back to pandas. Some files (e.g. P_INS) have
    variable labels with non-UTF8 characters like micro signs (µ), which
    pyreadstat can choke on -- fall back to pandas with latin1 encoding
    in that case."""
    try:
        import pyreadstat
        df, _ = pyreadstat.read_xport(path)
        return df
    except UnicodeDecodeError:
        return pd.read_sas(path, encoding="latin1")
    except ImportError:
        return pd.read_sas(path, encoding="latin1")


def load_data():
    demo = _read_xpt(f"{DATA_DIR}/P_DEMO.xpt")
    rhq = _read_xpt(f"{DATA_DIR}/P_RHQ.xpt")
    tst = _read_xpt(f"{DATA_DIR}/P_TST.xpt")

    df = demo.merge(rhq, on="SEQN", how="inner")
    df = df.merge(tst, on="SEQN", how="inner")

    return df


def build_label_and_features(df):
    n_before = len(df)

    def label_row(row):
        if row.get("RHQ031") == 1:
            return 0
        if row.get("RHQ031") == 2:
            reason = row.get("RHD043")
            if reason == 7:
                return 1
            if reason in (1, 2):
                return 0
            if reason == 3:
                if row.get("RHQ305") == 1:
                    return 1
                return None
        return None

    df["menopause_status"] = df.apply(label_row, axis=1)
    df = df.dropna(subset=["menopause_status"])
    n_after_label = len(df)
    print(f"Dropped {n_before - n_after_label} rows without a resolvable menopause label")

    df = df.rename(columns={
        "RIDAGEYR": "age",
        "LBXFSH": "fsh_level",
        "LBXLUH": "lh_level",
        "LBXAMH": "amh_level",
        "LBXEST": "estradiol_level",
    })
    df["hormone_therapy_use"] = (df["RHQ540"] == 1).astype(int)

    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
    n_final = len(df)
    print(f"Dropped {n_after_label - n_final} additional rows with missing feature values. Final sample size: {n_final}")

    output_cols = ["SEQN"] + FEATURE_COLUMNS + [TARGET_COLUMN]
    df[output_cols].to_csv("data/cleaned_menopause_dataset.csv", index=False)
    print(f"Cleaned dataset saved to data/cleaned_menopause_dataset.csv")

    return df


def naive_baseline(df):
    preds = (df["age"] > 51).astype(int)
    return preds


def edge_case_evaluation(df, model):
    baseline_preds = naive_baseline(df)
    disagreement = df[baseline_preds != df["menopause_status"]]
    print(f"\n--- Edge case evaluation ---")
    print(f"Rows where naive age-guess is WRONG: {len(disagreement)} out of {len(df)}")
    if len(disagreement) > 0:
        X_edge = disagreement[FEATURE_COLUMNS]
        y_edge = disagreement[TARGET_COLUMN]
        preds_edge = model.predict(X_edge)
        acc_edge = accuracy_score(y_edge, preds_edge)
        print(f"Model accuracy on these hard cases: {acc_edge:.3f}")
        print(f"True labels in this group: {y_edge.value_counts().to_dict()}")
        print(f"Model predicted:           {pd.Series(preds_edge).value_counts().to_dict()}")


def train_and_evaluate(df):
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    baseline_preds = naive_baseline(df.loc[X_test.index])
    baseline_acc = accuracy_score(y_test, baseline_preds)
    print(f"Naive baseline accuracy: {baseline_acc:.3f}")

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, preds, average="binary")

    print(f"Model accuracy: {acc:.3f}")
    print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, preds))

    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    edge_case_evaluation(df.loc[X_test.index], model)

    return model


def predict(features: dict):
    model = joblib.load(MODEL_PATH)
    X = pd.DataFrame([features])[FEATURE_COLUMNS]
    label = model.predict(X)[0]
    confidence = model.predict_proba(X).max()
    return ("post-menopausal" if label == 1 else "pre-menopausal"), float(confidence)


if __name__ == "__main__":
    df = load_data()
    df = build_label_and_features(df)
    train_and_evaluate(df)