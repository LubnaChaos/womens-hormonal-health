import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from src.train import _read_xpt, DATA_DIR
RANDOM_SEED = 42
MODEL_PATH = "src/model_premature.pkl"

FEATURE_COLUMNS = ["fsh_level", "lh_level", "amh_level", "estradiol_level", "hormone_therapy_use"]
TARGET_COLUMN = "premature_menopause"

PREMATURE_AGE_THRESHOLD = 40


def load_and_label():
    demo = _read_xpt(f"{DATA_DIR}/P_DEMO.xpt")
    rhq = _read_xpt(f"{DATA_DIR}/P_RHQ.xpt")
    tst = _read_xpt(f"{DATA_DIR}/P_TST.xpt")

    df = demo.merge(rhq, on="SEQN", how="inner")
    df = df.merge(tst, on="SEQN", how="inner")

    def menopause_label(row):
        if row.get("RHQ031") == 1:
            return 0
        if row.get("RHQ031") == 2:
            reason = row.get("RHD043")
            if reason == 7:
                return 1
            if reason in (1, 2):
                return 0
            if reason == 3 and row.get("RHQ305") == 1:
                return 1
        return None

    df["menopause_status"] = df.apply(menopause_label, axis=1)
    df = df.dropna(subset=["menopause_status"])

    df = df[df["menopause_status"] == 1].copy()
    n_menopausal = len(df)
    print(f"Post-menopausal women identified: {n_menopausal}")
    print("Note: true clinical premature menopause is rare (~1% of women); "
      "expect very few positive cases in a single NHANES cycle.")

    df = df.rename(columns={
        "LBXFSH": "fsh_level",
        "LBXLUH": "lh_level",
        "LBXAMH": "amh_level",
        "LBXEST": "estradiol_level",
        "RIDAGEYR": "age",
    })
    df["hormone_therapy_use"] = (df["RHQ540"] == 1).astype(int)

    df[TARGET_COLUMN] = (df["age"] < PREMATURE_AGE_THRESHOLD).astype(int)

    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
    n_final = len(df)
    n_premature = df[TARGET_COLUMN].sum()
    print(f"Final sample size: {n_final} ({n_premature} premature / {n_final - n_premature} typical-age)")

    return df


def train_and_evaluate(df):
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    majority_class = y_train.mode()[0]
    baseline_preds = pd.Series([majority_class] * len(y_test), index=y_test.index)
    baseline_acc = accuracy_score(y_test, baseline_preds)
    print(f"Naive majority-class baseline accuracy: {baseline_acc:.3f}")

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, preds, average="binary", zero_division=0)

    print(f"Model accuracy: {acc:.3f}")
    print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, preds))

    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    return model


def predict(features: dict):
    model = joblib.load(MODEL_PATH)
    X = pd.DataFrame([features])[FEATURE_COLUMNS]
    label = model.predict(X)[0]
    confidence = model.predict_proba(X).max()
    return ("premature/early menopause" if label == 1 else "typical-age menopause"), float(confidence)


if __name__ == "__main__":
    df = load_and_label()
    train_and_evaluate(df)