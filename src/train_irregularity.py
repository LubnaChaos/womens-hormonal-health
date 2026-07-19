import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from src.train import _read_xpt, DATA_DIR

RANDOM_SEED = 42
MODEL_PATH = "src/model_irregularity.pkl"

FEATURE_COLUMNS = ["age", "fsh_level", "lh_level", "amh_level", "estradiol_level", "insulin_level", "glucose_level"]
TARGET_COLUMN = "unexplained_irregularity"


def load_and_label():
    demo = _read_xpt(f"{DATA_DIR}/P_DEMO.xpt")
    rhq = _read_xpt(f"{DATA_DIR}/P_RHQ.xpt")
    tst = _read_xpt(f"{DATA_DIR}/P_TST.xpt")
    ins = _read_xpt(f"{DATA_DIR}/P_INS.xpt")
    glu = _read_xpt(f"{DATA_DIR}/P_GLU.xpt")

    df = demo.merge(rhq, on="SEQN", how="inner")
    df = df.merge(tst, on="SEQN", how="inner")
    df = df.merge(ins, on="SEQN", how="inner")
    df = df.merge(glu, on="SEQN", how="inner")

    df = df.rename(columns={"RIDAGEYR": "age"})
    df = df[df["age"] < 40].copy()
    n_before = len(df)

    def label_row(row):
        if row.get("RHQ031") == 1:
            return 0
        if row.get("RHQ031") == 2 and row.get("RHD043") == 9:
            return 1
        return None

    df[TARGET_COLUMN] = df.apply(label_row, axis=1)
    df = df.dropna(subset=[TARGET_COLUMN])
    n_after_label = len(df)
    print(f"Population under 40: {n_before}. Dropped {n_before - n_after_label} rows with an explained reason or missing data.")

    df = df.rename(columns={
        "LBXFSH": "fsh_level",
        "LBXLUH": "lh_level",
        "LBXAMH": "amh_level",
        "LBXEST": "estradiol_level",
        "LBXIN": "insulin_level",
        "LBXGLU": "glucose_level",
    })

    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
    n_final = len(df)
    n_positive = int(df[TARGET_COLUMN].sum())
    print(f"Final sample size: {n_final} ({n_positive} with unexplained irregularity / {n_final - n_positive} without)")

    output_cols = ["SEQN"] + FEATURE_COLUMNS + [TARGET_COLUMN]
    df[output_cols].to_csv("data/cleaned_irregularity_dataset.csv", index=False)
    print("Cleaned dataset saved to data/cleaned_irregularity_dataset.csv")

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
    return ("unexplained irregularity" if label == 1 else "regular"), float(confidence)


if __name__ == "__main__":
    df = load_and_label()
    train_and_evaluate(df)