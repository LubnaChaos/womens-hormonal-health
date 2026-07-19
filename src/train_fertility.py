import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from src.train import _read_xpt, DATA_DIR

RANDOM_SEED = 42
MODEL_PATH = "src/model_fertility.pkl"

FEATURE_COLUMNS = ["age", "fsh_level", "lh_level", "amh_level", "estradiol_level"]
TARGET_COLUMN = "infertility_history"


def load_and_label():
    demo = _read_xpt(f"{DATA_DIR}/P_DEMO.xpt")
    rhq = _read_xpt(f"{DATA_DIR}/P_RHQ.xpt")
    tst = _read_xpt(f"{DATA_DIR}/P_TST.xpt")

    df = demo.merge(rhq, on="SEQN", how="inner")
    df = df.merge(tst, on="SEQN", how="inner")

    n_before = len(df)

    def label_row(value):
        if value == 1:
            return 1  # tried 1+ year without success -> infertility history
        if value == 2:
            return 0  # no difficulty
        return None  # refused / don't know / not asked

    df[TARGET_COLUMN] = df["RHQ074"].apply(label_row)
    df = df.dropna(subset=[TARGET_COLUMN])
    n_after_label = len(df)
    print(f"Dropped {n_before - n_after_label} rows without a resolvable RHQ074 answer")

    df = df.rename(columns={
        "RIDAGEYR": "age",
        "LBXFSH": "fsh_level",
        "LBXLUH": "lh_level",
        "LBXAMH": "amh_level",
        "LBXEST": "estradiol_level",
    })

    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
    n_final = len(df)
    n_positive = int(df[TARGET_COLUMN].sum())
    print(f"Final sample size: {n_final} ({n_positive} with infertility history / {n_final - n_positive} without)")

    output_cols = ["SEQN"] + FEATURE_COLUMNS + [TARGET_COLUMN]
    df[output_cols].to_csv("data/cleaned_fertility_dataset.csv", index=False)
    print("Cleaned dataset saved to data/cleaned_fertility_dataset.csv")

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
    return ("infertility history" if label == 1 else "no infertility history"), float(confidence)


if __name__ == "__main__":
    df = load_and_label()
    train_and_evaluate(df)