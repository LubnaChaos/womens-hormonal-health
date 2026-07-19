import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from src.train import _read_xpt, DATA_DIR

RANDOM_SEED = 42
MODEL_PATH = "src/model_menopause_age.pkl"

FEATURE_COLUMNS = ["fsh_level", "lh_level", "amh_level", "estradiol_level"]
TARGET_COLUMN = "age_at_menopause"


def load_and_label():
    demo = _read_xpt(f"{DATA_DIR}/P_DEMO.xpt")
    rhq = _read_xpt(f"{DATA_DIR}/P_RHQ.xpt")
    tst = _read_xpt(f"{DATA_DIR}/P_TST.xpt")

    df = demo.merge(rhq, on="SEQN", how="inner")
    df = df.merge(tst, on="SEQN", how="inner")

    n_before = len(df)

    # RHQ060: "About how old were you when you had your last menstrual
    # period?" Only meaningful for women who have gone through natural
    # menopause (RHQ031 == 2 and RHD043 == 7, same logic as train.py).
    is_natural_menopause = (df["RHQ031"] == 2) & (df["RHD043"] == 7)
    df = df[is_natural_menopause].copy()

    # NHANES codes: 777 = Refused, 999 = Don't know -> treat as missing
    df[TARGET_COLUMN] = df["RHQ060"].where(~df["RHQ060"].isin([777, 999]))

    df = df.rename(columns={
        "LBXFSH": "fsh_level",
        "LBXLUH": "lh_level",
        "LBXAMH": "amh_level",
        "LBXEST": "estradiol_level",
    })

    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
    n_final = len(df)
    print(f"Women with natural menopause and a valid age-at-menopause answer: {n_final}")
    print(f"(started from {n_before} total merged rows)")
    print(df[TARGET_COLUMN].describe())

    output_cols = ["SEQN"] + FEATURE_COLUMNS + [TARGET_COLUMN]
    df[output_cols].to_csv("data/cleaned_menopause_age_dataset.csv", index=False)
    print("Cleaned dataset saved to data/cleaned_menopause_age_dataset.csv")

    return df


def train_and_evaluate(df):
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )

    # Naive baseline: always predict the average age at menopause
    naive_pred = y_train.mean()
    naive_mae = mean_absolute_error(y_test, [naive_pred] * len(y_test))
    print(f"\nNaive baseline (always predict mean age {naive_pred:.1f}): MAE = {naive_mae:.2f} years")

    model = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=RANDOM_SEED)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    print(f"Model MAE: {mae:.2f} years")
    print(f"Model R^2: {r2:.3f}  (fraction of variance explained; 0 = no better than guessing the mean, 1 = perfect)")

    joblib.dump(model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

    return model


def predict(features: dict):
    model = joblib.load(MODEL_PATH)
    X = pd.DataFrame([features])[FEATURE_COLUMNS]
    predicted_age = model.predict(X)[0]
    return float(predicted_age)


if __name__ == "__main__":
    df = load_and_label()
    train_and_evaluate(df)