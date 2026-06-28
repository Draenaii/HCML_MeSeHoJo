from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


RESPONSES_PATH = Path("questionnaire_output/questionnaire_responses.csv")
KEY_PATH = Path("questionnaire_output/researcher_questionnaire_key.csv")
OUTPUT_DIR = Path("analysis_output")

Q_COLS = [
    "Q1_understanding",
    "Q2_clear",
    "Q3_sufficient_detail",
    "Q4_completeness",
    "Q5_trust_prediction",
]

Q_LABELS = {
    "Q1_understanding": "Understanding",
    "Q2_clear": "Clarity",
    "Q3_sufficient_detail": "Detail",
    "Q4_completeness": "Completeness",
    "Q5_trust_prediction": "Trust",
}

MODEL_ORDER = ["EBM", "Random Forest"]
CORRECTNESS_ORDER = ["Correct", "Incorrect"]


def paired_test(data, column):
    means = data.groupby(["participant_id", "model_name"])[column].mean().reset_index()
    pivot = means.pivot(index="participant_id", columns="model_name", values=column).dropna()
    ebm = pivot["EBM"].to_numpy()
    rf = pivot["Random Forest"].to_numpy()
    if len(pivot) < 2:
        return len(pivot), ebm.mean(), rf.mean(), ebm.mean() - rf.mean(), np.nan, np.nan
    t_stat, p_value = stats.ttest_rel(ebm, rf)
    return len(pivot), ebm.mean(), rf.mean(), ebm.mean() - rf.mean(), t_stat, p_value


def add_key_columns(responses, key):
    key_cols = [
        "stimulus_id",
        "student_id",
        "model_name",
        "actual_label",
        "actual_G3",
        "model_correct",
    ]
    key = key[key_cols].drop_duplicates()
    drop_cols = [c for c in ["actual_label", "actual_G3", "model_correct", "G3"] if c in responses.columns]
    responses = responses.drop(columns=drop_cols)
    merged = responses.merge(key, on=["stimulus_id", "student_id", "model_name"], how="left")
    merged["G3"] = merged["actual_G3"]
    merged["appropriate_trust"] = np.where(
        merged["model_correct"],
        merged["Q5_trust_prediction"],
        6 - merged["Q5_trust_prediction"],
    )
    return merged


def make_sq2_table(df):
    table = df.groupby("model_name")[Q_COLS].mean().round(2)
    table = table.rename(columns=Q_LABELS).reindex(MODEL_ORDER)
    tests = []
    for col in Q_COLS:
        n, ebm, rf, diff, t_stat, p_value = paired_test(df, col)
        tests.append({
            "item": Q_LABELS[col],
            "n": n,
            "EBM": round(ebm, 2),
            "Random Forest": round(rf, 2),
            "diff_EBM_minus_RF": round(diff, 2),
            "t": round(t_stat, 3),
            "p": round(p_value, 3),
        })
    return table, pd.DataFrame(tests)


def make_trust_tables(df):
    table = (
        df.groupby(["model_name", "model_correct"])[["Q5_trust_prediction", "appropriate_trust"]]
        .mean()
        .round(2)
        .reset_index()
    )
    table["model_correct"] = table["model_correct"].map({True: "Correct", False: "Incorrect"})
    table.columns = ["Model", "Correct?", "Mean trust", "Approp. trust"]
    table["Model"] = pd.Categorical(table["Model"], MODEL_ORDER, ordered=True)
    table["Correct?"] = pd.Categorical(table["Correct?"], CORRECTNESS_ORDER, ordered=True)
    table = table.sort_values(["Model", "Correct?"]).reset_index(drop=True)

    n, ebm, rf, diff, t_stat, p_value = paired_test(df, "appropriate_trust")
    overall = pd.DataFrame([{
        "n": n,
        "EBM": round(ebm, 2),
        "Random Forest": round(rf, 2),
        "diff_EBM_minus_RF": round(diff, 2),
        "t": round(t_stat, 3),
        "p": round(p_value, 3),
    }])
    return table, overall


def make_quality_tables(df):
    trust_std = df.groupby("participant_id")["Q5_trust_prediction"].std().sort_values()
    flat = trust_std[trust_std < 0.4]
    response_time = df.groupby("model_name")["response_time_seconds"].agg(["mean", "median", "std"]).round(1)
    session_time = df.groupby("participant_id")["response_time_seconds"].sum().sort_values().round(1)
    return flat, response_time, session_time


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    responses = pd.read_csv(RESPONSES_PATH)
    key = pd.read_csv(KEY_PATH)
    df = add_key_columns(responses, key)

    table2, sq2_tests = make_sq2_table(df)
    table3, trust_test = make_trust_tables(df)
    flat_responders, response_time, session_time = make_quality_tables(df)

    df.to_csv(OUTPUT_DIR / "questionnaire_responses_enriched.csv", index=False)
    table2.to_csv(OUTPUT_DIR / "table2_questionnaire_ratings.csv")
    table3.to_csv(OUTPUT_DIR / "table3_trust_by_correctness.csv", index=False)
    sq2_tests.to_csv(OUTPUT_DIR / "sq2_paired_tests.csv", index=False)
    trust_test.to_csv(OUTPUT_DIR / "sq3_appropriate_trust_test.csv", index=False)

    print(f"Responses: {len(df)} rows, {df['participant_id'].nunique()} participants")
    print("\nTable 2 — mean questionnaire ratings")
    print(table2)
    print("\nSQ2 paired tests")
    print(sq2_tests.to_string(index=False))
    print("\nTable 3 — trust by correctness")
    print(table3.to_string(index=False))
    print("\nSQ3 appropriate trust test")
    print(trust_test.to_string(index=False))
    print("\nFlat responders")
    print(flat_responders.to_string() if len(flat_responders) else "None")
    print("\nResponse time by model")
    print(response_time)
    print("\nTotal session time")
    print(session_time.to_string())
    print(f"\nSaved output to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
