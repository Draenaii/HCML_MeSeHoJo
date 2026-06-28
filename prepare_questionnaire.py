from pathlib import Path

import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
DATA_PATH = Path("student-por.csv")
OUTPUT_DIR = Path("questionnaire_output")
SELECTED_STUDENT_IDS = [597, 604, 517, 443]
TOP_N = 5
RANDOMIZE_STIMULI = True
SHOW_MODEL_NAMES = False
MODEL_DISPLAY_NAMES = {"EBM": "A", "Random Forest": "B"}

FEATURE_LABELS = {
    "school": "School",
    "sex": "Gender",
    "age": "Age",
    "address": "Home address type",
    "famsize": "Family size",
    "Pstatus": "Parents' cohabitation status",
    "Medu": "Mother's education level",
    "Fedu": "Father's education level",
    "Mjob": "Mother's job",
    "Fjob": "Father's job",
    "reason": "Reason for choosing the school",
    "guardian": "Guardian",
    "traveltime": "Travel time to school",
    "studytime": "Weekly study time",
    "failures": "Number of previous class failures",
    "schoolsup": "Extra educational support from school",
    "famsup": "Family educational support",
    "paid": "Extra paid classes",
    "activities": "Extracurricular activities",
    "nursery": "Attended nursery school",
    "higher": "Wants to pursue higher education",
    "internet": "Internet access at home",
    "romantic": "In a romantic relationship",
    "famrel": "Quality of family relationships",
    "freetime": "Free time after school",
    "goout": "Going out with friends",
    "Dalc": "Weekday alcohol consumption",
    "Walc": "Weekend alcohol consumption",
    "health": "Current health status",
    "absences": "Number of school absences",
}

VALUE_LABELS = {
    "school": {"GP": "school GP", "MS": "school MS"},
    "sex": {"F": "female", "M": "male"},
    "address": {"U": "urban", "R": "rural"},
    "famsize": {"LE3": "3 or fewer family members", "GT3": "more than 3 family members"},
    "Pstatus": {"T": "parents living together", "A": "parents living apart"},
    "Medu": {0: "none", 1: "primary education", 2: "lower secondary education", 3: "secondary education", 4: "higher education"},
    "Fedu": {0: "none", 1: "primary education", 2: "lower secondary education", 3: "secondary education", 4: "higher education"},
    "Mjob": {"teacher": "teacher", "health": "health care", "services": "services", "at_home": "at home", "other": "other"},
    "Fjob": {"teacher": "teacher", "health": "health care", "services": "services", "at_home": "at home", "other": "other"},
    "reason": {"home": "close to home", "reputation": "school reputation", "course": "course preference", "other": "other"},
    "guardian": {"mother": "mother", "father": "father", "other": "other"},
    "traveltime": {1: "less than 15 minutes", 2: "15 to 30 minutes", 3: "30 minutes to 1 hour", 4: "more than 1 hour"},
    "studytime": {1: "less than 2 hours", 2: "2 to 5 hours", 3: "5 to 10 hours", 4: "more than 10 hours"},
}
for col in ["schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet", "romantic"]:
    VALUE_LABELS[col] = {"yes": "yes", "no": "no"}

LIKERT_TEXT = """Please answer the following statements on a 1-5 scale:

5 = I agree strongly
4 = I agree somewhat
3 = I'm neutral about it
2 = I disagree somewhat
1 = I disagree strongly"""

QUESTIONNAIRE_ITEMS = [
    "The explanation helped me understand why the model made this prediction.",
    "The explanation was clear.",
    "The explanation gave enough detail to understand the prediction.",
    "The explanation felt complete for this specific prediction.",
    "The explanation made me trust the prediction more.",
]
QUESTIONNAIRE_ITEM_NAMES = {
    "Q1": "understanding",
    "Q2": "clear",
    "Q3": "sufficient_detail",
    "Q4": "completeness",
    "Q5": "trust_prediction",
}



def readable_value(feature, value):
    return VALUE_LABELS.get(feature, {}).get(value, value)


def readable_feature_value(feature, value):
    return f"{FEATURE_LABELS.get(feature, feature)}: {readable_value(feature, value)}"


def pass_fail(value):
    return "pass" if int(value) == 1 else "fail"


def load_data(csv_path, selected_ids, random_state):
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find {csv_path}. Put student-por.csv next to this script.")

    df = pd.read_csv(csv_path, sep=";")
    df["pass"] = (df["G3"] >= 10).astype(int)

    x = df.drop(columns=["G1", "G2", "G3", "pass"])
    y = df["pass"]

    for col in x.select_dtypes(include="object"):
        x[col] = x[col].astype("category")

    cat_cols = x.select_dtypes(exclude="number").columns.tolist()
    x_enc = pd.get_dummies(x, columns=cat_cols)

    split = train_test_split(x, y, test_size=0.2, random_state=random_state, stratify=y)
    x_train, x_test, y_train, y_test = split
    x_train_enc = x_enc.loc[x_train.index]
    x_test_enc = x_enc.loc[x_test.index]

    missing = [sid for sid in selected_ids if sid not in x.index]
    if missing:
        raise ValueError(f"Selected students not found in the dataset index: {missing}")

    not_test = [sid for sid in selected_ids if sid not in x_test.index]
    if not_test:
        print(f"Warning: these selected students are not in the test split: {not_test}")

    return df, x, y, x_enc, x_train, x_test, y_train, y_test, x_train_enc, x_test_enc, cat_cols


def train_models(x_train, y_train, x_train_enc, random_state):
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=random_state)
    rf.fit(x_train_enc, y_train)

    ebm = ExplainableBoostingClassifier(random_state=random_state, n_jobs=1)
    ebm.fit(x_train, y_train)

    return {
        "EBM": {
            "model": ebm,
            "input_type": "raw",
            "model_type": "inherently interpretable",
            "explanation_type": "built-in local additive explanation",
        },
        "Random Forest": {
            "model": rf,
            "input_type": "encoded",
            "model_type": "black-box",
            "explanation_type": "post-hoc local perturbation explanation",
        },
    }


def evaluate_model(name, model, x_eval, y_true):
    y_pred = model.predict(x_eval)
    y_prob = model.predict_proba(x_eval)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else np.nan

    return {
        "Model": name,
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "ROC-AUC": roc_auc_score(y_true, y_prob),
        "Recall / TPR": recall_score(y_true, y_pred),
        "Precision / PPV": precision_score(y_true, y_pred),
        "Specificity / TNR": specificity,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }


def baseline_values(x_train, cat_cols):
    return {
        col: x_train[col].mode().iloc[0] if col in cat_cols else x_train[col].median()
        for col in x_train.columns
    }


def build_helpers(df, x, x_encoded, cat_cols, models, show_real_names):
    def student_row(student_id):
        return x.loc[[student_id]].copy()

    def encode(raw):
        return pd.get_dummies(raw, columns=cat_cols).reindex(columns=x_encoded.columns, fill_value=0)

    def model_input(raw, input_type):
        return raw if input_type == "raw" else encode(raw)

    def predict(model_name, student_id):
        info = models[model_name]
        raw = student_row(student_id)
        prepared = model_input(raw, info["input_type"])
        pred = int(info["model"].predict(prepared)[0])
        prob = float(info["model"].predict_proba(prepared)[0, 1])
        return pred, prob

    def actual(student_id):
        return int(df.loc[student_id, "pass"]), df.loc[student_id, "G3"]

    def model_label(model_name):
        return model_name if show_real_names else MODEL_DISPLAY_NAMES.get(model_name, "Model")

    return student_row, model_input, predict, actual, model_label


def term_features(term):
    return [part.strip() for part in str(term).split("&")]


def readable_ebm_term(term, row):
    parts = []
    values = []
    for feature in term_features(term):
        if feature in row.columns:
            value = row.iloc[0][feature]
            parts.append(readable_feature_value(feature, value))
            values.append(str(readable_value(feature, value)))
        else:
            parts.append(feature)
            values.append("unknown")
    return " and ".join(parts), " and ".join(values)


def perturbation_explanation(model_name, student_id, x, models, base, student_row, model_input, predict, top_n):
    info = models[model_name]
    row = student_row(student_id)
    original_prob = predict(model_name, student_id)[1]
    records = []

    for feature in x.columns:
        changed = row.copy()
        original_value = row.iloc[0][feature]
        changed.loc[changed.index[0], feature] = base[feature]
        changed_prob = float(info["model"].predict_proba(model_input(changed, info["input_type"]))[0, 1])
        contribution = original_prob - changed_prob
        records.append({
            "feature": feature,
            "readable_factor": readable_feature_value(feature, original_value),
            "actual_value": original_value,
            "baseline_value": base[feature],
            "contribution": contribution,
            "absolute_contribution": abs(contribution),
        })

    return pd.DataFrame(records).sort_values("absolute_contribution", ascending=False).head(top_n)


def ebm_explanation(student_id, ebm, student_row, fallback, top_n):
    row = student_row(student_id)
    try:
        data = ebm.explain_local(row).data(0)
        records = []
        for name, score in zip(data["names"], data["scores"]):
            readable_factor, actual_value = readable_ebm_term(name, row)
            records.append({
                "feature": name,
                "readable_factor": readable_factor,
                "actual_value": actual_value,
                "baseline_value": None,
                "contribution": float(score),
                "absolute_contribution": abs(float(score)),
            })
        return pd.DataFrame(records).sort_values("absolute_contribution", ascending=False).head(top_n)
    except Exception as exc:
        print(f"EBM explanation failed for student {student_id}; using perturbation instead. ({exc})")
        return fallback("EBM", student_id, top_n)


def explanation_text(explanation, top_n):
    inc = explanation[explanation["contribution"] > 0].sort_values("contribution", ascending=False).head(top_n)
    dec = explanation[explanation["contribution"] < 0].sort_values("contribution").head(top_n)

    lines = ["The following factors had the strongest influence on this prediction.", ""]
    lines.append("Factors increasing the predicted chance of passing:")
    lines += [f"- {row.readable_factor}" for row in inc.itertuples()] or ["- No strong increasing factors found."]
    lines.append("")
    lines.append("Factors decreasing the predicted chance of passing:")
    lines += [f"- {row.readable_factor}" for row in dec.itertuples()] or ["- No strong decreasing factors found."]
    return "\n".join(lines)


def participant_text(row, total, model_label):
    text = f"""Stimulus {int(row.display_order)} of {total}

Student {int(row.student_id)}

Model: {model_label(row.model_name)}

The model predicts that this student will {row.predicted_label}.
Predicted probability of passing: {row.predicted_probability_pass * 100:.1f}%.

Explanation:
{row.explanation_text}

{LIKERT_TEXT}"""
    for i, item in enumerate(QUESTIONNAIRE_ITEMS, 1):
        text += f"\n{i}. {item}"
    return text


def researcher_text(row, total, model_label):
    return f"""Stimulus {int(row.display_order)} of {total}

Student {int(row.student_id)}

Participant-facing model label: {model_label(row.model_name)}
Actual model name: {row.model_name}

The model predicts that this student will {row.predicted_label}.
Predicted probability of passing: {row.predicted_probability_pass * 100:.1f}%.

Actual outcome: {row.actual_label}
Actual G3 grade: {row.actual_G3}
Model prediction correct: {bool(row.model_correct)}

Explanation:
{row.explanation_text}"""


def make_stimuli(selected_ids, models, explain, predict, actual, model_label, randomize, random_state, top_n):
    records = []
    stimulus_id = 1

    for student_id in selected_ids:
        actual_pass, actual_g3 = actual(student_id)
        for model_name, info in models.items():
            pred, prob = predict(model_name, student_id)
            exp = explain(model_name, student_id)
            records.append({
                "stimulus_id": stimulus_id,
                "student_id": student_id,
                "model_name": model_name,
                "participant_model_label": model_label(model_name),
                "model_type": info["model_type"],
                "explanation_type": info["explanation_type"],
                "predicted_pass": pred,
                "predicted_label": pass_fail(pred),
                "predicted_probability_pass": prob,
                "actual_pass": actual_pass,
                "actual_label": pass_fail(actual_pass),
                "actual_G3": actual_g3,
                "model_correct": pred == actual_pass,
                "explanation_text": explanation_text(exp, top_n=top_n),
            })
            stimulus_id += 1

    df = pd.DataFrame(records)
    if randomize:
        df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    df["display_order"] = np.arange(1, len(df) + 1)
    total = len(df)
    df["participant_text"] = [participant_text(row, total, model_label) for row in df.itertuples()]
    df["researcher_text"] = [researcher_text(row, total, model_label) for row in df.itertuples()]
    return df


def export_files(stimuli, comparison, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    participant_cols = [
        "display_order", "stimulus_id", "student_id", "participant_model_label", "model_name",
        "model_type", "explanation_type", "predicted_label", "predicted_probability_pass", "participant_text",
    ]
    researcher_cols = [
        "display_order", "stimulus_id", "student_id", "participant_model_label", "model_name",
        "model_type", "explanation_type", "predicted_label", "predicted_probability_pass",
        "actual_label", "actual_G3", "model_correct", "researcher_text",
    ]

    stimuli[participant_cols].to_csv(output_dir / "participant_questionnaire_stimuli.csv", index=False)
    stimuli[researcher_cols].to_csv(output_dir / "researcher_questionnaire_key.csv", index=False)
    comparison.to_csv(output_dir / "model_comparison.csv", index=False)

    for filename, column in [
        ("participant_questionnaire.txt", "participant_text"),
        ("researcher_questionnaire_key.txt", "researcher_text"),
    ]:
        with open(output_dir / filename, "w", encoding="utf-8") as handle:
            for text in stimuli.sort_values("display_order")[column]:
                handle.write("=" * 100 + "\n")
                handle.write(text + "\n")
                handle.write("=" * 100 + "\n\n")

    template = stimuli[["display_order", "stimulus_id", "student_id", "participant_model_label", "model_name"]].copy()
    template["participant_id"] = ""
    for i, item in enumerate(QUESTIONNAIRE_ITEMS, 1):
        code = f"Q{i}"
        template[f"{code}_{QUESTIONNAIRE_ITEM_NAMES[code]}"] = ""
        template[f"{code}_text"] = item
    template["free_text_comment"] = ""
    template.to_csv(output_dir / "questionnaire_results_template.csv", index=False)


def main():
    data = load_data(DATA_PATH, SELECTED_STUDENT_IDS, RANDOM_STATE)
    df, x, _, x_encoded, x_train, x_test, y_train, y_test, x_train_enc, x_test_enc, cat_cols = data

    models = train_models(x_train, y_train, x_train_enc, RANDOM_STATE)
    comparison = pd.DataFrame([
        evaluate_model("EBM", models["EBM"]["model"], x_test, y_test),
        evaluate_model("Random Forest", models["Random Forest"]["model"], x_test_enc, y_test),
    ])

    student_row, model_input, predict, actual, model_label = build_helpers(
        df, x, x_encoded, cat_cols, models, SHOW_MODEL_NAMES
    )
    base = baseline_values(x_train, cat_cols)

    def explain(model_name, student_id):
        fallback = lambda name, sid, top_n: perturbation_explanation(
            name, sid, x, models, base, student_row, model_input, predict, top_n
        )
        if model_name == "EBM":
            return ebm_explanation(student_id, models["EBM"]["model"], student_row, fallback, TOP_N)
        return fallback(model_name, student_id, TOP_N)

    stimuli = make_stimuli(
        selected_ids=SELECTED_STUDENT_IDS,
        models=models,
        explain=explain,
        predict=predict,
        actual=actual,
        model_label=model_label,
        randomize=RANDOMIZE_STIMULI,
        random_state=RANDOM_STATE,
        top_n=TOP_N,
    )
    export_files(stimuli, comparison, OUTPUT_DIR)

    print("Model comparison:")
    print(comparison.round(3).to_string(index=False))
    print(f"\nSaved questionnaire files to: {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
