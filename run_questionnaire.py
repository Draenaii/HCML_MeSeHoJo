import os
import sys
from datetime import datetime
import pandas as pd


INPUT_PATH = "questionnaire_output/participant_questionnaire_stimuli.csv"
OUTPUT_PATH = "questionnaire_output/questionnaire_responses.csv"

QUESTIONNAIRE_ITEMS = {
    "Q1": "The explanation helped me understand why the model made this prediction.",
    "Q2": "The explanation was clear.",
    "Q3": "The explanation gave enough detail to understand the prediction.",
    "Q4": "The explanation felt complete for this specific prediction.",
    "Q5": "The explanation made me trust the prediction more.",
}

LIKERT_TEXT = """
Please answer each statement on a 1-5 scale:

5 = I agree strongly
4 = I agree somewhat
3 = I'm neutral about it
2 = I disagree somewhat
1 = I disagree strongly
""".strip()


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def ask_likert(question_text):
    while True:
        answer = input(f"{question_text}\nYour answer (1-5): ").strip()
        if answer in {"1", "2", "3", "4", "5"}:
            return int(answer)
        print("Invalid answer. Please enter a number from 1 to 5.\n")


def clean_stimulus_text(text):
    cut_markers = [
        "Please answer the following statements",
        "Please answer each statement",
        "1. From the explanation",
    ]
    for marker in cut_markers:
        if marker in text:
            return text.split(marker)[0].strip()
    return text.strip()


def append_result(output_path, row):
    df = pd.DataFrame([row])
    df.to_csv(output_path, mode="a", header=not os.path.exists(output_path), index=False)


def run_questionnaire(participant_id):
    stimuli_df = pd.read_csv(INPUT_PATH)

    required_columns = [
        "stimulus_id",
        "student_id",
        "model_name",
        "model_type",
        "explanation_type",
        "predicted_label",
        "predicted_probability_pass",
        "participant_text",
    ]
    missing = [col for col in required_columns if col not in stimuli_df.columns]
    if missing:
        raise ValueError("Missing columns: " + ", ".join(missing))

    if "display_order" in stimuli_df.columns:
        stimuli_df = stimuli_df.sort_values("display_order").reset_index(drop=True)

    total_stimuli = len(stimuli_df)

    clear_screen()
    print("=" * 80)
    print("QUESTIONNAIRE")
    print("=" * 80)
    print()
    print(f"Participant ID: {participant_id}")
    print(f"Number of stimuli: {total_stimuli}")
    print()
    print("You will see model predictions for several students.")
    print("For each prediction, read the explanation and answer 5 questions.")
    print()
    print("Important: there are no right or wrong answers.")
    print()
    input("Press Enter to start...")

    session_start = datetime.now().isoformat(timespec="seconds")

    try:
        for position, (_, stimulus) in enumerate(stimuli_df.iterrows(), start=1):
            clear_screen()
            stimulus_start_time = datetime.now()
            stimulus_start_iso = stimulus_start_time.isoformat(timespec="seconds")

            print(clean_stimulus_text(str(stimulus["participant_text"])))
            print()
            print("=" * 80)
            print(LIKERT_TEXT)
            print("=" * 80)
            print()

            answers = {}
            for q_code, q_text in QUESTIONNAIRE_ITEMS.items():
                answers[q_code] = ask_likert(q_text)
                print()

            stimulus_end_time = datetime.now()
            result_row = {
                "participant_id": participant_id,
                "session_start": session_start,
                "stimulus_start_time": stimulus_start_iso,
                "stimulus_end_time": stimulus_end_time.isoformat(timespec="seconds"),
                "response_time_seconds": (stimulus_end_time - stimulus_start_time).total_seconds(),
                "shown_order": position,
                "stimulus_id": stimulus["stimulus_id"],
                "student_id": stimulus["student_id"],
                "participant_model_label": stimulus.get("participant_model_label", ""),
                "model_name": stimulus["model_name"],
                "model_type": stimulus["model_type"],
                "explanation_type": stimulus["explanation_type"],
                "predicted_label": stimulus["predicted_label"],
                "predicted_probability_pass": stimulus["predicted_probability_pass"],
                "Q1_understanding": answers["Q1"],
                "Q2_clear": answers["Q2"],
                "Q3_sufficient_detail": answers["Q3"],
                "Q4_completeness": answers["Q4"],
                "Q5_trust_prediction": answers["Q5"],
            }

            append_result(OUTPUT_PATH, result_row)
            print("\nResponse saved.")
            input("Press Enter to continue...")

    except KeyboardInterrupt:
        print("\n\nQuestionnaire interrupted.")
        print("Completed responses have already been saved.")
        sys.exit(1)

    clear_screen()
    print("=" * 80)
    print("QUESTIONNAIRE COMPLETE")
    print("=" * 80)
    print()
    print(f"Responses saved to:\n{OUTPUT_PATH}")
    print()
    print("Thank you.")


def main():
    if len(sys.argv) != 2:
        print("Usage: python run_questionnaire.py PARTICIPANT_ID")
        sys.exit(1)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    run_questionnaire(sys.argv[1])


if __name__ == "__main__":
    main()
