# HCML_MeSeHoJo

This repository contains the code and data used for our Human-Centered Machine Learning (HCML) project. The project compares how understandable, clear, complete, and trustworthy explanations from two machine learning models are perceived by participants.

The models are trained on a [Portuguese student performance dataset](https://archive.ics.uci.edu/dataset/320/student%2Bperformance). The task is to predict whether a student passes or fails based on student, family, school, and social variables. The project compares:

* **Explainable Boosting Machine (EBM)**
  An inherently interpretable model with built-in local additive explanations.

* **Random Forest**
  A black-box model explained using a custom local perturbation-based explanation method.

The generated explanations are used in a questionnaire where participants rate the quality and trustworthiness of the explanations.

## Project overview

The project workflow consists of three main steps:

1. **Prepare questionnaire stimuli**
   Train the EBM and Random Forest models, generate predictions and local explanations for selected students, and export questionnaire materials.

2. **Run the questionnaire**
   Present each participant with model predictions and explanations, then collect Likert-scale responses.

3. **Analyse questionnaire responses**
   Analyse participant ratings for explanation quality and trust, including comparisons between EBM and Random Forest explanations.


## Data

The repository includes the student performance datasets:

* `student-por.csv`
* `student-mat.csv`

The current scripts use `student-por.csv`. The target variable is derived from the final grade `G3`:

```python
pass = 1 if G3 >= 10 else 0
```

## Models

### Explainable Boosting Machine

The EBM is trained directly on the raw feature table. It is treated as an inherently interpretable model because it can provide built-in local additive explanations for individual predictions.

### Random Forest

The Random Forest is trained on one-hot encoded features. Since Random Forest is a black-box model, explanations are generated using a custom perturbation method. For each feature, the method replaces the student's value with a baseline value and measures how much the predicted probability changes.

## Questionnaire stimuli

The questionnaire is generated using:

```bash
python prepare_questionnaire.py
```

This script:

* loads the student performance data;
* creates the binary pass/fail target;
* trains the EBM and Random Forest models;
* evaluates both models;
* generates predictions for selected students;
* generates local explanations for each prediction;
* anonymises model labels as `Model A` and `Model B`;
* exports participant-facing and researcher-facing questionnaire files.

The selected students are defined in `prepare_questionnaire.py`:

```python
SELECTED_STUDENT_IDS = [597, 604, 517, 443]
```

The generated files are saved in:

```text
questionnaire_output/
```

Important generated files include:

* `participant_questionnaire.txt`
  Human-readable questionnaire text for participants.

* `participant_questionnaire_stimuli.csv`
  Structured questionnaire stimuli used by the questionnaire runner.

* `researcher_questionnaire_key.csv`
  Researcher-facing key containing model names, predictions, actual labels, and correctness.

* `questionnaire_results_template.csv`
  Empty template for manually entering questionnaire responses.

## Running the questionnaire

To run the questionnaire for a participant, use:

```bash
python run_questionnaire.py PARTICIPANT_ID
```

Example:

```bash
python run_questionnaire.py P01
```

The script shows each stimulus one by one and asks the participant to answer five Likert-scale questions:

1. The explanation helped me understand why the model made this prediction.
2. The explanation was clear.
3. The explanation gave enough detail to understand the prediction.
4. The explanation felt complete for this specific prediction.
5. The explanation made me trust the prediction more.

Responses are saved to:

```text
questionnaire_output/questionnaire_responses.csv
```

Each response row includes:

* participant ID;
* stimulus ID;
* student ID;
* model name;
* model type;
* explanation type;
* predicted label;
* predicted probability of passing;
* Likert responses;
* response time.

## Analysing the results

After collecting questionnaire responses, run:

```bash
python sq2_sq3_analysis.py
```

This script combines participant responses with the researcher key and produces analysis tables for the second and third subquestions.

The enriched and analysed outputs are saved in:

```text
analysis_output/
```

Generated analysis files include:

* `questionnaire_responses_enriched.csv`
  Questionnaire responses merged with actual labels and model correctness.

* `table2_questionnaire_ratings.csv`
  Mean questionnaire ratings per model.

* `sq2_paired_tests.csv`
  Paired tests comparing EBM and Random Forest ratings.

* `table3_trust_by_correctness.csv`
  Mean trust scores split by model correctness.

* `sq3_appropriate_trust_test.csv`
  Paired test for appropriate trust.

## Research focus

The project investigates how participants perceive explanations from an inherently interpretable model compared to explanations from a black-box model.

The main comparison is between:

* EBM explanations, which come from the model structure itself;
* Random Forest explanations, which are produced post-hoc using local perturbation.

The questionnaire focuses on explanation quality dimensions such as:

* understanding;
* clarity;
* sufficient detail;
* completeness;
* trust.

The analysis also considers whether participants show more appropriate trust when a model prediction is correct or incorrect.

## Installation

This project uses Python. A virtual environment is recommended.

```bash
python -m venv .venv
```

Activate the environment:

On Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

Install the required packages manually:

```bash
pip install numpy pandas scipy scikit-learn interpret
```

## Reproducing the workflow

From the repository root, run:

```bash
python prepare_questionnaire.py
```

Then collect questionnaire responses:

```bash
python run_questionnaire.py P01
python run_questionnaire.py P02
python run_questionnaire.py P03
```

After responses have been collected, run:

```bash
python sq2_sq3_analysis.py
```

The final analysis tables will be written to `analysis_output/`.

## Notes

* The model labels shown to participants are anonymised as `Model A` and `Model B`.
* The researcher key contains the true model identities.
* The EBM uses built-in local explanations where possible.
* The Random Forest uses a custom local perturbation method rather than LIME or SHAP.
* The current workflow is designed for a small human-subject questionnaire study rather than large-scale model benchmarking.

## Authors

This project was created for the Human-Centered Machine Learning course at Utrecht University.

Team members:

* Thijmen van der Meijden
* Betül Selvi
* Kai ter Horst
* Finn Joosten
