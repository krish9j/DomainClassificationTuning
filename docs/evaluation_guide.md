# Evaluation Guide

This document explains how the model is evaluated and how to interpret the metrics.

## Metrics

### Exact Match Accuracy
*   **Definition**: Percentage of predictions where Domain, Attribute, AND Progression match the ground truth exactly.
*   **Target**: >80%
*   **Interpretation**: This is the strictest metric. Even if the domain is right, if the progression level is off by one (e.g., "Emerging" vs "Developing"), it counts as a failure.

### Domain Accuracy
*   **Definition**: Percentage of predictions where the Domain matches.
*   **Target**: >95%
*   **Interpretation**: The model should almost never miss the high-level domain (e.g., Physical vs Social).

### Progression Accuracy
*   **Definition**: Percentage of predictions where the Progression level matches.
*   **Target**: >75%
*   **Interpretation**: This is often the hardest part, as the distinction between "Developing" and "Progressing" can be subtle in the text.

## Running Evaluation

Use the `evaluation.py` script:

```bash
python scripts/evaluation.py --dataset data/training/valid.jsonl --endpoint <ENDPOINT_ID>
```

## Analyzing the Report

The script generates `evaluation_report.json`.

1.  **Failures Section**: Look at `failures` to see specific examples where the model guessed wrong.
2.  **Patterns**:
    *   *Are we consistently mixing up "Social" and "Language"?* -> Check if input attributes overlap.
    *   *Are we biased towards "Emerging"?* -> Check class balance in `train.jsonl`.

## Improving Performance

1.  **More Data**: Add more examples for the underperforming classes.
2.  **Better Prompts**: Adjust `CLASSIFICATION_PROMPT_TEMPLATE` in `config.py` to be more specific.
3.  **Data Cleaning**: Ensure observation notes in training data are high quality and actually justify the label.
