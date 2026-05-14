import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# IMPORTANT FIX FOR JOBLIB / PICKLE
# ---------------------------------------------------------------------
# The model was trained using custom objects from clean_arxiv_authors.py:
#   FeatureExtractor
#   get_candidate_texts
#
# When loading the model in this evaluation script, Python must be able
# to find those objects. Therefore clean_arxiv_authors.py must be in the
# same folder as this file.
# ---------------------------------------------------------------------

try:
    import clean_arxiv_authors as components

    sys.modules["__main__"].FeatureExtractor = components.FeatureExtractor
    sys.modules["__main__"].get_candidate_texts = components.get_candidate_texts

except Exception as exc:
    print("[ERROR] Could not import clean_arxiv_authors.py.")
    print("Make sure clean_arxiv_authors.py is in the same folder as this script.")
    print(f"Original error: {exc}")
    raise


from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.calibration import calibration_curve


# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def make_output_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_labelled_data(label_file: str) -> pd.DataFrame:
    df = pd.read_csv(label_file)

    required = {"candidate", "label"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns in label file: {missing}")

    df = df.dropna(subset=["candidate", "label"]).copy()
    df["candidate"] = df["candidate"].astype(str)

    # Convert labels robustly
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"]).copy()
    df["label"] = df["label"].astype(int)

    # Keep only binary labels
    df = df[df["label"].isin([0, 1])].copy()

    if df.empty:
        raise ValueError("No valid labelled rows found. Expected label values 0 and 1.")

    if df["label"].nunique() < 2:
        raise ValueError(
            "The label file must contain both classes: "
            "0 = NON_PERSON and 1 = PERSON."
        )

    return df


def prepare_X(df: pd.DataFrame):
    """
    This must match the input format used during training in clean_arxiv_authors.py.
    """
    if "ner_label" in df.columns:
        return [
            {
                "candidate": row["candidate"],
                "ner_label": row["ner_label"] if pd.notna(row["ner_label"]) else None,
            }
            for _, row in df.iterrows()
        ]

    return [
        {
            "candidate": candidate,
            "ner_label": None,
        }
        for candidate in df["candidate"]
    ]


def get_person_probabilities(model, X):
    proba = model.predict_proba(X)

    classes = list(model.classes_)

    if 1 not in classes:
        raise ValueError("Model classes do not contain label 1 for PERSON.")

    person_index = classes.index(1)
    return proba[:, person_index]


# ---------------------------------------------------------------------
# Reports and plots
# ---------------------------------------------------------------------

def save_metrics_report(
    y_true,
    y_pred,
    y_score,
    output_dir: str,
    threshold: float,
):
    accuracy = accuracy_score(y_true, y_pred)
    person_precision = precision_score(y_true, y_pred, zero_division=0)
    person_recall = recall_score(y_true, y_pred, zero_division=0)
    person_f1 = f1_score(y_true, y_pred, zero_division=0)

    try:
        roc_auc = roc_auc_score(y_true, y_score)
    except ValueError:
        roc_auc = np.nan

    pr_auc = average_precision_score(y_true, y_score)

    report = classification_report(
        y_true,
        y_pred,
        target_names=["NON_PERSON", "PERSON"],
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    tn, fp, fn, tp = cm.ravel()

    text = []
    text.append("AUTHOR PERSON-CLASSIFIER EVALUATION")
    text.append("=" * 80)
    text.append("")
    text.append(f"Decision threshold for PERSON: {threshold:.3f}")
    text.append("")
    text.append("Main metrics")
    text.append("-" * 80)
    text.append(f"Accuracy:          {accuracy:.4f}")
    text.append(f"PERSON precision:  {person_precision:.4f}")
    text.append(f"PERSON recall:     {person_recall:.4f}")
    text.append(f"PERSON F1-score:   {person_f1:.4f}")
    text.append(f"ROC-AUC:           {roc_auc:.4f}")
    text.append(f"PR-AUC:            {pr_auc:.4f}")
    text.append("")
    text.append("Confusion matrix counts")
    text.append("-" * 80)
    text.append(f"True NON_PERSON predicted NON_PERSON: {tn}")
    text.append(f"True NON_PERSON predicted PERSON:     {fp}  <-- false positives")
    text.append(f"True PERSON predicted NON_PERSON:     {fn}  <-- false negatives")
    text.append(f"True PERSON predicted PERSON:         {tp}")
    text.append("")
    text.append("Classification report")
    text.append("-" * 80)
    text.append(report)

    out_path = os.path.join(output_dir, "evaluation_metrics.txt")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(text))

    print("\n".join(text))
    print(f"\nSaved metrics report to: {out_path}")


def plot_confusion_matrix(y_true, y_pred, output_dir: str):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["NON_PERSON", "PERSON"],
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(values_format="d", ax=ax)
    plt.title("Confusion matrix")
    plt.tight_layout()

    path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {path}")


def plot_normalized_confusion_matrix(y_true, y_pred, output_dir: str):
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
        normalize="true",
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["NON_PERSON", "PERSON"],
    )

    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(values_format=".2f", ax=ax)
    plt.title("Normalized confusion matrix")
    plt.tight_layout()

    path = os.path.join(output_dir, "confusion_matrix_normalized.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {path}")


def plot_roc_curve(y_true, y_score, output_dir: str):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve")
    plt.legend()
    plt.tight_layout()

    path = os.path.join(output_dir, "roc_curve.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {path}")


def plot_precision_recall_curve(y_true, y_score, output_dir: str):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)
    baseline = np.mean(y_true)

    plt.figure(figsize=(7, 6))
    plt.plot(recall, precision, label=f"PR-AUC = {pr_auc:.3f}")
    plt.axhline(baseline, linestyle="--", label=f"Baseline = {baseline:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-recall curve")
    plt.legend()
    plt.tight_layout()

    path = os.path.join(output_dir, "precision_recall_curve.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {path}")


def plot_threshold_metrics(y_true, y_score, output_dir: str):
    thresholds = np.linspace(0.01, 0.99, 99)

    rows = []

    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)

        rows.append({
            "threshold": threshold,
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "accepted_rate": y_pred.mean(),
        })

    df = pd.DataFrame(rows)

    path_csv = os.path.join(output_dir, "threshold_metrics.csv")
    df.to_csv(path_csv, index=False)

    print(f"Saved: {path_csv}")

    plt.figure(figsize=(9, 6))
    plt.plot(df["threshold"], df["precision"], label="Precision")
    plt.plot(df["threshold"], df["recall"], label="Recall")
    plt.plot(df["threshold"], df["f1"], label="F1")
    plt.plot(df["threshold"], df["accepted_rate"], label="Accepted rate")
    plt.xlabel("PERSON decision threshold")
    plt.ylabel("Metric value")
    plt.title("Threshold-dependent classifier behavior")
    plt.legend()
    plt.tight_layout()

    path = os.path.join(output_dir, "threshold_precision_recall_f1.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {path}")


def plot_score_distribution_by_class(y_true, y_score, output_dir: str):
    df = pd.DataFrame({
        "label": y_true,
        "score": y_score,
    })

    plt.figure(figsize=(9, 6))

    plt.hist(
        df[df["label"] == 0]["score"],
        bins=40,
        alpha=0.6,
        label="True NON_PERSON",
    )

    plt.hist(
        df[df["label"] == 1]["score"],
        bins=40,
        alpha=0.6,
        label="True PERSON",
    )

    plt.xlabel("Model-estimated probability of PERSON")
    plt.ylabel("Number of labelled candidates")
    plt.title("Score distribution by true class")
    plt.legend()
    plt.tight_layout()

    path = os.path.join(output_dir, "score_distribution_by_class.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {path}")


def plot_calibration_curve(y_true, y_score, output_dir: str):
    prob_true, prob_pred = calibration_curve(
        y_true,
        y_score,
        n_bins=10,
        strategy="uniform",
    )

    plt.figure(figsize=(7, 6))
    plt.plot(prob_pred, prob_true, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed fraction of PERSON")
    plt.title("Calibration curve")
    plt.legend()
    plt.tight_layout()

    path = os.path.join(output_dir, "calibration_curve.png")
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {path}")


def export_error_examples(
    df_eval: pd.DataFrame,
    y_true,
    y_pred,
    y_score,
    output_dir: str,
):
    result = df_eval.copy()
    result["true_label"] = y_true
    result["predicted_label"] = y_pred
    result["person_score"] = y_score

    result["true_label_name"] = result["true_label"].map({
        0: "NON_PERSON",
        1: "PERSON",
    })

    result["predicted_label_name"] = result["predicted_label"].map({
        0: "NON_PERSON",
        1: "PERSON",
    })

    errors = result[result["true_label"] != result["predicted_label"]].copy()

    false_positives = errors[
        (errors["true_label"] == 0) & (errors["predicted_label"] == 1)
    ].sort_values("person_score", ascending=False)

    false_negatives = errors[
        (errors["true_label"] == 1) & (errors["predicted_label"] == 0)
    ].sort_values("person_score", ascending=True)

    path_all = os.path.join(output_dir, "error_examples_all.csv")
    path_fp = os.path.join(output_dir, "error_examples_false_positives.csv")
    path_fn = os.path.join(output_dir, "error_examples_false_negatives.csv")

    errors.to_csv(path_all, index=False)
    false_positives.to_csv(path_fp, index=False)
    false_negatives.to_csv(path_fn, index=False)

    print(f"Saved: {path_all}")
    print(f"Saved: {path_fp}")
    print(f"Saved: {path_fn}")


def export_predictions(
    df_eval: pd.DataFrame,
    y_true,
    y_pred,
    y_score,
    output_dir: str,
):
    result = df_eval.copy()
    result["true_label"] = y_true
    result["predicted_label"] = y_pred
    result["person_score"] = y_score

    result["true_label_name"] = result["true_label"].map({
        0: "NON_PERSON",
        1: "PERSON",
    })

    result["predicted_label_name"] = result["predicted_label"].map({
        0: "NON_PERSON",
        1: "PERSON",
    })

    path = os.path.join(output_dir, "all_test_predictions.csv")
    result.to_csv(path, index=False)

    print(f"Saved: {path}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the supervised person-name classifier."
    )

    parser.add_argument(
        "--label-file",
        required=True,
        help="CSV with manually labelled candidates. Must contain candidate,label.",
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Trained joblib model, e.g. author_person_classifier.joblib.",
    )

    parser.add_argument(
        "--output-dir",
        default="author_model_evaluation",
        help="Directory for plots and reports.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.90,
        help="Decision threshold for PERSON.",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.20,
        help="Test split size.",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for train/test split reproduction.",
    )

    parser.add_argument(
        "--use-full-label-file-as-test",
        action="store_true",
        help=(
            "Use the entire label file as the test set. "
            "Use this only if the file was NOT used for training."
        ),
    )

    args = parser.parse_args()

    make_output_dir(args.output_dir)

    print("Loading labelled data...")
    df = load_labelled_data(args.label_file)

    print(f"Loaded labelled candidates: {len(df):,}")
    print(df["label"].value_counts().rename(index={0: "NON_PERSON", 1: "PERSON"}))

    if args.use_full_label_file_as_test:
        test_df = df.copy()
        print("\nUsing the full label file as test data.")
        print("Only do this if this file was not used for model training.")
    else:
        _, test_df = train_test_split(
            df,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=df["label"],
        )

        print(f"\nUsing a stratified test split: {len(test_df):,} rows")

    print("\nLoading trained model...")
    model = joblib.load(args.model)
    print("Model loaded successfully.")

    X_test = prepare_X(test_df)
    y_true = test_df["label"].values

    print("\nGenerating predictions...")
    y_score = get_person_probabilities(model, X_test)
    y_pred = (y_score >= args.threshold).astype(int)

    print("\nSaving evaluation outputs...")

    save_metrics_report(
        y_true=y_true,
        y_pred=y_pred,
        y_score=y_score,
        output_dir=args.output_dir,
        threshold=args.threshold,
    )

    plot_confusion_matrix(y_true, y_pred, args.output_dir)
    plot_normalized_confusion_matrix(y_true, y_pred, args.output_dir)
    plot_roc_curve(y_true, y_score, args.output_dir)
    plot_precision_recall_curve(y_true, y_score, args.output_dir)
    plot_threshold_metrics(y_true, y_score, args.output_dir)
    plot_score_distribution_by_class(y_true, y_score, args.output_dir)
    plot_calibration_curve(y_true, y_score, args.output_dir)

    export_error_examples(
        df_eval=test_df,
        y_true=y_true,
        y_pred=y_pred,
        y_score=y_score,
        output_dir=args.output_dir,
    )

    export_predictions(
        df_eval=test_df,
        y_true=y_true,
        y_pred=y_pred,
        y_score=y_score,
        output_dir=args.output_dir,
    )

    print("\nDone.")
    print(f"Evaluation outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()