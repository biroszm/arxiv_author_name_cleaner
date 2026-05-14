import argparse
import json
import os
import re
import sys
import unicodedata
from collections import Counter
from typing import Optional, List, Dict, Tuple

import joblib
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


# ---------------------------------------------------------------------
# IMPORTANT JOBLIB FIX
# ---------------------------------------------------------------------
# The trained model depends on custom objects from clean_arxiv_authors.py.
# Therefore clean_arxiv_authors.py must be in the same folder.
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


# ---------------------------------------------------------------------
# Optional spaCy
# ---------------------------------------------------------------------

def load_spacy_model(enabled: bool):
    if not enabled:
        return None

    try:
        import spacy
        return spacy.load("en_core_web_sm")
    except Exception as exc:
        print("[WARNING] spaCy requested but could not be loaded.")
        print(f"[WARNING] Reason: {exc}")
        return None


# ---------------------------------------------------------------------
# Text cleaning and splitting
# ---------------------------------------------------------------------

def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", str(text))


def normalize_spaces(text: str) -> str:
    return " ".join(str(text).strip().split())


def remove_parenthetical_content(text: str) -> str:
    return re.sub(r"\([^)]*\)", " ", text)


def clean_raw_author_field(text: str) -> str:
    text = normalize_unicode(text)
    text = remove_parenthetical_content(text)
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")
    text = normalize_spaces(text)
    return text


def split_author_candidates(authors_raw: str) -> List[str]:
    text = clean_raw_author_field(authors_raw)

    text = re.sub(r"\s*[;|]\s*", ",", text)
    text = re.sub(r"\s+\band\b\s+", ",", text, flags=re.IGNORECASE)

    raw_candidates = [normalize_spaces(x) for x in text.split(",")]

    candidates = []
    for cand in raw_candidates:
        cand = cand.strip(" .;:-")
        cand = normalize_spaces(cand)

        if cand:
            candidates.append(cand)

    return candidates


# ---------------------------------------------------------------------
# NER helper
# ---------------------------------------------------------------------

def get_ner_label(candidate: str, nlp=None) -> Optional[str]:
    if nlp is None:
        return None

    doc = nlp(candidate)
    labels = [ent.label_ for ent in doc.ents]

    if "PERSON" in labels:
        return "PERSON"

    if "ORG" in labels:
        return "ORG"

    if "GPE" in labels or "LOC" in labels:
        return "LOCATION"

    return None


# ---------------------------------------------------------------------
# Data reading
# ---------------------------------------------------------------------

def iter_arxiv_records(
    file_path: str,
    max_records: Optional[int] = None,
    start_record: int = 0,
):
    """
    Reads JSONL arXiv records.

    start_record lets you use a later part of the file as a pseudo-test subset.
    Example:
        start_record = 100000
        max_records = 50000
    """
    yielded = 0

    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < start_record:
                continue

            if max_records is not None and yielded >= max_records:
                break

            line = line.strip()

            if not line:
                continue

            try:
                yield json.loads(line)
                yielded += 1
            except json.JSONDecodeError:
                continue


def collect_scored_candidates(
    file_path: str,
    model,
    max_records: Optional[int],
    start_record: int,
    use_spacy: bool,
) -> pd.DataFrame:
    """
    Reads a fixed test subset and computes p(person) for every candidate.

    Important:
        Scores are computed once.
        Threshold variations are applied afterward.
    """
    nlp = load_spacy_model(use_spacy)

    classes = list(model.classes_)

    if 1 not in classes:
        raise ValueError("Model classes do not contain label 1 for PERSON.")

    person_index = classes.index(1)

    rows = []

    records = iter_arxiv_records(
        file_path=file_path,
        max_records=max_records,
        start_record=start_record,
    )

    for record in tqdm(records, desc="Scoring candidates"):
        arxiv_id = record.get("id", "")
        authors_raw = record.get("authors", "")

        if not authors_raw:
            continue

        candidates = split_author_candidates(authors_raw)

        for position, candidate in enumerate(candidates):
            ner_label = get_ner_label(candidate, nlp=nlp)

            X = [{
                "candidate": candidate,
                "ner_label": ner_label,
            }]

            person_score = float(model.predict_proba(X)[0][person_index])

            rows.append({
                "arxiv_id": arxiv_id,
                "candidate": candidate,
                "position": position,
                "ner_label": ner_label,
                "person_score": person_score,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Threshold classification
# ---------------------------------------------------------------------

def classify_by_thresholds(
    score: float,
    person_threshold: float,
    nonperson_threshold: float,
) -> str:
    if score >= person_threshold:
        return "PERSON"

    if score <= nonperson_threshold:
        return "NON_PERSON"

    return "UNCERTAIN"


def apply_threshold_setting(
    scored_df: pd.DataFrame,
    person_threshold: float,
    nonperson_threshold: float,
) -> pd.DataFrame:
    df = scored_df.copy()

    df["decision"] = df["person_score"].apply(
        lambda s: classify_by_thresholds(
            s,
            person_threshold=person_threshold,
            nonperson_threshold=nonperson_threshold,
        )
    )

    df["person_threshold"] = person_threshold
    df["nonperson_threshold"] = nonperson_threshold

    return df


def make_threshold_tag(person_threshold: float, nonperson_threshold: float) -> str:
    p = str(person_threshold).replace(".", "")
    n = str(nonperson_threshold).replace(".", "")
    return f"p{p}_np{n}"


# ---------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------

def summarize_threshold_result(
    df: pd.DataFrame,
    person_threshold: float,
    nonperson_threshold: float,
) -> Dict:
    decision_counts = df["decision"].value_counts().to_dict()

    person_df = df[df["decision"] == "PERSON"]
    nonperson_df = df[df["decision"] == "NON_PERSON"]
    uncertain_df = df[df["decision"] == "UNCERTAIN"]

    return {
        "person_threshold": person_threshold,
        "nonperson_threshold": nonperson_threshold,
        "total_candidate_occurrences": len(df),

        "person_occurrences": decision_counts.get("PERSON", 0),
        "nonperson_occurrences": decision_counts.get("NON_PERSON", 0),
        "uncertain_occurrences": decision_counts.get("UNCERTAIN", 0),

        "person_unique": person_df["candidate"].nunique(),
        "nonperson_unique": nonperson_df["candidate"].nunique(),
        "uncertain_unique": uncertain_df["candidate"].nunique(),

        "person_occurrence_rate": decision_counts.get("PERSON", 0) / len(df) if len(df) else 0,
        "nonperson_occurrence_rate": decision_counts.get("NON_PERSON", 0) / len(df) if len(df) else 0,
        "uncertain_occurrence_rate": decision_counts.get("UNCERTAIN", 0) / len(df) if len(df) else 0,
    }


def aggregate_uncertain_candidates(df: pd.DataFrame) -> pd.DataFrame:
    uncertain = df[df["decision"] == "UNCERTAIN"].copy()

    if uncertain.empty:
        return pd.DataFrame(columns=[
            "candidate",
            "frequency",
            "mean_person_score",
            "min_person_score",
            "max_person_score",
            "most_common_ner_label",
        ])

    grouped = (
        uncertain
        .groupby("candidate")
        .agg(
            frequency=("candidate", "size"),
            mean_person_score=("person_score", "mean"),
            min_person_score=("person_score", "min"),
            max_person_score=("person_score", "max"),
            most_common_ner_label=(
                "ner_label",
                lambda x: x.mode().iloc[0] if len(x.mode()) else "",
            ),
        )
        .reset_index()
        .sort_values(["frequency", "mean_person_score"], ascending=[False, False])
    )

    return grouped


def aggregate_accepted_candidates(df: pd.DataFrame) -> pd.DataFrame:
    accepted = df[df["decision"] == "PERSON"].copy()

    if accepted.empty:
        return pd.DataFrame(columns=[
            "candidate",
            "frequency",
            "mean_person_score",
            "min_person_score",
            "max_person_score",
            "most_common_ner_label",
        ])

    grouped = (
        accepted
        .groupby("candidate")
        .agg(
            frequency=("candidate", "size"),
            mean_person_score=("person_score", "mean"),
            min_person_score=("person_score", "min"),
            max_person_score=("person_score", "max"),
            most_common_ner_label=(
                "ner_label",
                lambda x: x.mode().iloc[0] if len(x.mode()) else "",
            ),
        )
        .reset_index()
        .sort_values(["frequency", "mean_person_score"], ascending=[False, False])
    )

    return grouped


def aggregate_rejected_candidates(df: pd.DataFrame) -> pd.DataFrame:
    rejected = df[df["decision"] == "NON_PERSON"].copy()

    if rejected.empty:
        return pd.DataFrame(columns=[
            "candidate",
            "frequency",
            "mean_person_score",
            "min_person_score",
            "max_person_score",
            "most_common_ner_label",
        ])

    grouped = (
        rejected
        .groupby("candidate")
        .agg(
            frequency=("candidate", "size"),
            mean_person_score=("person_score", "mean"),
            min_person_score=("person_score", "min"),
            max_person_score=("person_score", "max"),
            most_common_ner_label=(
                "ner_label",
                lambda x: x.mode().iloc[0] if len(x.mode()) else "",
            ),
        )
        .reset_index()
        .sort_values(["frequency", "mean_person_score"], ascending=[False, False])
    )

    return grouped


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def plot_top_uncertain_candidates(
    uncertain_grouped: pd.DataFrame,
    output_path: str,
    title: str,
    top_n: int = 30,
):
    if uncertain_grouped.empty:
        plt.figure(figsize=(10, 4))
        plt.text(
            0.5,
            0.5,
            "No uncertain candidates for this threshold setting.",
            ha="center",
            va="center",
            fontsize=12,
        )
        plt.axis("off")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()
        return

    df = uncertain_grouped.head(top_n).copy()

    labels = df["candidate"][::-1]
    values = df["frequency"][::-1]

    plt.figure(figsize=(14, 9))
    plt.barh(labels, values)
    plt.xlabel("Frequency in uncertain bucket")
    plt.ylabel("Candidate")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_uncertain_score_distribution(
    df_with_decisions: pd.DataFrame,
    output_path: str,
    title: str,
):
    uncertain = df_with_decisions[df_with_decisions["decision"] == "UNCERTAIN"]

    plt.figure(figsize=(9, 6))

    if uncertain.empty:
        plt.text(
            0.5,
            0.5,
            "No uncertain candidates.",
            ha="center",
            va="center",
            fontsize=12,
        )
        plt.axis("off")
    else:
        plt.hist(uncertain["person_score"], bins=40)
        plt.xlabel("Model-estimated probability of PERSON")
        plt.ylabel("Number of uncertain candidate occurrences")

    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_decision_counts_across_thresholds(summary_df: pd.DataFrame, output_path: str):
    plt.figure(figsize=(10, 6))

    x_labels = [
        f"{row.person_threshold:.2f}/{row.nonperson_threshold:.2f}"
        for _, row in summary_df.iterrows()
    ]

    x = range(len(summary_df))

    plt.plot(x, summary_df["person_occurrences"], marker="o", label="PERSON")
    plt.plot(x, summary_df["nonperson_occurrences"], marker="o", label="NON_PERSON")
    plt.plot(x, summary_df["uncertain_occurrences"], marker="o", label="UNCERTAIN")

    plt.xticks(x, x_labels, rotation=45, ha="right")
    plt.xlabel("person_threshold / nonperson_threshold")
    plt.ylabel("Candidate occurrences")
    plt.title("Decision counts across threshold settings")
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_uncertain_unique_across_thresholds(summary_df: pd.DataFrame, output_path: str):
    plt.figure(figsize=(10, 6))

    x_labels = [
        f"{row.person_threshold:.2f}/{row.nonperson_threshold:.2f}"
        for _, row in summary_df.iterrows()
    ]

    x = range(len(summary_df))

    plt.plot(
        x,
        summary_df["uncertain_unique"],
        marker="o",
        label="Unique uncertain candidates",
    )

    plt.xticks(x, x_labels, rotation=45, ha="right")
    plt.xlabel("person_threshold / nonperson_threshold")
    plt.ylabel("Unique uncertain candidates")
    plt.title("Unique uncertain candidates across threshold settings")
    plt.legend()
    plt.tight_layout()

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------

def parse_threshold_pairs(raw: str) -> List[Tuple[float, float]]:
    """
    Example:
        "0.95:0.05,0.90:0.10,0.80:0.20"
    """
    pairs = []

    for item in raw.split(","):
        item = item.strip()

        if not item:
            continue

        if ":" not in item:
            raise ValueError(
                "Threshold pairs must use format person:nonperson, "
                "e.g. 0.90:0.10"
            )

        p_str, n_str = item.split(":", 1)

        p = float(p_str)
        n = float(n_str)

        if not (0 <= n < p <= 1):
            raise ValueError(
                f"Invalid threshold pair {p}:{n}. "
                "Expected 0 <= nonperson_threshold < person_threshold <= 1."
            )

        pairs.append((p, n))

    return pairs


def run_threshold_sensitivity(
    file_path: str,
    model_path: str,
    output_dir: str,
    max_records: Optional[int],
    start_record: int,
    threshold_pairs: List[Tuple[float, float]],
    top_n: int,
    use_spacy: bool,
    save_all_candidate_scores: bool,
):
    os.makedirs(output_dir, exist_ok=True)

    print("Loading model...")
    model = joblib.load(model_path)
    print("Model loaded successfully.")

    print("\nScoring fixed test subset...")
    scored_df = collect_scored_candidates(
        file_path=file_path,
        model=model,
        max_records=max_records,
        start_record=start_record,
        use_spacy=use_spacy,
    )

    if scored_df.empty:
        raise ValueError("No candidates were extracted from the selected test subset.")

    scored_path = os.path.join(output_dir, "scored_candidates_test_subset.csv")

    if save_all_candidate_scores:
        scored_df.to_csv(scored_path, index=False)
        print(f"Saved scored candidates: {scored_path}")

    print(f"\nCandidate occurrences scored: {len(scored_df):,}")
    print(f"Unique candidates scored: {scored_df['candidate'].nunique():,}")

    summary_rows = []

    for person_threshold, nonperson_threshold in threshold_pairs:
        tag = make_threshold_tag(person_threshold, nonperson_threshold)

        setting_dir = os.path.join(output_dir, tag)
        os.makedirs(setting_dir, exist_ok=True)

        print("\n" + "=" * 80)
        print(f"Threshold setting: PERSON >= {person_threshold}, NON_PERSON <= {nonperson_threshold}")
        print("=" * 80)

        df = apply_threshold_setting(
            scored_df=scored_df,
            person_threshold=person_threshold,
            nonperson_threshold=nonperson_threshold,
        )

        summary = summarize_threshold_result(
            df=df,
            person_threshold=person_threshold,
            nonperson_threshold=nonperson_threshold,
        )
        summary_rows.append(summary)

        print("Decision counts:")
        print(df["decision"].value_counts().to_string())

        uncertain_grouped = aggregate_uncertain_candidates(df)
        accepted_grouped = aggregate_accepted_candidates(df)
        rejected_grouped = aggregate_rejected_candidates(df)

        # Save CSVs for manual inspection.
        uncertain_csv = os.path.join(setting_dir, f"top_uncertain_candidates_{tag}.csv")
        accepted_csv = os.path.join(setting_dir, f"top_accepted_candidates_{tag}.csv")
        rejected_csv = os.path.join(setting_dir, f"top_rejected_candidates_{tag}.csv")

        uncertain_grouped.to_csv(uncertain_csv, index=False)
        accepted_grouped.to_csv(accepted_csv, index=False)
        rejected_grouped.to_csv(rejected_csv, index=False)

        print(f"Saved: {uncertain_csv}")
        print(f"Saved: {accepted_csv}")
        print(f"Saved: {rejected_csv}")

        # Save full uncertain occurrences too.
        uncertain_occurrences = df[df["decision"] == "UNCERTAIN"].copy()
        uncertain_occurrences_csv = os.path.join(
            setting_dir,
            f"uncertain_occurrences_{tag}.csv",
        )
        uncertain_occurrences.to_csv(uncertain_occurrences_csv, index=False)
        print(f"Saved: {uncertain_occurrences_csv}")

        # Save PNG: top uncertain candidates.
        uncertain_png = os.path.join(setting_dir, f"top_uncertain_candidates_{tag}.png")
        plot_top_uncertain_candidates(
            uncertain_grouped=uncertain_grouped,
            output_path=uncertain_png,
            title=(
                f"Top uncertain candidates\n"
                f"PERSON >= {person_threshold}, NON_PERSON <= {nonperson_threshold}"
            ),
            top_n=top_n,
        )
        print(f"Saved: {uncertain_png}")

        # Save PNG: uncertain score distribution.
        score_png = os.path.join(setting_dir, f"uncertain_score_distribution_{tag}.png")
        plot_uncertain_score_distribution(
            df_with_decisions=df,
            output_path=score_png,
            title=(
                f"Uncertain score distribution\n"
                f"PERSON >= {person_threshold}, NON_PERSON <= {nonperson_threshold}"
            ),
        )
        print(f"Saved: {score_png}")

    summary_df = pd.DataFrame(summary_rows)

    summary_csv = os.path.join(output_dir, "threshold_sensitivity_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"\nSaved summary: {summary_csv}")

    decision_counts_png = os.path.join(output_dir, "decision_counts_across_thresholds.png")
    plot_decision_counts_across_thresholds(summary_df, decision_counts_png)
    print(f"Saved: {decision_counts_png}")

    uncertain_unique_png = os.path.join(output_dir, "uncertain_unique_across_thresholds.png")
    plot_uncertain_unique_across_thresholds(summary_df, uncertain_unique_png)
    print(f"Saved: {uncertain_unique_png}")

    print("\nDone.")
    print(f"All outputs saved to: {output_dir}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Threshold-sensitivity analysis for author person-name cleaning. "
            "Runs several person/nonperson threshold pairs on a fixed test subset "
            "and saves uncertain-candidate PNGs and CSVs."
        )
    )

    parser.add_argument(
        "--file",
        required=True,
        help="Path to arxiv-metadata-oai-snapshot.json",
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Path to trained model, e.g. author_person_classifier.joblib",
    )

    parser.add_argument(
        "--output-dir",
        default="threshold_sensitivity_uncertain",
        help="Directory where outputs will be saved.",
    )

    parser.add_argument(
        "--max-records",
        type=int,
        default=50000,
        help="Number of records to use as the fixed test subset.",
    )

    parser.add_argument(
        "--start-record",
        type=int,
        default=0,
        help=(
            "Record offset for the test subset. "
            "Use a later offset if you want a pseudo-test subset from another part of the file."
        ),
    )

    parser.add_argument(
        "--threshold-pairs",
        default="0.95:0.05,0.90:0.10,0.85:0.15,0.80:0.20,0.75:0.25,0.70:0.30",
        help=(
            "Comma-separated threshold pairs in format person:nonperson. "
            "Example: 0.90:0.10,0.80:0.20"
        ),
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=30,
        help="Number of top uncertain candidates to show in each PNG.",
    )

    parser.add_argument(
        "--use-spacy",
        action="store_true",
        help="Use spaCy NER labels during scoring, if available.",
    )

    parser.add_argument(
        "--save-all-candidate-scores",
        action="store_true",
        help="Save all scored candidate occurrences to CSV.",
    )

    args = parser.parse_args()

    threshold_pairs = parse_threshold_pairs(args.threshold_pairs)

    run_threshold_sensitivity(
        file_path=args.file,
        model_path=args.model,
        output_dir=args.output_dir,
        max_records=args.max_records,
        start_record=args.start_record,
        threshold_pairs=threshold_pairs,
        top_n=args.top_n,
        use_spacy=args.use_spacy,
        save_all_candidate_scores=args.save_all_candidate_scores,
    )


if __name__ == "__main__":
    main()