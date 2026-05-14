import argparse
import os
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


# ---------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------

def load_csv_if_exists(path: Optional[str]) -> Optional[pd.DataFrame]:
    if path is None:
        return None

    if not os.path.exists(path):
        print(f"[WARNING] File not found: {path}")
        return None

    df = pd.read_csv(path)
    print(f"Loaded {path}: {len(df):,} rows")
    return df


def save_or_show(output_dir: Optional[str], filename: str):
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, filename)
        plt.savefig(path, dpi=300, bbox_inches="tight")
        print(f"Saved: {path}")
        plt.close()
    else:
        plt.show()


def print_section(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# ---------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------

def table_top_accepted_authors(author_counts: pd.DataFrame, top_n: int = 30):
    print_section(f"Top {top_n} accepted authors")

    if author_counts is None or author_counts.empty:
        print("No accepted-author data available.")
        return

    print(author_counts.head(top_n).to_string(index=False))


def table_top_uncertain_candidates(uncertain: pd.DataFrame, top_n: int = 30):
    print_section(f"Top {top_n} uncertain candidates by frequency")

    if uncertain is None or uncertain.empty:
        print("No uncertain-candidate data available.")
        return

    grouped = (
        uncertain
        .groupby("candidate")
        .agg(
            frequency=("candidate", "size"),
            mean_score=("score", "mean"),
            min_score=("score", "min"),
            max_score=("score", "max"),
            most_common_reason=("reason", lambda x: x.mode().iloc[0] if len(x.mode()) else ""),
            most_common_ner=("ner_label", lambda x: x.mode().iloc[0] if len(x.mode()) else ""),
        )
        .reset_index()
        .sort_values(["frequency", "mean_score"], ascending=[False, False])
    )

    print(grouped.head(top_n).to_string(index=False))


def table_top_rejected_candidates(rejected: pd.DataFrame, top_n: int = 30):
    print_section(f"Top {top_n} rejected candidates by frequency")

    if rejected is None or rejected.empty:
        print("No rejected-candidate data available.")
        return

    grouped = (
        rejected
        .groupby("candidate")
        .agg(
            frequency=("candidate", "size"),
            mean_score=("score", "mean") if "score" in rejected.columns else ("candidate", "size"),
            most_common_reason=("reason", lambda x: x.mode().iloc[0] if len(x.mode()) else "")
            if "reason" in rejected.columns else ("candidate", "size"),
        )
        .reset_index()
        .sort_values("frequency", ascending=False)
    )

    print(grouped.head(top_n).to_string(index=False))


def save_summary_tables(
    author_counts: Optional[pd.DataFrame],
    uncertain: Optional[pd.DataFrame],
    rejected: Optional[pd.DataFrame],
    output_dir: str,
    top_n: int = 100,
):
    os.makedirs(output_dir, exist_ok=True)

    if author_counts is not None and not author_counts.empty:
        path = os.path.join(output_dir, "table_top_accepted_authors.csv")
        author_counts.head(top_n).to_csv(path, index=False)
        print(f"Saved: {path}")

    if uncertain is not None and not uncertain.empty:
        grouped_uncertain = (
            uncertain
            .groupby("candidate")
            .agg(
                frequency=("candidate", "size"),
                mean_score=("score", "mean"),
                min_score=("score", "min"),
                max_score=("score", "max"),
                most_common_reason=("reason", lambda x: x.mode().iloc[0] if len(x.mode()) else ""),
                most_common_ner=("ner_label", lambda x: x.mode().iloc[0] if len(x.mode()) else ""),
            )
            .reset_index()
            .sort_values(["frequency", "mean_score"], ascending=[False, False])
        )

        path = os.path.join(output_dir, "table_top_uncertain_candidates.csv")
        grouped_uncertain.head(top_n).to_csv(path, index=False)
        print(f"Saved: {path}")

    if rejected is not None and not rejected.empty:
        aggregations = {
            "frequency": ("candidate", "size"),
        }

        if "score" in rejected.columns:
            aggregations["mean_score"] = ("score", "mean")

        if "reason" in rejected.columns:
            aggregations["most_common_reason"] = (
                "reason",
                lambda x: x.mode().iloc[0] if len(x.mode()) else ""
            )

        grouped_rejected = (
            rejected
            .groupby("candidate")
            .agg(**aggregations)
            .reset_index()
            .sort_values("frequency", ascending=False)
        )

        path = os.path.join(output_dir, "table_top_rejected_candidates.csv")
        grouped_rejected.head(top_n).to_csv(path, index=False)
        print(f"Saved: {path}")


# ---------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------

def plot_top_accepted_authors(
    author_counts: Optional[pd.DataFrame],
    top_n: int = 30,
    output_dir: Optional[str] = None,
):
    if author_counts is None or author_counts.empty:
        print("Skipping top accepted authors plot.")
        return

    df = author_counts.head(top_n).copy()

    plt.figure(figsize=(14, 8))
    plt.barh(df["author"][::-1], df["count"][::-1])
    plt.xlabel("Frequency")
    plt.ylabel("Accepted author")
    plt.title(f"Top {top_n} accepted person-name candidates")
    plt.tight_layout()
    save_or_show(output_dir, "top_accepted_authors.png")


def plot_author_frequency_distribution(
    author_counts: Optional[pd.DataFrame],
    output_dir: Optional[str] = None,
):
    if author_counts is None or author_counts.empty:
        print("Skipping author frequency distribution plot.")
        return

    counts = author_counts["count"]

    plt.figure(figsize=(10, 6))
    plt.hist(counts, bins=50)
    plt.xlabel("Author candidate frequency")
    plt.ylabel("Number of unique accepted authors")
    plt.title("Distribution of accepted-author frequencies")
    plt.yscale("log")
    plt.tight_layout()
    save_or_show(output_dir, "accepted_author_frequency_distribution.png")


def plot_uncertain_score_distribution(
    uncertain: Optional[pd.DataFrame],
    output_dir: Optional[str] = None,
):
    if uncertain is None or uncertain.empty or "score" not in uncertain.columns:
        print("Skipping uncertain score distribution plot.")
        return

    plt.figure(figsize=(10, 6))
    plt.hist(uncertain["score"].dropna(), bins=40)
    plt.xlabel("Model-estimated probability of PERSON")
    plt.ylabel("Number of uncertain candidate occurrences")
    plt.title("Distribution of scores among uncertain candidates")
    plt.tight_layout()
    save_or_show(output_dir, "uncertain_score_distribution.png")


def plot_top_uncertain_candidates(
    uncertain: Optional[pd.DataFrame],
    top_n: int = 30,
    output_dir: Optional[str] = None,
):
    if uncertain is None or uncertain.empty:
        print("Skipping top uncertain candidates plot.")
        return

    grouped = (
        uncertain
        .groupby("candidate")
        .size()
        .reset_index(name="frequency")
        .sort_values("frequency", ascending=False)
        .head(top_n)
    )

    plt.figure(figsize=(14, 8))
    plt.barh(grouped["candidate"][::-1], grouped["frequency"][::-1])
    plt.xlabel("Frequency")
    plt.ylabel("Uncertain candidate")
    plt.title(f"Top {top_n} uncertain candidates")
    plt.tight_layout()
    save_or_show(output_dir, "top_uncertain_candidates.png")


def plot_uncertain_reasons(
    uncertain: Optional[pd.DataFrame],
    output_dir: Optional[str] = None,
):
    if uncertain is None or uncertain.empty or "reason" not in uncertain.columns:
        print("Skipping uncertain reasons plot.")
        return

    reason_counts = (
        uncertain["reason"]
        .fillna("missing")
        .value_counts()
        .head(20)
    )

    plt.figure(figsize=(12, 7))
    plt.barh(reason_counts.index[::-1], reason_counts.values[::-1])
    plt.xlabel("Number of uncertain candidate occurrences")
    plt.ylabel("Reason")
    plt.title("Most common reasons for uncertainty")
    plt.tight_layout()
    save_or_show(output_dir, "uncertain_reasons.png")


def plot_ner_label_distribution(
    uncertain: Optional[pd.DataFrame],
    rejected: Optional[pd.DataFrame],
    output_dir: Optional[str] = None,
):
    frames = []

    if uncertain is not None and not uncertain.empty and "ner_label" in uncertain.columns:
        temp = uncertain.copy()
        temp["decision"] = "UNCERTAIN"
        frames.append(temp[["ner_label", "decision"]])

    if rejected is not None and not rejected.empty and "ner_label" in rejected.columns:
        temp = rejected.copy()
        temp["decision"] = "REJECTED"
        frames.append(temp[["ner_label", "decision"]])

    if not frames:
        print("Skipping NER label distribution plot.")
        return

    df = pd.concat(frames, ignore_index=True)
    df["ner_label"] = df["ner_label"].fillna("NO_NER_LABEL")

    pivot = (
        df
        .groupby(["decision", "ner_label"])
        .size()
        .reset_index(name="count")
    )

    for decision in pivot["decision"].unique():
        sub = pivot[pivot["decision"] == decision].sort_values("count", ascending=False)

        plt.figure(figsize=(10, 6))
        plt.barh(sub["ner_label"][::-1], sub["count"][::-1])
        plt.xlabel("Count")
        plt.ylabel("NER label")
        plt.title(f"NER label distribution for {decision} candidates")
        plt.tight_layout()
        save_or_show(output_dir, f"ner_label_distribution_{decision.lower()}.png")


def plot_rejected_candidates(
    rejected: Optional[pd.DataFrame],
    top_n: int = 30,
    output_dir: Optional[str] = None,
):
    if rejected is None or rejected.empty:
        print("Skipping rejected candidates plot.")
        return

    grouped = (
        rejected
        .groupby("candidate")
        .size()
        .reset_index(name="frequency")
        .sort_values("frequency", ascending=False)
        .head(top_n)
    )

    plt.figure(figsize=(14, 8))
    plt.barh(grouped["candidate"][::-1], grouped["frequency"][::-1])
    plt.xlabel("Frequency")
    plt.ylabel("Rejected candidate")
    plt.title(f"Top {top_n} rejected candidates")
    plt.tight_layout()
    save_or_show(output_dir, "top_rejected_candidates.png")


def plot_decision_overview(
    author_counts: Optional[pd.DataFrame],
    uncertain: Optional[pd.DataFrame],
    rejected: Optional[pd.DataFrame],
    output_dir: Optional[str] = None,
):
    values = {}

    if author_counts is not None and not author_counts.empty:
        values["ACCEPTED_PERSON_UNIQUE"] = len(author_counts)
        values["ACCEPTED_PERSON_OCCURRENCES"] = int(author_counts["count"].sum())

    if uncertain is not None and not uncertain.empty:
        values["UNCERTAIN_OCCURRENCES"] = len(uncertain)
        values["UNCERTAIN_UNIQUE"] = uncertain["candidate"].nunique()

    if rejected is not None and not rejected.empty:
        values["REJECTED_OCCURRENCES"] = len(rejected)
        values["REJECTED_UNIQUE"] = rejected["candidate"].nunique()

    if not values:
        print("Skipping decision overview plot.")
        return

    plt.figure(figsize=(12, 6))
    plt.bar(values.keys(), values.values())
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("Count")
    plt.title("Author-cleaning decision overview")
    plt.tight_layout()
    save_or_show(output_dir, "decision_overview.png")


def plot_uncertain_score_by_frequency(
    uncertain: Optional[pd.DataFrame],
    output_dir: Optional[str] = None,
):
    if uncertain is None or uncertain.empty or "score" not in uncertain.columns:
        print("Skipping uncertain score by frequency plot.")
        return

    grouped = (
        uncertain
        .groupby("candidate")
        .agg(
            frequency=("candidate", "size"),
            mean_score=("score", "mean"),
        )
        .reset_index()
    )

    plt.figure(figsize=(10, 7))
    plt.scatter(grouped["mean_score"], grouped["frequency"], alpha=0.6)
    plt.xlabel("Mean model-estimated probability of PERSON")
    plt.ylabel("Candidate frequency")
    plt.yscale("log")
    plt.title("Uncertain candidates: frequency vs. person-probability score")
    plt.tight_layout()
    save_or_show(output_dir, "uncertain_score_by_frequency.png")


# ---------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------

def print_summary(
    author_counts: Optional[pd.DataFrame],
    uncertain: Optional[pd.DataFrame],
    rejected: Optional[pd.DataFrame],
):
    print_section("Summary")

    if author_counts is not None and not author_counts.empty:
        unique_authors = len(author_counts)
        total_accepted_occurrences = int(author_counts["count"].sum())

        print(f"Unique accepted person-name candidates: {unique_authors:,}")
        print(f"Total accepted person-name occurrences: {total_accepted_occurrences:,}")

        print("\nAccepted-author count distribution:")
        print(author_counts["count"].describe().to_string())

    if uncertain is not None and not uncertain.empty:
        print(f"\nUncertain candidate occurrences: {len(uncertain):,}")
        print(f"Unique uncertain candidates: {uncertain['candidate'].nunique():,}")

        if "score" in uncertain.columns:
            print("\nUncertain score distribution:")
            print(uncertain["score"].describe().to_string())

    if rejected is not None and not rejected.empty:
        print(f"\nRejected candidate occurrences: {len(rejected):,}")
        print(f"Unique rejected candidates: {rejected['candidate'].nunique():,}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Visualize results of the arXiv author-name cleaning pipeline."
    )

    parser.add_argument(
        "--author-counts",
        default="author_counts_full.csv",
        help="CSV with accepted author counts. Expected columns: author,count",
    )

    parser.add_argument(
        "--uncertain",
        default="uncertain_full.csv",
        help="CSV with uncertain candidates.",
    )

    parser.add_argument(
        "--rejected",
        default=None,
        help="Optional CSV with rejected candidates.",
    )

    parser.add_argument(
        "--output-dir",
        default="author_cleaning_visualizations",
        help="Directory where plots and summary tables will be saved.",
    )

    parser.add_argument(
        "--top-n",
        type=int,
        default=30,
        help="Number of top candidates to show in charts.",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plots interactively instead of saving them.",
    )

    args = parser.parse_args()

    output_dir = None if args.show else args.output_dir

    author_counts = load_csv_if_exists(args.author_counts)
    uncertain = load_csv_if_exists(args.uncertain)
    rejected = load_csv_if_exists(args.rejected)

    print_summary(author_counts, uncertain, rejected)

    table_top_accepted_authors(author_counts, top_n=args.top_n)
    table_top_uncertain_candidates(uncertain, top_n=args.top_n)
    table_top_rejected_candidates(rejected, top_n=args.top_n)

    if output_dir:
        save_summary_tables(
            author_counts=author_counts,
            uncertain=uncertain,
            rejected=rejected,
            output_dir=output_dir,
            top_n=100,
        )

    plot_decision_overview(author_counts, uncertain, rejected, output_dir)
    plot_top_accepted_authors(author_counts, top_n=args.top_n, output_dir=output_dir)
    plot_author_frequency_distribution(author_counts, output_dir)
    plot_uncertain_score_distribution(uncertain, output_dir)
    plot_top_uncertain_candidates(uncertain, top_n=args.top_n, output_dir=output_dir)
    plot_uncertain_reasons(uncertain, output_dir)
    plot_uncertain_score_by_frequency(uncertain, output_dir)
    plot_ner_label_distribution(uncertain, rejected, output_dir)
    plot_rejected_candidates(rejected, top_n=args.top_n, output_dir=output_dir)

    print_section("Done")
    if output_dir:
        print(f"Visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()