import argparse
import csv
import json
import math
import os
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

ORG_KEYWORDS = {
    "university", "université", "universidad", "universitat",
    "institute", "institut", "institution",
    "department", "dept", "faculty", "school", "college",
    "laboratory", "laboratories", "lab", "labs",
    "center", "centre", "zentrum", "centro",
    "group", "team", "collaboration", "consortium", "project",
    "foundation", "society", "association", "academy",
    "company", "corporation", "inc", "ltd", "llc", "gmbh", "srl", "sa",
    "research", "science", "sciences", "engineering",
    "physics", "mathematics", "biology", "chemistry", "medicine",
    "hospital", "clinic", "observatory", "laboratoire",
    "division", "unit", "program", "programme",
    "ministry", "agency", "administration",
    "collaboratory", "collaboration",
}

ORG_PHRASES = {
    "department of",
    "school of",
    "faculty of",
    "institute of",
    "university of",
    "center for",
    "centre for",
    "research group",
    "working group",
    "the collaboration",
    "the consortium",
    "national laboratory",
    "max planck",
    "cnrs",
    "cern",
    "nasa",
}

CONNECTOR_WORDS = {
    "of", "for", "the", "and", "in", "on", "at", "by", "with",
    "de", "del", "della", "di", "da", "dos", "das",
    "van", "von", "der", "den", "ten", "ter",
    "la", "le", "du",
}

# These are allowed inside personal names, but they are suspicious
# if the whole candidate looks institutional.
NAME_PARTICLES = {
    "de", "del", "della", "di", "da", "dos", "das",
    "van", "von", "der", "den", "ten", "ter",
    "la", "le", "du", "el", "al", "bin", "ibn",
}

COLLABORATION_WORDS = {
    "collaboration", "collaborations", "group", "team", "consortium",
    "working", "experiment", "survey", "project",
}

SUFFIXES = {
    "jr", "sr", "ii", "iii", "iv", "phd", "md"
}


# ---------------------------------------------------------------------
# Optional spaCy loading
# ---------------------------------------------------------------------

def load_spacy_model(enabled: bool):
    if not enabled:
        return None

    try:
        import spacy
        return spacy.load("en_core_web_sm")
    except Exception as exc:
        print(
            "[WARNING] spaCy requested but could not be loaded. "
            "Continuing without NER."
        )
        print(f"[WARNING] Reason: {exc}")
        return None


# ---------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------

def normalize_unicode(text: str) -> str:
    """
    Normalizes Unicode while preserving accented characters.
    """
    text = unicodedata.normalize("NFKC", text)
    return text


def normalize_spaces(text: str) -> str:
    return " ".join(text.strip().split())


def remove_parenthetical_content(text: str) -> str:
    """
    Removes affiliation-like or note-like parenthetical content.

    Example:
        John Smith (University of X) -> John Smith
    """
    return re.sub(r"\([^)]*\)", " ", text)


def clean_raw_author_field(text: str) -> str:
    text = normalize_unicode(text)
    text = remove_parenthetical_content(text)
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")
    text = normalize_spaces(text)
    return text


# ---------------------------------------------------------------------
# Candidate extraction
# ---------------------------------------------------------------------

def split_author_candidates(authors_raw: str) -> List[str]:
    """
    Splits the raw author field into possible entity/name candidates.

    arXiv author fields are often comma-separated, but this also handles
    semicolons, pipes, and some ' and ' cases.

    This is intentionally high-recall: it may produce organization names too.
    The classifier handles those later.
    """
    text = clean_raw_author_field(authors_raw)

    # Normalize common separators.
    text = re.sub(r"\s*[;|]\s*", ",", text)

    # Avoid splitting surname particles like "van der" etc. We only split
    # on explicit " and " when it looks like it separates two name-like chunks.
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
# Token-level helpers
# ---------------------------------------------------------------------

def strip_outer_punctuation(token: str) -> str:
    return token.strip(".,;:()[]{}<>\"“”‘’")


def is_initial(token: str) -> bool:
    token = strip_outer_punctuation(token)
    return bool(re.fullmatch(r"[A-Z]\.", token))


def is_alpha_name_piece(piece: str) -> bool:
    """
    Unicode-aware check for name components.

    Accepts:
        Smith
        Balázs
        Jean-Luc
        O'Connor
        D'Angelo

    Rejects:
        physics
        group2
        C++
    """
    piece = strip_outer_punctuation(piece)

    if not piece:
        return False

    # Remove allowed intra-name punctuation and check remaining chars.
    reduced = piece.replace("-", "").replace("'", "").replace("’", "")
    if not reduced:
        return False

    return all(ch.isalpha() for ch in reduced)


def tokenize_candidate(candidate: str) -> List[str]:
    candidate = normalize_spaces(candidate)
    tokens = [strip_outer_punctuation(t) for t in candidate.split()]
    return [t for t in tokens if t]


def lower_tokens(candidate: str) -> List[str]:
    return [t.lower().strip(".,;:()[]{}") for t in tokenize_candidate(candidate)]


def has_mixed_case_or_initials(candidate: str) -> bool:
    tokens = tokenize_candidate(candidate)
    for t in tokens:
        if is_initial(t):
            return True
        if len(t) >= 2 and t[0].isupper() and any(ch.islower() for ch in t[1:]):
            return True
    return False


# ---------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------

def extract_features(candidate: str, ner_label: Optional[str] = None) -> Dict[str, float]:
    """
    Extracts structural features for a candidate string.

    These features are used both by the rule scorer and by the supervised model.
    """
    c = normalize_spaces(candidate)
    c_lower = c.lower()
    tokens = tokenize_candidate(c)
    ltokens = [t.lower() for t in tokens]

    token_count = len(tokens)
    char_count = len(c)

    org_keyword_count = sum(1 for t in ltokens if t in ORG_KEYWORDS)
    connector_count = sum(1 for t in ltokens if t in CONNECTOR_WORDS)
    particle_count = sum(1 for t in ltokens if t in NAME_PARTICLES)
    collaboration_count = sum(1 for t in ltokens if t in COLLABORATION_WORDS)

    org_phrase_count = sum(1 for phrase in ORG_PHRASES if phrase in c_lower)

    initial_count = sum(1 for t in tokens if is_initial(t))
    alpha_name_piece_count = sum(1 for t in tokens if is_alpha_name_piece(t))

    uppercase_tokens = sum(1 for t in tokens if len(t) > 1 and t.isupper())
    titlecase_tokens = sum(
        1 for t in tokens
        if len(t) > 1 and t[0].isupper() and any(ch.islower() for ch in t[1:])
    )

    has_digit = int(bool(re.search(r"\d", c)))
    has_email = int(bool(re.search(r"\S+@\S+", c)))
    has_url = int(bool(re.search(r"https?://|www\.", c_lower)))
    has_ampersand = int("&" in c)
    has_slash = int("/" in c)
    has_hyphen = int("-" in c)
    has_apostrophe = int("'" in c or "’" in c)

    all_tokens_name_like = int(
        token_count > 0 and all(is_initial(t) or is_alpha_name_piece(t) for t in tokens)
    )

    plausible_person_length = int(2 <= token_count <= 5)
    too_long_for_person = int(token_count > 6)

    starts_with_the = int(c_lower.startswith("the "))
    contains_of = int(" of " in f" {c_lower} ")

    # Some arXiv records contain "CMS Collaboration", "ATLAS Collaboration", etc.
    looks_like_collaboration = int(
        collaboration_count > 0 or c_lower.endswith("collaboration")
    )

    ner_is_person = int(ner_label == "PERSON")
    ner_is_org = int(ner_label == "ORG")

    return {
        "token_count": token_count,
        "char_count": char_count,
        "log_char_count": math.log1p(char_count),
        "org_keyword_count": org_keyword_count,
        "connector_count": connector_count,
        "particle_count": particle_count,
        "collaboration_count": collaboration_count,
        "org_phrase_count": org_phrase_count,
        "initial_count": initial_count,
        "alpha_name_piece_count": alpha_name_piece_count,
        "uppercase_tokens": uppercase_tokens,
        "titlecase_tokens": titlecase_tokens,
        "has_digit": has_digit,
        "has_email": has_email,
        "has_url": has_url,
        "has_ampersand": has_ampersand,
        "has_slash": has_slash,
        "has_hyphen": has_hyphen,
        "has_apostrophe": has_apostrophe,
        "all_tokens_name_like": all_tokens_name_like,
        "plausible_person_length": plausible_person_length,
        "too_long_for_person": too_long_for_person,
        "starts_with_the": starts_with_the,
        "contains_of": contains_of,
        "looks_like_collaboration": looks_like_collaboration,
        "ner_is_person": ner_is_person,
        "ner_is_org": ner_is_org,
    }


# ---------------------------------------------------------------------
# NER helper
# ---------------------------------------------------------------------

def get_ner_label(candidate: str, nlp=None) -> Optional[str]:
    """
    Returns a coarse spaCy NER label if available.

    For very short name candidates, NER may return nothing.
    That is fine; the rule scorer and supervised classifier still work.
    """
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
# Rule-based weak scorer
# ---------------------------------------------------------------------

@dataclass
class RuleDecision:
    label: str
    score: float
    reason: str


def rule_score_candidate(candidate: str, ner_label: Optional[str] = None) -> RuleDecision:
    """
    Produces a weak person/non-person/uncertain decision.

    label:
        PERSON
        NON_PERSON
        UNCERTAIN

    score:
        Approximate confidence-like score from 0 to 1.
    """
    c = normalize_spaces(candidate)
    c_lower = c.lower()
    tokens = tokenize_candidate(c)
    ltokens = [t.lower() for t in tokens]
    f = extract_features(c, ner_label=ner_label)

    if not c:
        return RuleDecision("NON_PERSON", 1.0, "empty")

    if f["has_digit"] or f["has_email"] or f["has_url"]:
        return RuleDecision("NON_PERSON", 0.99, "contains digit/email/url")

    if f["looks_like_collaboration"]:
        return RuleDecision("NON_PERSON", 0.98, "collaboration/group-like")

    if f["org_phrase_count"] > 0:
        return RuleDecision("NON_PERSON", 0.97, "contains organization phrase")

    if f["org_keyword_count"] >= 2:
        return RuleDecision("NON_PERSON", 0.96, "multiple organization keywords")

    if f["org_keyword_count"] >= 1 and f["contains_of"]:
        return RuleDecision("NON_PERSON", 0.95, "organization keyword plus 'of'")

    if f["too_long_for_person"]:
        return RuleDecision("NON_PERSON", 0.93, "too many tokens")

    if f["starts_with_the"]:
        return RuleDecision("NON_PERSON", 0.92, "starts with 'the'")

    if f["uppercase_tokens"] >= 1 and len(c) <= 8:
        return RuleDecision("NON_PERSON", 0.88, "short acronym-like candidate")

    if ner_label == "ORG":
        return RuleDecision("NON_PERSON", 0.80, "NER says ORG")

    # Strong positive signals.
    if (
        f["plausible_person_length"]
        and f["all_tokens_name_like"]
        and f["org_keyword_count"] == 0
        and has_mixed_case_or_initials(c)
    ):
        # Example: "J. Smith", "John Smith", "Jean-Pierre Dupont"
        return RuleDecision("PERSON", 0.90, "name-like structure")

    if ner_label == "PERSON" and f["plausible_person_length"] and f["org_keyword_count"] == 0:
        return RuleDecision("PERSON", 0.88, "NER says PERSON and structure plausible")

    # Single-token authors are dangerous because they may be surnames,
    # but also organizations/acronyms. Keep them uncertain.
    if len(tokens) == 1:
        return RuleDecision("UNCERTAIN", 0.50, "single-token candidate")

    # Particles can appear in real names, but they also appear in institutions.
    if f["connector_count"] > 0 and f["particle_count"] == 0:
        return RuleDecision("UNCERTAIN", 0.45, "connector words present")

    return RuleDecision("UNCERTAIN", 0.50, "ambiguous")


# ---------------------------------------------------------------------
# Candidate collection
# ---------------------------------------------------------------------

def iter_arxiv_records(file_path: str, max_records: Optional[int] = None) -> Iterable[dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_records is not None and i >= max_records:
                break

            line = line.strip()
            if not line:
                continue

            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def collect_candidates(
    file_path: str,
    max_records: Optional[int] = None,
    nlp=None,
) -> pd.DataFrame:
    """
    Reads arXiv records and returns one row per candidate occurrence.
    """
    rows = []

    for record in tqdm(iter_arxiv_records(file_path, max_records=max_records), desc="Reading records"):
        arxiv_id = record.get("id", "")
        authors_raw = record.get("authors", "")

        if not authors_raw:
            continue

        candidates = split_author_candidates(authors_raw)

        for position, candidate in enumerate(candidates):
            ner_label = get_ner_label(candidate, nlp=nlp)
            decision = rule_score_candidate(candidate, ner_label=ner_label)

            rows.append({
                "arxiv_id": arxiv_id,
                "candidate": candidate,
                "position": position,
                "ner_label": ner_label,
                "rule_label": decision.label,
                "rule_score": decision.score,
                "rule_reason": decision.reason,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Supervised model
# ---------------------------------------------------------------------

class FeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Converts candidate strings into feature dictionaries.
    Used inside sklearn pipeline.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []

        for item in X:
            if isinstance(item, dict):
                candidate = item.get("candidate", "")
                ner_label = item.get("ner_label", None)
            else:
                candidate = str(item)
                ner_label = None

            rows.append(extract_features(candidate, ner_label=ner_label))

        return rows


def get_candidate_texts(X):
    """
    Extracts raw candidate text for character n-gram TF-IDF.
    """
    texts = []

    for item in X:
        if isinstance(item, dict):
            texts.append(item.get("candidate", ""))
        else:
            texts.append(str(item))

    return texts


def build_supervised_model() -> Pipeline:
    """
    Hybrid model:
        - handcrafted structural features
        - character n-gram features
        - logistic regression classifier

    Character n-grams are very useful here because they learn patterns like:
        university
        institute
        lab
        collaboration
        initials
        name-like endings
    """
    feature_union = FeatureUnion([
        (
            "manual_features",
            Pipeline([
                ("extract", FeatureExtractor()),
                ("vectorize", DictVectorizer()),
            ])
        ),
        (
            "char_ngrams",
            Pipeline([
                ("text", FunctionTransformer(get_candidate_texts, validate=False)),
                ("tfidf", TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(2, 5),
                    min_df=2,
                    max_features=100_000,
                )),
            ])
        ),
    ])

    model = Pipeline([
        ("features", feature_union),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            n_jobs=-1,
        )),
    ])

    return model


def train_model(label_file: str, model_out: str):
    """
    label_file must be a CSV with columns:
        candidate,label

    label should be:
        1 for person
        0 for non-person
    """
    df = pd.read_csv(label_file)

    required = {"candidate", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Label file is missing columns: {missing}")

    df = df.dropna(subset=["candidate", "label"]).copy()
    df["candidate"] = df["candidate"].astype(str)
    df["label"] = df["label"].astype(int)

    X = [{"candidate": c, "ner_label": None} for c in df["candidate"]]
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = build_supervised_model()
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    print("\nValidation report:")
    print(classification_report(y_test, preds, target_names=["NON_PERSON", "PERSON"]))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, preds))

    joblib.dump(model, model_out)
    print(f"\nSaved model to: {model_out}")


def predict_with_model(model, candidate: str, ner_label: Optional[str] = None) -> float:
    """
    Returns probability that candidate is a person.
    """
    X = [{"candidate": candidate, "ner_label": ner_label}]
    proba = model.predict_proba(X)[0]
    person_index = list(model.classes_).index(1)
    return float(proba[person_index])


# ---------------------------------------------------------------------
# Cleaning / counting logic
# ---------------------------------------------------------------------

def classify_candidate(
    candidate: str,
    ner_label: Optional[str] = None,
    model=None,
    person_threshold: float = 0.90,
    nonperson_threshold: float = 0.10,
) -> Tuple[str, float, str]:
    """
    Final classifier.

    If supervised model is available:
        uses model probability.

    If no model is available:
        uses rule decision.
    """
    if model is not None:
        p_person = predict_with_model(model, candidate, ner_label=ner_label)

        if p_person >= person_threshold:
            return "PERSON", p_person, "supervised_model"
        elif p_person <= nonperson_threshold:
            return "NON_PERSON", p_person, "supervised_model"
        else:
            return "UNCERTAIN", p_person, "supervised_model"

    decision = rule_score_candidate(candidate, ner_label=ner_label)

    if decision.label == "PERSON":
        return "PERSON", decision.score, decision.reason
    elif decision.label == "NON_PERSON":
        return "NON_PERSON", 1.0 - decision.score, decision.reason
    else:
        return "UNCERTAIN", 0.5, decision.reason


def clean_and_count_authors(
    file_path: str,
    max_records: Optional[int] = None,
    use_spacy: bool = False,
    model_path: Optional[str] = None,
    person_threshold: float = 0.90,
    nonperson_threshold: float = 0.10,
    export_uncertain: Optional[str] = None,
    output_counts: Optional[str] = None,
) -> Counter:
    nlp = load_spacy_model(use_spacy)

    model = None
    if model_path:
        model = joblib.load(model_path)
        print(f"Loaded supervised model from: {model_path}")

    author_counter = Counter()
    decision_counter = Counter()
    uncertain_rows = []

    records = iter_arxiv_records(file_path, max_records=max_records)

    for record in tqdm(records, desc="Cleaning authors"):
        arxiv_id = record.get("id", "")
        authors_raw = record.get("authors", "")

        if not authors_raw:
            continue

        candidates = split_author_candidates(authors_raw)

        for position, candidate in enumerate(candidates):
            ner_label = get_ner_label(candidate, nlp=nlp)
            label, score, reason = classify_candidate(
                candidate,
                ner_label=ner_label,
                model=model,
                person_threshold=person_threshold,
                nonperson_threshold=nonperson_threshold,
            )

            decision_counter[label] += 1

            if label == "PERSON":
                author_counter[candidate] += 1
            elif label == "UNCERTAIN":
                uncertain_rows.append({
                    "arxiv_id": arxiv_id,
                    "candidate": candidate,
                    "position": position,
                    "ner_label": ner_label,
                    "score": score,
                    "reason": reason,
                })

    print("\nDecision counts:")
    for k, v in decision_counter.most_common():
        print(f"{k}: {v}")

    print(f"\nUnique accepted person authors: {len(author_counter)}")

    if export_uncertain:
        pd.DataFrame(uncertain_rows).to_csv(export_uncertain, index=False)
        print(f"Exported uncertain candidates to: {export_uncertain}")

    if output_counts:
        count_df = pd.DataFrame(
            author_counter.most_common(),
            columns=["author", "count"]
        )
        count_df.to_csv(output_counts, index=False)
        print(f"Saved author counts to: {output_counts}")

    return author_counter


# ---------------------------------------------------------------------
# Labelling file generation
# ---------------------------------------------------------------------

def export_labelling_candidates(
    file_path: str,
    output_file: str,
    max_records: Optional[int] = 50_000,
    use_spacy: bool = False,
    max_candidates: int = 5_000,
):
    """
    Exports a deduplicated candidate list for manual labelling.

    The output CSV has:
        candidate,frequency,rule_label,rule_score,rule_reason,label

    You should manually fill:
        label = 1 for person
        label = 0 for non-person

    Prioritizes frequent and uncertain candidates.
    """
    nlp = load_spacy_model(use_spacy)

    df = collect_candidates(
        file_path=file_path,
        max_records=max_records,
        nlp=nlp,
    )

    if df.empty:
        print("No candidates found.")
        return

    grouped = (
        df.groupby("candidate")
        .agg(
            frequency=("candidate", "size"),
            rule_label=("rule_label", lambda x: x.mode().iloc[0]),
            rule_score=("rule_score", "mean"),
            rule_reason=("rule_reason", lambda x: x.mode().iloc[0]),
        )
        .reset_index()
    )

    # Prioritize:
    #   1. frequent candidates
    #   2. uncertain candidates
    grouped["priority"] = grouped["frequency"]

    grouped.loc[grouped["rule_label"] == "UNCERTAIN", "priority"] *= 3
    grouped = grouped.sort_values("priority", ascending=False)

    grouped = grouped.head(max_candidates).copy()
    grouped["label"] = ""

    grouped = grouped[
        ["candidate", "frequency", "rule_label", "rule_score", "rule_reason", "label"]
    ]

    grouped.to_csv(output_file, index=False)
    print(f"Exported labelling file to: {output_file}")
    print("\nFill the 'label' column manually:")
    print("    1 = person")
    print("    0 = non-person")


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

def plot_top_authors(author_counter: Counter, top_n: int = 30):
    top = author_counter.most_common(top_n)

    if not top:
        print("No authors to plot.")
        return

    labels = [a for a, _ in top]
    values = [c for _, c in top]

    plt.figure(figsize=(16, 8))
    plt.bar(labels, values)
    plt.xticks(rotation=75, ha="right")
    plt.xlabel("Author")
    plt.ylabel("Count")
    plt.title(f"Top {top_n} accepted person authors")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Hybrid author-name cleaner for arXiv metadata."
    )

    parser.add_argument(
        "--file",
        required=True,
        help="Path to arxiv-metadata-oai-snapshot.json",
    )

    parser.add_argument(
        "--mode",
        required=True,
        choices=["trial", "export-labels", "train", "full"],
        help=(
            "trial: run on a small subset; "
            "export-labels: create CSV for manual labelling; "
            "train: train supervised model from labelled CSV; "
            "full: run on full or large dataset"
        ),
    )

    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Maximum number of arXiv records to process.",
    )

    parser.add_argument(
        "--use-spacy",
        action="store_true",
        help="Use spaCy NER if installed.",
    )

    parser.add_argument(
        "--label-file",
        default=None,
        help="CSV file with columns candidate,label for supervised training.",
    )

    parser.add_argument(
        "--model-in",
        default=None,
        help="Path to trained model for prediction.",
    )

    parser.add_argument(
        "--model-out",
        default="author_person_classifier.joblib",
        help="Path to save trained model.",
    )

    parser.add_argument(
        "--export-uncertain",
        default=None,
        help="CSV path to export uncertain candidates.",
    )

    parser.add_argument(
        "--output-counts",
        default=None,
        help="CSV path to save accepted author counts.",
    )

    parser.add_argument(
        "--label-output",
        default="author_labelling_candidates.csv",
        help="CSV path for exported labelling candidates.",
    )

    parser.add_argument(
        "--max-label-candidates",
        type=int,
        default=5000,
        help="Maximum number of candidates to export for labelling.",
    )

    parser.add_argument(
        "--person-threshold",
        type=float,
        default=0.90,
        help="Supervised probability threshold for accepting PERSON.",
    )

    parser.add_argument(
        "--nonperson-threshold",
        type=float,
        default=0.10,
        help="Supervised probability threshold for rejecting NON_PERSON.",
    )

    args = parser.parse_args()

    if args.mode == "trial":
        max_records = args.max_records if args.max_records is not None else 10_000

        author_counter = clean_and_count_authors(
            file_path=args.file,
            max_records=max_records,
            use_spacy=args.use_spacy,
            model_path=args.model_in,
            person_threshold=args.person_threshold,
            nonperson_threshold=args.nonperson_threshold,
            export_uncertain=args.export_uncertain or "uncertain_trial.csv",
            output_counts=args.output_counts or "author_counts_trial.csv",
        )

        print("\nTop 30 accepted authors:")
        for author, count in author_counter.most_common(30):
            print(f"{author}: {count}")

        plot_top_authors(author_counter, top_n=30)

    elif args.mode == "export-labels":
        max_records = args.max_records if args.max_records is not None else 50_000

        export_labelling_candidates(
            file_path=args.file,
            output_file=args.label_output,
            max_records=max_records,
            use_spacy=args.use_spacy,
            max_candidates=args.max_label_candidates,
        )

    elif args.mode == "train":
        if not args.label_file:
            raise ValueError("--label-file is required in train mode.")

        train_model(
            label_file=args.label_file,
            model_out=args.model_out,
        )

    elif args.mode == "full":
        author_counter = clean_and_count_authors(
            file_path=args.file,
            max_records=args.max_records,
            use_spacy=args.use_spacy,
            model_path=args.model_in,
            person_threshold=args.person_threshold,
            nonperson_threshold=args.nonperson_threshold,
            export_uncertain=args.export_uncertain or "uncertain_full.csv",
            output_counts=args.output_counts or "author_counts_full.csv",
        )

        print("\nTop 30 accepted authors:")
        for author, count in author_counter.most_common(30):
            print(f"{author}: {count}")

        plot_top_authors(author_counter, top_n=30)


if __name__ == "__main__":
    main()