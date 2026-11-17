"""
Model 1 (Area B) - TF-IDF + Logistic Regression baseline.

Features: word 1-2 grams + character 2-5 grams (char_wb). Character n-grams
make the classical model robust to Turkish suffixes and typos.

Run:  python -m src.train_baseline
"""
from __future__ import annotations

import json
import time

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import FeatureUnion, Pipeline

from src import config as C
from src.nlp_utils import preprocess_batch
from src.utils import read_jsonl, set_seed


def build_pipeline() -> Pipeline:
    features = FeatureUnion([
        ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1,
                                 sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                                 sublinear_tf=True, max_features=200_000)),
    ])
    clf = LogisticRegression(C=C.BASELINE_C, max_iter=3000, class_weight="balanced",
                             random_state=C.SEED)
    return Pipeline([("tfidf", features), ("logreg", clf)])


def main() -> None:
    set_seed(C.SEED)
    train, val = read_jsonl(C.TRAIN_FILE), read_jsonl(C.VAL_FILE)
    x_train = preprocess_batch([r["text"] for r in train])
    x_val = preprocess_batch([r["text"] for r in val])
    y_train, y_val = [r["label"] for r in train], [r["label"] for r in val]

    pipe = build_pipeline()
    t0 = time.time()
    pipe.fit(x_train, y_train)
    print(f"[baseline] trained on {len(x_train)} messages in {time.time() - t0:.1f}s")

    pred = pipe.predict(x_val)
    acc = accuracy_score(y_val, pred)
    print(f"[baseline] validation accuracy: {acc * 100:.2f}%")
    print(classification_report(y_val, pred, labels=C.LABELS, digits=3, zero_division=0))

    C.BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, C.BASELINE_DIR / "tfidf_logreg.joblib", compress=3)
    (C.BASELINE_DIR / "train_info.json").write_text(json.dumps(
        {"val_accuracy": acc, "n_train": len(x_train), "C": C.BASELINE_C}, indent=2))
    print(f"[baseline] saved to {C.BASELINE_DIR}")


if __name__ == "__main__":
    main()
