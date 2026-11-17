"""
Benchmark all four models on the SAME held-out test set (data/processed/test.jsonl).

Reports per model:
  * Accuracy (%), Out-of-Scope recall (%) [argmax], OOS recall incl. the
    low-confidence -> Human Agent fallback, in-scope accuracy, macro-F1,
    accuracy on the hand-written domain test examples, EN / TR accuracy
  * Model size on disk (MB)
  * CPU latency per single message (ms): mean / p50 / p95 over >= 100 runs
    (default 200) after warm-up, with the hardware logged.
  * Failure analysis: 5 concrete misclassified test examples.

Outputs: results/results.md, results/results.csv, results/predictions.csv,
         results/hardware.json

Run:  python -m src.evaluate [--runs 200] [--threads N] [--no_ablation]
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import Counter

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score

from src import config as C
from src.inference import load_routers
from src.nlp_utils import preprocess
from src.utils import get_hardware_info, read_jsonl, set_seed


def md_table(headers: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def measure_latency(router, samples: list[str], runs: int, warmup: int) -> np.ndarray:
    """Single-message CPU latency in ms (tokenisation/vectorisation + forward + softmax)."""
    for i in range(warmup):
        router.predict_proba([samples[i % len(samples)]])
    times = np.empty(runs)
    for i in range(runs):
        t0 = time.perf_counter()
        router.predict_proba([samples[i % len(samples)]])
        times[i] = (time.perf_counter() - t0) * 1000.0
    return times


def pick_failures(test, inputs, preds, confs, primary, routers, k=5):
    """Pick k diverse, informative errors of the primary (production) model."""
    y = [r["label"] for r in test]
    errs = [i for i in range(len(test)) if preds[primary][i] != y[i]]

    def category(i):
        if y[i] == C.OOS_LABEL:
            return "out-of-scope accepted as a team (false accept)"
        if preds[primary][i] == C.OOS_LABEL:
            return "in-scope rejected as out-of-scope (false reject)"
        return "wrong team"

    # Priority: hand-written domain examples first, then confident mistakes.
    errs.sort(key=lambda i: (test[i]["source"] != "domain", -confs[primary][i]))
    chosen, langs, cats = [], Counter(), Counter()
    for i in errs:
        if langs[test[i]["lang"]] >= 3 or cats[category(i)] >= 2:
            continue
        chosen.append(i)
        langs[test[i]["lang"]] += 1
        cats[category(i)] += 1
        if len(chosen) == k:
            break
    for i in errs:                               # fill up if diversity limits hit
        if len(chosen) == k:
            break
        if i not in chosen:
            chosen.append(i)

    blocks = []
    for n, i in enumerate(chosen, 1):
        others = ", ".join(f"{routers[m].name.split()[0]}={preds[m][i]} ({confs[m][i]:.2f})"
                           for m in preds)
        blocks.append(
            f"**{n}. [{test[i]['lang'].upper()}, {test[i]['source']}] {category(i)}**\n\n"
            f"- Message: `{test[i]['text']}`\n"
            f"- Model input after preprocessing: `{inputs[i]}`\n"
            f"- True label: `{y[i]}` (MASSIVE intent: `{test[i]['orig_intent']}`)\n"
            f"- Predicted by {routers[primary].name}: `{preds[primary][i]}` "
            f"(confidence {confs[primary][i]:.2f})\n"
            f"- All models: {others}\n")
    return blocks, len(errs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=C.LATENCY_RUNS)
    ap.add_argument("--warmup", type=int, default=C.LATENCY_WARMUP)
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    ap.add_argument("--threshold", type=float, default=C.CONFIDENCE_THRESHOLD)
    ap.add_argument("--no_ablation", action="store_true",
                    help="Do not report the FP32 distilled student row")
    args = ap.parse_args()
    if args.runs < 100:
        raise SystemExit("--runs must be >= 100 (assignment requirement)")

    set_seed(C.SEED)
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    C.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    test = read_jsonl(C.TEST_FILE)
    print(f"[eval] test set: {len(test)} messages "
          f"({dict(Counter(r['label'] for r in test))})")
    t0 = time.perf_counter()
    inputs = [preprocess(r["text"]).model_input for r in test]
    prep_ms = (time.perf_counter() - t0) * 1000 / len(test)
    y = np.array([r["label"] for r in test])
    is_oos = y == C.OOS_LABEL
    is_domain = np.array([r["source"] == "domain" for r in test])
    lang = np.array([r["lang"] for r in test])

    routers = load_routers(include_ablation=not args.no_ablation)
    if not routers:
        raise SystemExit("No trained models found in models/.")
    hw = get_hardware_info()
    print(f"[eval] hardware: {hw}")

    rng = random.Random(C.SEED)
    lat_samples = [inputs[i] for i in rng.sample(range(len(inputs)), min(100, len(inputs)))]

    rows, preds, confs = [], {}, {}
    for key, router in routers.items():
        print(f"[eval] {router.name}: predicting test set ...")
        probs = np.concatenate([router.predict_proba(inputs[i:i + 64])
                                for i in range(0, len(inputs), 64)])
        p = np.array([C.LABELS[j] for j in probs.argmax(1)])
        conf = probs.max(1)
        preds[key], confs[key] = p, conf
        correct = p == y
        human = (p == C.OOS_LABEL) | (conf < args.threshold)
        print(f"[eval] {router.name}: measuring latency ({args.runs} runs) ...")
        lat = measure_latency(router, lat_samples, args.runs, args.warmup)
        rows.append({
            "model": router.name, "area": router.area, "precision": router.precision,
            "accuracy": 100 * correct.mean(),
            "oos_recall": 100 * (p[is_oos] == C.OOS_LABEL).mean() if is_oos.any() else float("nan"),
            "oos_recall_with_fallback": 100 * human[is_oos].mean() if is_oos.any() else float("nan"),
            "in_scope_accuracy": 100 * correct[~is_oos].mean(),
            "macro_f1": 100 * f1_score(y, p, labels=C.LABELS, average="macro", zero_division=0),
            "domain_accuracy": 100 * correct[is_domain].mean() if is_domain.any() else float("nan"),
            "en_accuracy": 100 * correct[lang == "en"].mean(),
            "tr_accuracy": 100 * correct[lang == "tr"].mean(),
            "size_mb": router.size_mb,
            "latency_mean_ms": lat.mean(), "latency_p50_ms": float(np.median(lat)),
            "latency_p95_ms": float(np.percentile(lat, 95)), "latency_runs": args.runs,
        })
        r = rows[-1]
        print(f"[eval]   acc {r['accuracy']:.2f}%  OOS recall {r['oos_recall']:.2f}%  "
              f"size {r['size_mb']:.1f} MB  latency {r['latency_mean_ms']:.2f} ms")

    # ---------------- CSV outputs ----------------
    with open(C.RESULTS_DIR / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()})
    with open(C.RESULTS_DIR / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "lang", "source", "text", "model_input", "label"] +
                   [f"{k}_{c}" for k in preds for c in ("pred", "conf")])
        for i, r in enumerate(test):
            w.writerow([r["id"], r["lang"], r["source"], r["text"], inputs[i], r["label"]] +
                       [v for k in preds for v in (preds[k][i], f"{confs[k][i]:.4f}")])
    (C.RESULTS_DIR / "hardware.json").write_text(json.dumps(hw, indent=2))

    # ---------------- Markdown report ----------------
    main_tbl = md_table(
        ["Model", "Area", "Precision", "Accuracy (%)", "OOS Recall (%)", "Size (MB)",
         "Latency mean (ms)", "Latency p95 (ms)"],
        [[r["model"], r["area"], r["precision"], f"{r['accuracy']:.2f}", f"{r['oos_recall']:.2f}",
          f"{r['size_mb']:.1f}", f"{r['latency_mean_ms']:.2f}", f"{r['latency_p95_ms']:.2f}"]
         for r in rows])
    ext_tbl = md_table(
        ["Model", "OOS Recall + fallback (%)", "In-scope Acc (%)", "Macro-F1 (%)",
         "Domain Acc (%)", "EN Acc (%)", "TR Acc (%)"],
        [[r["model"], f"{r['oos_recall_with_fallback']:.2f}", f"{r['in_scope_accuracy']:.2f}",
          f"{r['macro_f1']:.2f}", f"{r['domain_accuracy']:.2f}", f"{r['en_accuracy']:.2f}",
          f"{r['tr_accuracy']:.2f}"] for r in rows])

    primary = "m4" if "m4" in preds else list(preds)[-1]
    cm = confusion_matrix(y, preds[primary], labels=C.LABELS)
    cm_tbl = md_table(["true \\ pred"] + C.LABELS,
                      [[C.LABELS[i]] + list(cm[i]) for i in range(len(C.LABELS))])
    failures, n_err = pick_failures(test, inputs, preds, confs, primary, routers)

    hw_lines = "\n".join(f"- **{k}**: {v}" for k, v in hw.items())
    report = f"""# Benchmark results - Customer Support Ticket Router

All models evaluated on the same held-out test set: **{len(test)} messages**
({int((lang == 'en').sum())} EN / {int((lang == 'tr').sum())} TR, {int(is_oos.sum())} out-of-scope,
{int(is_domain.sum())} hand-written domain examples).

## Main results table (R4)

{main_tbl}

*Latency = average over **{args.runs} single-message runs** on CPU after {args.warmup} warm-up runs
(tokenisation/vectorisation + forward pass + softmax). Shared preprocessing
(normalisation + spell correction) adds **{prep_ms:.3f} ms** per message for every model.
Size = total bytes of the saved model directory (weights + tokenizer/vocab + config).*

## Additional metrics

{ext_tbl}

*OOS Recall = share of out-of-scope test messages predicted as `out_of_scope`.
"OOS Recall + fallback" also counts messages sent to the Human Agent because the
top probability was below the threshold ({args.threshold:.2f}).*

## Hardware

{hw_lines}

## Confusion matrix - {routers[primary].name}

{cm_tbl}

## Failure analysis - {routers[primary].name} ({n_err} errors of {len(test)})

{chr(10).join(failures)}
"""
    (C.RESULTS_DIR / "results.md").write_text(report, encoding="utf-8")
    print("\n" + main_tbl + "\n\n" + ext_tbl)
    print(f"\nHardware: {hw}")
    print(f"\nFailure analysis ({routers[primary].name}):\n")
    print("\n".join(failures))
    print(f"[eval] wrote {C.RESULTS_DIR / 'results.md'} and results.csv / predictions.csv")


if __name__ == "__main__":
    main()
