"""
Step 1 - Data preparation.

  1. Downloads Amazon MASSIVE v1.1 (en-US + tr-TR) from the official S3 bucket.
  2. Maps MASSIVE intents to support teams (src/config.py) and everything else
     to `out_of_scope`. Uses MASSIVE's OFFICIAL train/dev/test partitions, so
     the held-out test set is fixed; down-sampling uses a fixed seed.
  3. Splits the 80 hand-written appliance examples (data/synthetic/)
     stratified by (label, language) into train/val/test.
  4. Augments ONLY the training part of the domain examples (new model codes,
     typos, missing Turkish characters, greetings, suffixes) - the val/test
     domain examples stay untouched, so there is no leakage.
  5. Saves data/processed/{train,val,test}.jsonl + dataset_stats.json.

Run:  python -m src.data_prep            (add --massive_tar PATH for offline use)
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import tarfile
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

from src import config as C
from src.nlp_utils import MODEL_CODE_RE, fold_turkish
from src.utils import read_jsonl, set_seed, write_jsonl


# --------------------------------------------------------------------------- #
# MASSIVE download + loading
# --------------------------------------------------------------------------- #
def download_massive(dest: Path) -> Path:
    if dest.exists():  # .part file is renamed only after a complete download
        print(f"[data] Using cached archive {dest}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[data] Downloading MASSIVE v1.1 from {C.MASSIVE_URL}")

    def hook(blocks, block_size, total):
        if total > 0:
            pct = min(100, blocks * block_size * 100 // total)
            sys.stdout.write(f"\r[data]   {pct:3d}%")
            sys.stdout.flush()

    tmp = dest.with_suffix(".part")
    urllib.request.urlretrieve(C.MASSIVE_URL, tmp, reporthook=hook)
    tmp.rename(dest)
    print()
    return dest


def load_massive(tar_path: Path, locales: list[str]) -> list[dict]:
    """Read <locale>.jsonl straight from the tar archive (nothing extracted)."""
    rows = []
    with tarfile.open(tar_path, "r:gz") as tar:
        members = {m.name: m for m in tar.getmembers() if m.isfile()}
        for loc in locales:
            name = next((n for n in members if n.endswith(f"/{loc}.jsonl")), None)
            if name is None:
                raise FileNotFoundError(f"{loc}.jsonl not found in {tar_path}")
            with tar.extractfile(members[name]) as f:
                for line in f:
                    r = json.loads(line)
                    rows.append({
                        "id": f"massive-{loc}-{r['id']}",
                        "text": r["utt"],
                        "label": C.MASSIVE_INTENT_TO_LABEL.get(r["intent"], C.OOS_LABEL),
                        "lang": loc.split("-")[0],
                        "source": "massive",
                        "orig_intent": r["intent"],
                        "partition": r["partition"],      # train / dev / test
                    })
    print(f"[data] Loaded {len(rows)} MASSIVE utterances for {locales}")
    return rows


def cap_massive(rows: list[dict], seed: int) -> dict[str, list[dict]]:
    """Deterministically down-sample per (partition, lang, label)."""
    rng = random.Random(seed)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["partition"], r["lang"], r["label"])].append(r)
    out = {"train": [], "dev": [], "test": []}
    for (part, _lang, label), items in sorted(groups.items()):
        items = sorted(items, key=lambda x: x["id"])
        cap = C.CAPS[part].get(label)
        if cap is not None and len(items) > cap:
            items = rng.sample(items, cap)
        out[part].extend(items)
    return out


# --------------------------------------------------------------------------- #
# Hand-written domain examples: split + augmentation
# --------------------------------------------------------------------------- #
def split_synthetic(rows: list[dict], seed: int) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["label"], r["lang"])].append(r)
    out = {"train": [], "val": [], "test": []}
    for key in sorted(groups):
        items = groups[key][:]
        rng.shuffle(items)
        n_test = max(1, round(len(items) * C.SYNTH_TEST_FRACTION))
        n_val = max(1, round(len(items) * C.SYNTH_VAL_FRACTION))
        out["test"] += items[:n_test]
        out["val"] += items[n_test:n_test + n_val]
        out["train"] += items[n_test + n_val:]
    for split, items in out.items():
        for r in items:
            r["source"] = "domain"
            r["orig_intent"] = "handwritten"
    return out


_CODE_PREFIXES = ["DW", "WM", "RF", "OV", "DR", "AC", "MW", "VC", "TD", "BK"]
_PREFIX = {"en": ["hi,", "hello,", "hey,", "good morning,"],
           "tr": ["merhaba,", "selam,", "iyi günler,", "günaydın,"]}
_SUFFIX = {"en": ["please help", "thanks", "asap", "what should I do?"],
           "tr": ["lütfen yardım edin", "teşekkürler", "acil", "ne yapmalıyım?"]}


def _random_code(rng: random.Random) -> str:
    code = f"{rng.choice(_CODE_PREFIXES)}-{rng.randint(100, 9999)}"
    return code + (rng.choice(["", "", "X", "S", "W"]))


def _typo(word: str, rng: random.Random) -> str:
    if len(word) < 5:
        return word
    i = rng.randint(1, len(word) - 2)
    op = rng.choice(["delete", "swap", "double", "replace"])
    if op == "delete":
        return word[:i] + word[i + 1:]
    if op == "swap":
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    if op == "double":
        return word[:i] + word[i] + word[i:]
    return word[:i] + rng.choice("aeiou") + word[i + 1:]


def augment(example: dict, n: int, rng: random.Random) -> list[dict]:
    """Create up to n label-preserving variants of one training example."""
    text, lang = example["text"], example["lang"]
    variants, seen, tries = [], {text}, 0
    while len(variants) < n and tries < n * 10:
        tries += 1
        t = text
        if MODEL_CODE_RE.search(t):                      # new model number
            t = MODEL_CODE_RE.sub(lambda _m: _random_code(rng), t)
        elif rng.random() < 0.4:
            t = f"{t} {'model' if lang == 'en' else rng.choice(['model', 'modeli'])} {_random_code(rng)}"
        words = t.split()
        for _ in range(rng.choice([0, 1, 1, 2])):        # typos
            j = rng.randrange(len(words))
            words[j] = _typo(words[j], rng)
        t = " ".join(words)
        if lang == "tr" and rng.random() < 0.5:          # ASCII Turkish
            t = fold_turkish(t)
        if rng.random() < 0.3:
            t = f"{rng.choice(_PREFIX[lang])} {t}"
        if rng.random() < 0.3:
            t = f"{t} {rng.choice(_SUFFIX[lang])}"
        r = rng.random()
        if r < 0.15:
            t = t.lower()
        elif r < 0.25:
            t = t.capitalize()
        if t not in seen:
            seen.add(t)
            variants.append({**example, "id": f"{example['id']}-aug{len(variants) + 1}",
                             "text": t, "source": "domain_aug"})
    return variants


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def summarize(splits: dict[str, list[dict]]) -> dict:
    stats = {}
    for name, rows in splits.items():
        stats[name] = {
            "total": len(rows),
            "by_label": dict(sorted(Counter(r["label"] for r in rows).items())),
            "by_lang": dict(sorted(Counter(r["lang"] for r in rows).items())),
            "by_source": dict(sorted(Counter(r["source"] for r in rows).items())),
        }
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--massive_tar", type=Path, default=None,
                    help="Path to an already downloaded amazon-massive-dataset-1.1.tar.gz")
    ap.add_argument("--seed", type=int, default=C.SEED)
    args = ap.parse_args()
    set_seed(args.seed)

    tar_path = args.massive_tar or download_massive(C.MASSIVE_TAR)
    massive = cap_massive(load_massive(tar_path, C.LOCALES), args.seed)

    synth = split_synthetic(read_jsonl(C.SYNTHETIC_FILE), args.seed)
    rng = random.Random(args.seed)
    synth_aug = [v for ex in synth["train"]
                 for v in augment(ex, C.SYNTH_AUGMENT_PER_EXAMPLE, rng)]

    def clean(rows):
        return [{k: r[k] for k in ("id", "text", "label", "lang", "source", "orig_intent")}
                for r in rows]

    splits = {
        "train": clean(massive["train"] + synth["train"] + synth_aug),
        "val": clean(massive["dev"] + synth["val"]),
        "test": clean(massive["test"] + synth["test"]),
    }
    # Deterministic order (shuffled with fixed seed).
    for name in splits:
        random.Random(args.seed).shuffle(splits[name])

    write_jsonl(splits["train"], C.TRAIN_FILE)
    write_jsonl(splits["val"], C.VAL_FILE)
    write_jsonl(splits["test"], C.TEST_FILE)
    stats = summarize(splits)
    C.STATS_FILE.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n[data] Saved splits to", C.PROCESSED_DIR)
    print(f"{'split':6} {'total':>6}  " + "  ".join(f"{l[:14]:>14}" for l in C.LABELS))
    for name, s in stats.items():
        print(f"{name:6} {s['total']:6d}  " +
              "  ".join(f"{s['by_label'].get(l, 0):14d}" for l in C.LABELS))
    print("[data] by language:", {k: v["by_lang"] for k, v in stats.items()})
    print("[data] by source:  ", {k: v["by_source"] for k, v in stats.items()})


if __name__ == "__main__":
    main()
