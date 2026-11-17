"""
Model 3 (Area C) - Fine-tune a multilingual Transformer encoder
(default: bert-base-multilingual-cased, WordPiece tokenizer) with a
sequence-classification head. This model is also the TEACHER for Model 4.

Run:  python -m src.train_transformer [--epochs 3] [--device auto|cpu|cuda]
"""
from __future__ import annotations

import argparse
import json

import torch.nn as nn

from src import config as C
from src.hf_utils import (accuracy_from_logits, class_weights, load_hf_classifier,
                          make_loader, predict_logits, train_loop)
from src.nlp_utils import preprocess_batch
from src.utils import read_jsonl, resolve_device, set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default=C.TEACHER_MODEL_NAME)
    ap.add_argument("--epochs", type=int, default=C.TEACHER_EPOCHS)
    ap.add_argument("--lr", type=float, default=C.TEACHER_LR)
    ap.add_argument("--batch_size", type=int, default=C.TEACHER_BATCH)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    set_seed(C.SEED)
    device = resolve_device(args.device)
    print(f"[teacher] fine-tuning {args.model_name} on {device}")

    train, val = read_jsonl(C.TRAIN_FILE), read_jsonl(C.VAL_FILE)
    x_train = preprocess_batch([r["text"] for r in train])
    x_val = preprocess_batch([r["text"] for r in val])
    y_train = [C.LABEL2ID[r["label"]] for r in train]
    y_val = [C.LABEL2ID[r["label"]] for r in val]

    tokenizer, model = load_hf_classifier(args.model_name)
    print(f"[teacher] example WordPiece tokens: {tokenizer.tokenize(x_train[0])[:20]}")
    loader = make_loader(tokenizer, x_train, y_train, batch_size=args.batch_size, shuffle=True)
    ce = nn.CrossEntropyLoss(weight=class_weights(y_train, len(C.LABELS)).to(device))

    def loss_fn(logits, batch):
        return ce(logits, batch[2])

    def eval_fn():
        return accuracy_from_logits(predict_logits(model, tokenizer, x_val, device), y_val)

    def save_fn():
        C.TEACHER_DIR.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(C.TEACHER_DIR, safe_serialization=True)
        tokenizer.save_pretrained(C.TEACHER_DIR)

    best = train_loop(model, loader, loss_fn, args.epochs, args.lr, device,
                      eval_fn, save_fn, tag="teacher")
    (C.TEACHER_DIR / "train_info.json").write_text(json.dumps(
        {"base_model": args.model_name, "val_accuracy": best, "epochs": args.epochs,
         "lr": args.lr, "batch_size": args.batch_size, "max_len": C.MAX_LEN}, indent=2))
    print(f"[teacher] best validation accuracy {best * 100:.2f}% -> {C.TEACHER_DIR}")


if __name__ == "__main__":
    main()
