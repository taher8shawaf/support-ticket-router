"""
Model 4 (Area D) - Knowledge distillation + dynamic INT8 quantization.

  1. Knowledge distillation (Hinton et al., 2015): the fine-tuned mBERT teacher
     (Model 3) produces soft targets; a smaller student
     (distilbert-base-multilingual-cased, 6 layers instead of 12) is trained on
         L = alpha * CE(student, y) + (1 - alpha) * T^2 * KL(p_teacher^T || p_student^T)
  2. Post-training dynamic INT8 quantization of the best student with
     torch.ao.quantization.quantize_dynamic (Linear -> int8, Embedding -> 8-bit).
  3. Round-trip check: the saved INT8 model is re-loaded exactly the way
     evaluate.py / app.py load it, and its validation accuracy is reported.

Run:  python -m src.distill_quantize [--epochs 4] [--device auto] [--skip_distill]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src import config as C
from src.hf_utils import (accuracy_from_logits, class_weights, load_hf_classifier,
                          load_quantized, make_loader, predict_logits,
                          quantize_dynamic_int8, save_quantized, train_loop)
from src.nlp_utils import preprocess_batch
from src.utils import path_size_mb, read_jsonl, resolve_device, set_seed


def distill(args, x_train, y_train, x_val, y_val, device) -> None:
    if not C.TEACHER_DIR.exists():
        raise SystemExit("Teacher not found - run `python -m src.train_transformer` first.")
    print(f"[distill] computing teacher logits on {len(x_train)} training messages")
    t_tok = AutoTokenizer.from_pretrained(C.TEACHER_DIR)
    teacher = AutoModelForSequenceClassification.from_pretrained(C.TEACHER_DIR).to(device)
    teacher_logits = predict_logits(teacher, t_tok, x_train, device)
    n_t = sum(p.numel() for p in teacher.parameters())
    print(f"[distill] teacher train accuracy: "
          f"{accuracy_from_logits(teacher_logits, y_train) * 100:.2f}%")
    del teacher
    if device.type == "cuda":
        torch.cuda.empty_cache()

    tokenizer, student = load_hf_classifier(args.student_name)
    n_s = sum(p.numel() for p in student.parameters())
    print(f"[distill] teacher params {n_t / 1e6:.1f}M -> student params {n_s / 1e6:.1f}M")

    loader = make_loader(tokenizer, x_train, y_train, soft=teacher_logits,
                         batch_size=args.batch_size, shuffle=True)
    ce = nn.CrossEntropyLoss(weight=class_weights(y_train, len(C.LABELS)).to(device))
    T, alpha = args.temperature, args.alpha

    def loss_fn(logits, batch):
        hard = ce(logits, batch[2])
        soft = F.kl_div(F.log_softmax(logits / T, dim=-1),
                        F.softmax(batch[3] / T, dim=-1),
                        reduction="batchmean") * (T * T)
        return alpha * hard + (1.0 - alpha) * soft

    def eval_fn():
        return accuracy_from_logits(predict_logits(student, tokenizer, x_val, device), y_val)

    def save_fn():
        C.STUDENT_FP32_DIR.mkdir(parents=True, exist_ok=True)
        student.save_pretrained(C.STUDENT_FP32_DIR, safe_serialization=True)
        tokenizer.save_pretrained(C.STUDENT_FP32_DIR)

    best = train_loop(student, loader, loss_fn, args.epochs, args.lr, device,
                      eval_fn, save_fn, tag="student")
    (C.STUDENT_FP32_DIR / "train_info.json").write_text(json.dumps(
        {"base_model": args.student_name, "teacher": str(C.TEACHER_DIR.name),
         "val_accuracy": best, "temperature": T, "alpha": alpha,
         "epochs": args.epochs, "lr": args.lr}, indent=2))
    print(f"[distill] best student validation accuracy {best * 100:.2f}%")


def quantize(x_val, y_val, quantize_embeddings: bool) -> None:
    if not C.STUDENT_FP32_DIR.exists():
        raise SystemExit("Student not found - run distillation first.")
    tokenizer = AutoTokenizer.from_pretrained(C.STUDENT_FP32_DIR)
    fp32 = AutoModelForSequenceClassification.from_pretrained(C.STUDENT_FP32_DIR).eval()
    fp32_logits = predict_logits(fp32, tokenizer, x_val, "cpu")

    try:
        qmodel = quantize_dynamic_int8(fp32, quantize_embeddings)
        predict_logits(qmodel, tokenizer, x_val[:4], "cpu")    # smoke test
    except Exception as exc:  # noqa: BLE001 - fall back to Linear-only
        if not quantize_embeddings:
            raise
        print(f"[quant] embedding quantization failed ({exc}); using Linear-only INT8")
        quantize_embeddings = False
        fp32 = AutoModelForSequenceClassification.from_pretrained(C.STUDENT_FP32_DIR).eval()
        qmodel = quantize_dynamic_int8(fp32, False)

    save_quantized(qmodel, tokenizer, C.STUDENT_INT8_DIR, quantize_embeddings)

    # Round trip: load exactly like evaluate.py / app.py.
    tok2, reloaded = load_quantized(C.STUDENT_INT8_DIR)
    q_logits = predict_logits(reloaded, tok2, x_val, "cpu")
    agree = float((q_logits.argmax(1) == fp32_logits.argmax(1)).mean())
    info = {
        "fp32_val_accuracy": accuracy_from_logits(fp32_logits, y_val),
        "int8_val_accuracy": accuracy_from_logits(q_logits, y_val),
        "fp32_int8_prediction_agreement": agree,
        "fp32_size_mb": path_size_mb(C.STUDENT_FP32_DIR),
        "int8_size_mb": path_size_mb(C.STUDENT_INT8_DIR),
        "quantized_modules": ["Linear"] + (["Embedding"] if quantize_embeddings else []),
    }
    (C.STUDENT_INT8_DIR / "train_info.json").write_text(json.dumps(info, indent=2))
    print(f"[quant] FP32 val acc {info['fp32_val_accuracy'] * 100:.2f}%  "
          f"INT8 val acc {info['int8_val_accuracy'] * 100:.2f}%  agreement {agree * 100:.1f}%")
    print(f"[quant] size {info['fp32_size_mb']:.1f} MB -> {info['int8_size_mb']:.1f} MB "
          f"-> {C.STUDENT_INT8_DIR}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--student_name", default=C.STUDENT_MODEL_NAME)
    ap.add_argument("--epochs", type=int, default=C.STUDENT_EPOCHS)
    ap.add_argument("--lr", type=float, default=C.STUDENT_LR)
    ap.add_argument("--batch_size", type=int, default=C.STUDENT_BATCH)
    ap.add_argument("--temperature", type=float, default=C.KD_TEMPERATURE)
    ap.add_argument("--alpha", type=float, default=C.KD_ALPHA)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--skip_distill", action="store_true",
                    help="Only (re-)quantize an already distilled student")
    ap.add_argument("--no_embedding_quant", action="store_true",
                    help="Quantize Linear layers only")
    args = ap.parse_args()

    set_seed(C.SEED)
    device = resolve_device(args.device)
    train, val = read_jsonl(C.TRAIN_FILE), read_jsonl(C.VAL_FILE)
    x_train = preprocess_batch([r["text"] for r in train])
    x_val = preprocess_batch([r["text"] for r in val])
    y_train = [C.LABEL2ID[r["label"]] for r in train]
    y_val = list(np.array([C.LABEL2ID[r["label"]] for r in val]))

    if not args.skip_distill:
        distill(args, x_train, y_train, x_val, y_val, device)
    quantize(x_val, y_val, C.QUANTIZE_EMBEDDINGS and not args.no_embedding_quant)


if __name__ == "__main__":
    main()
