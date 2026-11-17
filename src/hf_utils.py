"""
Shared Hugging Face / PyTorch helpers for Model 3 (teacher) and Model 4
(distilled + INT8-quantized student): tokenisation, data loaders, a generic
training loop, batched prediction and dynamic quantization.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import (AutoConfig, AutoModelForSequenceClassification,
                          AutoTokenizer, get_linear_schedule_with_warmup)

from src import config as C
from src.utils import configure_quantized_engine


def encode(tokenizer, texts: list[str]) -> dict[str, torch.Tensor]:
    """Subword tokenisation (WordPiece for mBERT / DistilmBERT)."""
    return tokenizer(texts, truncation=True, max_length=C.MAX_LEN,
                     padding=True, return_tensors="pt")


def make_loader(tokenizer, texts: list[str], labels: list[int] | None = None,
                soft: np.ndarray | None = None, batch_size: int = 32,
                shuffle: bool = False, seed: int = C.SEED) -> DataLoader:
    enc = encode(tokenizer, texts)
    tensors = [enc["input_ids"], enc["attention_mask"]]
    if labels is not None:
        tensors.append(torch.tensor(labels, dtype=torch.long))
    if soft is not None:
        tensors.append(torch.tensor(soft, dtype=torch.float32))
    g = torch.Generator().manual_seed(seed)
    return DataLoader(TensorDataset(*tensors), batch_size=batch_size,
                      shuffle=shuffle, generator=g if shuffle else None)


def class_weights(labels: list[int], num_classes: int) -> torch.Tensor:
    """Inverse-sqrt-frequency weights (mild re-balancing, mean = 1)."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    w = 1.0 / np.sqrt(np.maximum(counts, 1.0))
    return torch.tensor(w / w.mean(), dtype=torch.float32)


@torch.no_grad()
def predict_logits(model, tokenizer, texts: list[str], device="cpu",
                   batch_size: int = 64) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(texts), batch_size):
        enc = encode(tokenizer, texts[i:i + batch_size])
        logits = model(input_ids=enc["input_ids"].to(device),
                       attention_mask=enc["attention_mask"].to(device)).logits
        out.append(logits.float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(C.LABELS)), dtype=np.float32)


def accuracy_from_logits(logits: np.ndarray, labels: list[int]) -> float:
    return float((logits.argmax(1) == np.asarray(labels)).mean())


def train_loop(model, loader: DataLoader, loss_fn: Callable, epochs: int, lr: float,
               device, eval_fn: Callable[[], float], save_fn: Callable[[], None],
               tag: str) -> float:
    """AdamW + linear warm-up/decay; keeps the epoch with best validation accuracy."""
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total = epochs * len(loader)
    sched = get_linear_schedule_with_warmup(opt, math.ceil(C.WARMUP_RATIO * total), total)
    best = -1.0
    for epoch in range(1, epochs + 1):
        model.train()
        t0, running = time.time(), 0.0
        for step, batch in enumerate(loader, start=1):
            batch = [b.to(device) for b in batch]
            logits = model(input_ids=batch[0], attention_mask=batch[1]).logits
            loss = loss_fn(logits, batch)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            running += loss.item()
            if step % 20 == 0 or step == len(loader):
                print(f"\r[{tag}] epoch {epoch}/{epochs} step {step}/{len(loader)} "
                      f"loss {running / step:.4f} ({time.time() - t0:.0f}s)", end="", flush=True)
        val_acc = eval_fn()
        print(f"\n[{tag}] epoch {epoch}: val accuracy {val_acc * 100:.2f}%")
        if val_acc > best:
            best = val_acc
            save_fn()
            print(f"[{tag}]   new best -> saved")
    return best


def load_hf_classifier(name_or_dir: str | Path):
    """Load tokenizer + sequence classifier with our label mapping."""
    tok = AutoTokenizer.from_pretrained(str(name_or_dir))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(name_or_dir), num_labels=len(C.LABELS),
        id2label=C.ID2LABEL, label2id=C.LABEL2ID, ignore_mismatched_sizes=True)
    return tok, model


# --------------------------------------------------------------------------- #
# Dynamic INT8 quantization
# --------------------------------------------------------------------------- #
def quantize_dynamic_int8(model: nn.Module, quantize_embeddings: bool) -> nn.Module:
    """Post-training dynamic quantization (CPU).

    nn.Linear    -> int8 weights, activations quantized on the fly per batch.
    nn.Embedding -> 8-bit per-row weights (float_qparams), optional. The
                    multilingual vocabulary (119k x 768) dominates model size,
                    so this is where most of the MB saving comes from.
    """
    from torch.ao.quantization import (default_dynamic_qconfig,
                                       float_qparams_weight_only_qconfig,
                                       quantize_dynamic)

    configure_quantized_engine()
    model = model.to("cpu").eval()
    spec = {nn.Linear: default_dynamic_qconfig}
    if quantize_embeddings:
        spec[nn.Embedding] = float_qparams_weight_only_qconfig
    return quantize_dynamic(model, qconfig_spec=spec, dtype=torch.qint8)


def save_quantized(qmodel: nn.Module, tokenizer, out_dir: Path, quantize_embeddings: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(qmodel.state_dict(), out_dir / "quantized_state_dict.pt")
    qmodel.config.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    (out_dir / "quantization.json").write_text(json.dumps({
        "method": "torch.ao.quantization.quantize_dynamic",
        "dtype": "qint8",
        "modules": ["Linear"] + (["Embedding"] if quantize_embeddings else []),
    }, indent=2), encoding="utf-8")


def load_quantized(model_dir: Path):
    """Re-create the architecture, apply the same quantization, load weights."""
    meta = json.loads((model_dir / "quantization.json").read_text(encoding="utf-8"))
    config = AutoConfig.from_pretrained(str(model_dir))
    model = AutoModelForSequenceClassification.from_config(config)
    qmodel = quantize_dynamic_int8(model, "Embedding" in meta["modules"])
    state = torch.load(model_dir / "quantized_state_dict.pt", map_location="cpu",
                       weights_only=False)
    qmodel.load_state_dict(state)
    qmodel.eval()
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    return tok, qmodel
