"""
Model 2 (Area A + B) - Feed-forward neural network on averaged fastText-style
embeddings.

  Stage 1 (Area B): train fastText-style skip-gram embeddings (word + hashed
                    char n-grams, negative sampling) on the TRAINING split only.
  Stage 2 (Area A): average the word vectors of each message and train an MLP
                    (ReLU + Dropout) with torch.optim.SGD (momentum) and
                    back-propagation (loss.backward()), early-stopped on
                    validation accuracy.

Run:  python -m src.train_nn
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch
import torch.nn as nn

from src import config as C
from src.nlp_utils import preprocess_batch, tokenize
from src.nn_models import FastTextSkipGram, FeedForwardNet, SentenceEncoder, SubwordVocab
from src.utils import read_jsonl, set_seed


# --------------------------------------------------------------------------- #
# Stage 1: fastText-style embeddings
# --------------------------------------------------------------------------- #
def train_embeddings(sentences: list[list[str]]) -> SentenceEncoder:
    vocab = SubwordVocab.build(sentences, min_count=1)
    print(f"[nn] embedding vocab: {len(vocab.words)} words + {vocab.buckets} n-gram buckets")
    w2i = vocab.word2id

    # Skip-gram (centre, context) pairs within a symmetric window.
    centres, contexts = [], []
    for s in sentences:
        ids = [w2i[t] for t in s if t in w2i]
        for i, c in enumerate(ids):
            for j in range(max(0, i - C.FT_WINDOW), min(len(ids), i + C.FT_WINDOW + 1)):
                if j != i:
                    centres.append(c)
                    contexts.append(ids[j])
    centres_t = torch.tensor(centres, dtype=torch.long)
    contexts_t = torch.tensor(contexts, dtype=torch.long)
    print(f"[nn] {len(centres)} skip-gram pairs")

    # Negative-sampling distribution: unigram^0.75 (index 0 = padding -> 0).
    freq = torch.tensor([0.0] + [float(c) for c in vocab.counts]) ** 0.75
    neg_dist = freq / freq.sum()

    model = FastTextSkipGram(vocab, C.FT_DIM)
    opt = torch.optim.Adam(model.parameters(), lr=C.FT_LR)
    g = torch.Generator().manual_seed(C.SEED)
    for epoch in range(1, C.FT_EPOCHS + 1):
        perm = torch.randperm(len(centres_t), generator=g)
        total, t0 = 0.0, time.time()
        for i in range(0, len(perm), C.FT_BATCH):
            idx = perm[i:i + C.FT_BATCH]
            negs = torch.multinomial(neg_dist, len(idx) * C.FT_NEG, replacement=True,
                                     generator=g).view(len(idx), C.FT_NEG)
            loss = model(centres_t[idx], contexts_t[idx], negs)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        print(f"[nn] embeddings epoch {epoch}/{C.FT_EPOCHS}: "
              f"loss {total / len(perm):.4f} ({time.time() - t0:.1f}s)")
    return SentenceEncoder(vocab, model.in_emb.weight.detach().cpu().numpy())


# --------------------------------------------------------------------------- #
# Stage 2: feed-forward network with SGD
# --------------------------------------------------------------------------- #
def train_mlp(x_tr: np.ndarray, y_tr: np.ndarray, x_va: np.ndarray, y_va: np.ndarray):
    model = FeedForwardNet(x_tr.shape[1], len(C.LABELS))
    model.mean.copy_(torch.from_numpy(x_tr.mean(0)))
    model.std.copy_(torch.from_numpy(x_tr.std(0) + 1e-6))

    counts = np.bincount(y_tr, minlength=len(C.LABELS)).astype(np.float64)
    w = 1.0 / np.sqrt(np.maximum(counts, 1.0))
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(w / w.mean(), dtype=torch.float32))
    opt = torch.optim.SGD(model.parameters(), lr=C.NN_LR, momentum=C.NN_MOMENTUM,
                          weight_decay=C.NN_WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=C.NN_EPOCHS)

    xt, yt = torch.from_numpy(x_tr), torch.from_numpy(y_tr)
    xv = torch.from_numpy(x_va)
    g = torch.Generator().manual_seed(C.SEED)
    best_acc, best_state, bad = -1.0, None, 0
    for epoch in range(1, C.NN_EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(xt), generator=g)
        total = 0.0
        for i in range(0, len(perm), C.NN_BATCH):
            idx = perm[i:i + C.NN_BATCH]
            logits = model(xt[idx])                 # forward pass
            loss = loss_fn(logits, yt[idx])
            opt.zero_grad()
            loss.backward()                         # back-propagation
            opt.step()                              # SGD update
            total += loss.item() * len(idx)
        sched.step()
        model.eval()
        with torch.no_grad():
            val_acc = float((model(xv).argmax(1).numpy() == y_va).mean())
        if epoch == 1 or epoch % 5 == 0:
            print(f"[nn] mlp epoch {epoch:3d}: train loss {total / len(perm):.4f} "
                  f"val acc {val_acc * 100:.2f}%")
        if val_acc > best_acc:
            best_acc, bad = val_acc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= C.NN_PATIENCE:
                print(f"[nn] early stopping at epoch {epoch}")
                break
    model.load_state_dict(best_state)
    return model, best_acc


def main() -> None:
    set_seed(C.SEED)
    train, val = read_jsonl(C.TRAIN_FILE), read_jsonl(C.VAL_FILE)
    x_train_txt = preprocess_batch([r["text"] for r in train])
    x_val_txt = preprocess_batch([r["text"] for r in val])
    y_train = np.array([C.LABEL2ID[r["label"]] for r in train])
    y_val = np.array([C.LABEL2ID[r["label"]] for r in val])

    encoder = train_embeddings([tokenize(t) for t in x_train_txt])
    x_tr, x_va = encoder.encode(x_train_txt), encoder.encode(x_val_txt)

    model, best = train_mlp(x_tr, y_train, x_va, y_val)
    print(f"[nn] best validation accuracy: {best * 100:.2f}%")

    C.FFNN_DIR.mkdir(parents=True, exist_ok=True)
    encoder.save(C.FFNN_DIR)
    torch.save(model.state_dict(), C.FFNN_DIR / "ffnn.pt")
    (C.FFNN_DIR / "ffnn_config.json").write_text(json.dumps(model.config, indent=2))
    (C.FFNN_DIR / "train_info.json").write_text(json.dumps(
        {"val_accuracy": best, "optimizer": "SGD", "lr": C.NN_LR,
         "momentum": C.NN_MOMENTUM, "embedding_dim": C.FT_DIM}, indent=2))
    print(f"[nn] saved to {C.FFNN_DIR}")


if __name__ == "__main__":
    main()
