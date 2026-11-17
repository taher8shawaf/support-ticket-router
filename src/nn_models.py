"""
Model 2 building blocks.

  * Area B: a fastText-style skip-gram embedding model (words + hashed
    character n-grams, negative sampling) trained FROM SCRATCH on the training
    split. Character n-grams give vectors to unseen / misspelled words and to
    Turkish inflections ("makinemin", "makinesinin", ...).
  * Area A: a feed-forward network (MLP: Linear-ReLU-Dropout x2 -> Linear)
    that classifies the AVERAGED word embeddings of a message. It is trained
    with plain SGD (+momentum) and back-propagation in train_nn.py.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src import config as C
from src.nlp_utils import tokenize


def fnv1a_32(s: str) -> int:
    """Deterministic 32-bit FNV-1a hash (Python's hash() is randomised)."""
    h = 0x811C9DC5
    for b in s.encode("utf-8"):
        h ^= b
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h


class SubwordVocab:
    """Word vocabulary + hashed character n-gram buckets.

    Id layout of the input-embedding table:
        0                 padding
        1 .. V            whole words
        V+1 .. V+B        character n-gram buckets
    """

    def __init__(self, words: list[str], counts: list[int],
                 buckets: int = C.FT_BUCKETS, minn: int = C.FT_MINN, maxn: int = C.FT_MAXN):
        self.words, self.counts = words, counts
        self.word2id = {w: i + 1 for i, w in enumerate(words)}
        self.buckets, self.minn, self.maxn = buckets, minn, maxn

    @classmethod
    def build(cls, sentences: list[list[str]], min_count: int = 1) -> "SubwordVocab":
        cnt = Counter(t for s in sentences for t in s)
        items = sorted([(w, c) for w, c in cnt.items() if c >= min_count],
                       key=lambda x: (-x[1], x[0]))
        return cls([w for w, _ in items], [c for _, c in items])

    @property
    def num_embeddings(self) -> int:
        return 1 + len(self.words) + self.buckets

    def ngrams(self, word: str) -> list[str]:
        w = f"<{word}>"
        return [w[i:i + n] for n in range(self.minn, self.maxn + 1)
                for i in range(len(w) - n + 1)]

    def subword_ids(self, word: str) -> list[int]:
        ids = [self.word2id[word]] if word in self.word2id else []
        base = 1 + len(self.words)
        ids += [base + fnv1a_32(g) % self.buckets for g in self.ngrams(word)]
        return ids or [0]

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"words": self.words, "counts": self.counts,
                                    "buckets": self.buckets, "minn": self.minn,
                                    "maxn": self.maxn}, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "SubwordVocab":
        d = json.loads(path.read_text(encoding="utf-8"))
        return cls(d["words"], d["counts"], d["buckets"], d["minn"], d["maxn"])


class FastTextSkipGram(nn.Module):
    """Skip-gram with negative sampling where the centre-word vector is the
    mean of its word + character n-gram vectors (Bojanowski et al., 2017)."""

    def __init__(self, vocab: SubwordVocab, dim: int = C.FT_DIM):
        super().__init__()
        self.in_emb = nn.Embedding(vocab.num_embeddings, dim, padding_idx=0)
        self.out_emb = nn.Embedding(len(vocab.words) + 1, dim, padding_idx=0)
        nn.init.uniform_(self.in_emb.weight, -0.5 / dim, 0.5 / dim)
        nn.init.zeros_(self.out_emb.weight)
        with torch.no_grad():
            self.in_emb.weight[0].zero_()
        # Padded [V+1, L] table of subword ids for every vocabulary word.
        subs = [[0]] + [vocab.subword_ids(w) for w in vocab.words]
        width = max(len(s) for s in subs)
        table = torch.zeros(len(subs), width, dtype=torch.long)
        for i, s in enumerate(subs):
            table[i, :len(s)] = torch.tensor(s)
        self.register_buffer("sub_table", table, persistent=False)

    def centre_vectors(self, word_ids: torch.Tensor) -> torch.Tensor:
        subs = self.sub_table[word_ids]                          # [B, L]
        mask = (subs != 0).float().unsqueeze(-1)                 # [B, L, 1]
        summed = (self.in_emb(subs) * mask).sum(1)
        return summed / mask.sum(1).clamp(min=1.0)

    def forward(self, centre: torch.Tensor, context: torch.Tensor,
                negatives: torch.Tensor) -> torch.Tensor:
        u = self.centre_vectors(centre)                          # [B, D]
        v_pos = self.out_emb(context)                            # [B, D]
        v_neg = self.out_emb(negatives)                          # [B, K, D]
        pos = F.logsigmoid((u * v_pos).sum(-1))
        neg = F.logsigmoid(-torch.bmm(v_neg, u.unsqueeze(-1)).squeeze(-1)).sum(-1)
        return -(pos + neg).mean()


class SentenceEncoder:
    """Averages fastText-style word vectors -> one fixed-size message vector."""

    def __init__(self, vocab: SubwordVocab, in_emb: np.ndarray):
        self.vocab, self.emb = vocab, in_emb.astype(np.float32)
        self.dim = in_emb.shape[1]
        self._cache: dict[str, np.ndarray] = {}

    def word_vector(self, word: str) -> np.ndarray:
        vec = self._cache.get(word)
        if vec is None:
            vec = self.emb[self.vocab.subword_ids(word)].mean(0)
            if len(self._cache) < 200_000:
                self._cache[word] = vec
        return vec

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            toks = tokenize(t)
            if toks:
                out[i] = np.mean([self.word_vector(w) for w in toks], axis=0)
        return out

    def save(self, directory: Path) -> None:
        self.vocab.save(directory / "ft_vocab.json")
        torch.save(torch.from_numpy(self.emb), directory / "ft_embeddings.pt")

    @classmethod
    def load(cls, directory: Path) -> "SentenceEncoder":
        vocab = SubwordVocab.load(directory / "ft_vocab.json")
        emb = torch.load(directory / "ft_embeddings.pt", map_location="cpu", weights_only=True)
        return cls(vocab, emb.numpy())


class FeedForwardNet(nn.Module):
    """MLP: [standardise] -> Linear -> ReLU -> Dropout -> ... -> Linear (logits)."""

    def __init__(self, in_dim: int, num_classes: int,
                 hidden: tuple[int, ...] = C.NN_HIDDEN, dropout: float = C.NN_DROPOUT):
        super().__init__()
        self.register_buffer("mean", torch.zeros(in_dim))
        self.register_buffer("std", torch.ones(in_dim))
        layers, prev = [], in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)
        self.config = {"in_dim": in_dim, "num_classes": num_classes,
                       "hidden": list(hidden), "dropout": dropout}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net((x - self.mean) / self.std)
