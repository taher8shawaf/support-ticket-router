"""
Unified inference wrappers for the four models (used by evaluate.py AND
app.py, so evaluation and the demo run the exact same code path).

Every router exposes   predict_proba(model_inputs: list[str]) -> np.ndarray
where `model_inputs` are already preprocessed with nlp_utils.preprocess()
and the returned columns follow config.LABELS.
"""
from __future__ import annotations

import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import torch

from src import config as C
from src.utils import path_size_mb


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


class Router:
    key: str = ""
    name: str = ""
    area: str = ""
    precision: str = ""
    path: Path

    def predict_proba(self, texts: list[str]) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    @property
    def size_mb(self) -> float:
        return path_size_mb(self.path)


class BaselineRouter(Router):
    key, name, area, precision = "m1", "M1 TF-IDF + LogReg", "B", "FP64 (sklearn)"

    def __init__(self, path: Path = C.BASELINE_DIR):
        self.path = path
        self.pipe = joblib.load(path / "tfidf_logreg.joblib")
        classes = list(self.pipe.classes_)
        self._order = [classes.index(l) for l in C.LABELS]

    def predict_proba(self, texts):
        return self.pipe.predict_proba(texts)[:, self._order]


class FFNNRouter(Router):
    key, name, area, precision = "m2", "M2 fastText-avg + FFNN (SGD)", "A+B", "FP32"

    def __init__(self, path: Path = C.FFNN_DIR):
        from src.nn_models import FeedForwardNet, SentenceEncoder

        self.path = path
        self.encoder = SentenceEncoder.load(path)
        cfg = json.loads((path / "ffnn_config.json").read_text())
        self.model = FeedForwardNet(cfg["in_dim"], cfg["num_classes"],
                                    tuple(cfg["hidden"]), cfg["dropout"])
        self.model.load_state_dict(torch.load(path / "ffnn.pt", map_location="cpu",
                                              weights_only=True))
        self.model.eval()

    @torch.no_grad()
    def predict_proba(self, texts):
        x = torch.from_numpy(self.encoder.encode(texts))
        return torch.softmax(self.model(x), dim=-1).numpy()


class HFRouter(Router):
    """FP32 Hugging Face classifier (Model 3 teacher or distilled FP32 student)."""

    def __init__(self, path: Path, key: str, name: str, area: str):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.path, self.key, self.name, self.area = path, key, name, area
        self.precision = "FP32"
        self.tokenizer = AutoTokenizer.from_pretrained(str(path))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(path)).eval()

    def predict_proba(self, texts):
        from src.hf_utils import predict_logits

        return softmax(predict_logits(self.model, self.tokenizer, texts, "cpu"))


class QuantizedRouter(HFRouter):
    """Model 4: distilled student with dynamic INT8 quantization (production)."""

    def __init__(self, path: Path = C.STUDENT_INT8_DIR):
        from src.hf_utils import load_quantized

        self.path, self.key = path, "m4"
        self.name, self.area, self.precision = "M4 Distilled mBERT INT8", "C+D", "INT8 (dynamic)"
        self.tokenizer, self.model = load_quantized(path)


def _builders(include_ablation: bool):
    b = OrderedDict()
    b["m1"] = (C.BASELINE_DIR, lambda: BaselineRouter())
    b["m2"] = (C.FFNN_DIR, lambda: FFNNRouter())
    b["m3"] = (C.TEACHER_DIR, lambda: HFRouter(C.TEACHER_DIR, "m3",
                                               "M3 mBERT fine-tuned (teacher)", "C"))
    if include_ablation:
        b["m4a"] = (C.STUDENT_FP32_DIR, lambda: HFRouter(
            C.STUDENT_FP32_DIR, "m4a", "M4a Distilled student FP32 (ablation)", "C+D"))
    b["m4"] = (C.STUDENT_INT8_DIR, lambda: QuantizedRouter())
    return b


def load_routers(include_ablation: bool = False, verbose: bool = True) -> "OrderedDict[str, Router]":
    """Load every trained model; missing models are skipped with a message."""
    routers = OrderedDict()
    for key, (path, build) in _builders(include_ablation).items():
        if not Path(path).exists():
            if verbose:
                print(f"[load] {key}: not found at {path} - train it first (skipped)")
            continue
        routers[key] = build()
        if verbose:
            print(f"[load] {key}: {routers[key].name} ({routers[key].size_mb:.1f} MB)")
    return routers


@dataclass
class RoutingDecision:
    label: str
    confidence: float
    team: str
    to_human: bool
    reason: str


def route(probs: np.ndarray, threshold: float = C.CONFIDENCE_THRESHOLD) -> RoutingDecision:
    """Turn one probability vector into a routing decision with HITL fallback."""
    idx = int(np.argmax(probs))
    label, conf = C.LABELS[idx], float(probs[idx])
    if label == C.OOS_LABEL:
        return RoutingDecision(label, conf, C.HUMAN_AGENT, True, "out of scope")
    if conf < threshold:
        return RoutingDecision(label, conf, C.HUMAN_AGENT, True,
                               f"low confidence ({conf:.2f} < {threshold:.2f})")
    return RoutingDecision(label, conf, C.TEAM_NAMES[label], False, "confident")
