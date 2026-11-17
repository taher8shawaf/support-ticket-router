"""
Central configuration for the Customer Support Ticket Router.

Every script (data prep, the four trainers, evaluation and the app) imports
paths, label names and hyper-parameters from THIS file, so the pipeline is
guaranteed to be identical between training, evaluation and app.py.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SYNTHETIC_FILE = DATA_DIR / "synthetic" / "appliance_examples.jsonl"

TRAIN_FILE = PROCESSED_DIR / "train.jsonl"
VAL_FILE = PROCESSED_DIR / "val.jsonl"
TEST_FILE = PROCESSED_DIR / "test.jsonl"
STATS_FILE = PROCESSED_DIR / "dataset_stats.json"

MODELS_DIR = ROOT_DIR / "models"
BASELINE_DIR = MODELS_DIR / "model1_tfidf_logreg"          # Model 1
FFNN_DIR = MODELS_DIR / "model2_ffnn_fasttext"             # Model 2
TEACHER_DIR = MODELS_DIR / "model3_mbert_teacher"          # Model 3
STUDENT_FP32_DIR = MODELS_DIR / "model4a_student_fp32"     # distilled student (ablation row)
STUDENT_INT8_DIR = MODELS_DIR / "model4_student_int8"      # Model 4 (production)

RESULTS_DIR = ROOT_DIR / "results"

# --------------------------------------------------------------------------- #
# Dataset: Amazon MASSIVE v1.1 (CC BY 4.0)
# --------------------------------------------------------------------------- #
MASSIVE_URL = (
    "https://amazon-massive-nlu-dataset.s3.amazonaws.com/"
    "amazon-massive-dataset-1.1.tar.gz"
)
MASSIVE_TAR = RAW_DIR / "amazon-massive-dataset-1.1.tar.gz"
LOCALES = ["en-US", "tr-TR"]

# Routing labels (= support teams + out_of_scope). Order defines class ids.
LABELS = [
    "technical_support",
    "smart_home_app",
    "service_scheduling",
    "orders_delivery",
    "out_of_scope",
]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for l, i in LABEL2ID.items()}
OOS_LABEL = "out_of_scope"

# Human-readable team names shown in the app.
TEAM_NAMES = {
    "technical_support": "Technical Support (faults & error codes)",
    "smart_home_app": "Smart Home & App Support",
    "service_scheduling": "Service Scheduling (technician visits)",
    "orders_delivery": "Orders, Delivery & Returns",
    "out_of_scope": "Human Agent",
}
HUMAN_AGENT = "Human Agent"

# MASSIVE intent -> routing label. Every intent NOT listed here becomes
# out_of_scope (weather, music, news, trivia, jokes, ...). The mapped MASSIVE
# intents act as *proxies* for the team (e.g. "turn on the robot vacuum" is a
# smart-home device request). Fault reports (technical_support) come only
# from the hand-written domain examples, see data/synthetic/.
MASSIVE_INTENT_TO_LABEL = {
    # Smart home / connected-device control  -> Smart Home & App Support
    "iot_cleaning": "smart_home_app",
    "iot_coffee": "smart_home_app",
    "iot_hue_lightchange": "smart_home_app",
    "iot_hue_lightdim": "smart_home_app",
    "iot_hue_lightoff": "smart_home_app",
    "iot_hue_lighton": "smart_home_app",
    "iot_hue_lightup": "smart_home_app",
    "iot_wemo_off": "smart_home_app",
    "iot_wemo_on": "smart_home_app",
    # Calendar / reminders -> Service Scheduling
    "calendar_set": "service_scheduling",
    "calendar_query": "service_scheduling",
    "calendar_remove": "service_scheduling",
    "alarm_set": "service_scheduling",
    "alarm_query": "service_scheduling",
    "alarm_remove": "service_scheduling",
    # Order placing / order status -> Orders, Delivery & Returns
    "takeaway_order": "orders_delivery",
    "takeaway_query": "orders_delivery",
}

# Deterministic down-sampling caps per (partition, locale, label).
# None = keep everything. Keeps CPU training time reasonable and limits the
# out_of_scope class, which would otherwise be ~75% of MASSIVE.
CAPS = {
    "train": {"smart_home_app": None, "service_scheduling": 700,
              "orders_delivery": None, "out_of_scope": 1000},
    "dev": {"smart_home_app": None, "service_scheduling": 150,
            "orders_delivery": None, "out_of_scope": 200},
    "test": {"smart_home_app": None, "service_scheduling": None,
             "orders_delivery": None, "out_of_scope": 300},
}

# Hand-written domain examples: per (label, lang) group, this many go to val
# and test; the rest go to train and are then augmented (train only).
SYNTH_VAL_FRACTION = 0.2
SYNTH_TEST_FRACTION = 0.2
SYNTH_AUGMENT_PER_EXAMPLE = 8

SEED = 42

# --------------------------------------------------------------------------- #
# Routing
# --------------------------------------------------------------------------- #
# If the top softmax probability is below this, the ticket goes to a human.
CONFIDENCE_THRESHOLD = 0.50

# --------------------------------------------------------------------------- #
# Model 1 - TF-IDF + Logistic Regression
# --------------------------------------------------------------------------- #
BASELINE_C = 5.0

# --------------------------------------------------------------------------- #
# Model 2 - fastText-style embeddings + feed-forward network (SGD)
# --------------------------------------------------------------------------- #
FT_DIM = 100            # embedding size
FT_BUCKETS = 50_000     # hashed character n-gram buckets
FT_MINN, FT_MAXN = 3, 5 # character n-gram range
FT_WINDOW = 3
FT_NEG = 5
FT_EPOCHS = 8
FT_LR = 0.01
FT_BATCH = 1024

NN_HIDDEN = (256, 128)
NN_DROPOUT = 0.3
NN_LR = 0.05
NN_MOMENTUM = 0.9
NN_WEIGHT_DECAY = 1e-4
NN_EPOCHS = 80
NN_BATCH = 64
NN_PATIENCE = 12

# --------------------------------------------------------------------------- #
# Model 3 / 4 - Transformers
# --------------------------------------------------------------------------- #
# Environment variables allow swapping checkpoints (e.g. xlm-roberta-base)
# without code changes. Defaults follow the assignment brief.
TEACHER_MODEL_NAME = os.environ.get("TEACHER_MODEL", "bert-base-multilingual-cased")
STUDENT_MODEL_NAME = os.environ.get("STUDENT_MODEL", "distilbert-base-multilingual-cased")

MAX_LEN = 64
TEACHER_EPOCHS = 3
TEACHER_LR = 3e-5
TEACHER_BATCH = 32
STUDENT_EPOCHS = 4
STUDENT_LR = 5e-5
STUDENT_BATCH = 32
WARMUP_RATIO = 0.1

KD_TEMPERATURE = 2.0
KD_ALPHA = 0.5          # weight of hard-label CE; (1-alpha) on soft KL term

# Also quantize the (very large) multilingual word-embedding matrix to 8 bit.
QUANTIZE_EMBEDDINGS = True

# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
LATENCY_RUNS = 200      # >= 100 single-message runs per model
LATENCY_WARMUP = 10
