# 🛠️ Customer Support Ticket Router (English / Turkish)

**SWE015 – Introduction to Large Language Models · Homework · Use Case 2**

A home-appliance brand receives support messages in English and Turkish, for example
*"bulaşık makinem su boşaltmıyor, model DW-6534"* or *"My washing machine error E04"*.
For each message, this application:

1. answers greetings and small talk instantly with **ELIZA-style regex rules**,
2. **normalizes** the text, **extracts model and error codes**, and **spell-corrects** product terms with minimum edit distance,
3. predicts the **intent / support team** with a confidence score, using the **distilled, INT8-quantized multilingual BERT** (Model 4) on CPU,
4. sends **out-of-scope or low-confidence** messages to a **Human Agent**,
5. shows the predictions, confidence and live latency of **all four models side by side**.

| # | Model | Area | Where |
|---|---|---|---|
| 1 | TF-IDF (word + char n-grams) + Logistic Regression | B · Classical NLP | `src/train_baseline.py` |
| 2 | fastText-style embeddings (trained from scratch) → averaged → feed-forward NN trained with **SGD + backprop** | A · Neural nets (+B) | `src/nn_models.py`, `src/train_nn.py` |
| 3 | `bert-base-multilingual-cased` fine-tuned (WordPiece) | C · Transformers | `src/train_transformer.py` |
| 4 | Knowledge distillation → `distilbert-base-multilingual-cased`, then **dynamic INT8 quantization** | D · Efficient models | `src/distill_quantize.py` |

The architecture diagram, the technique map and a slide-by-slide outline are in
[`docs/architecture_and_slides_outline.md`](docs/architecture_and_slides_outline.md).

---

## 🚀 Quick start

```bash
# 1. Create a fresh virtual environment (Python 3.10 – 3.13)
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# 2. Install pinned dependencies
pip install -r requirements.txt
#   (Linux, optional, smaller CPU-only torch: install it first with
#    pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu)

# 3a. EITHER reproduce everything (data → 4 models → benchmark)
python run_pipeline.py

# 3b. OR download already trained weights (see "Model weights" below)
python -m src.hub_weights download --repo <your-hf-username>/support-ticket-router
```

### ▶️ Single run command for the app

```bash
python app.py
```

Then open **http://127.0.0.1:7860**. Use `python app.py --share` to get a temporary public link.

---

## 🔁 Reproducing the training and evaluation step by step

All commands are run from the repository root. Each step reads its settings from `src/config.py`.

| Step | Command | Output |
|---|---|---|
| 1. Data | `python -m src.data_prep` | `data/processed/{train,val,test}.jsonl`, `dataset_stats.json` |
| 2. Model 1 | `python -m src.train_baseline` | `models/model1_tfidf_logreg/` |
| 3. Model 2 | `python -m src.train_nn` | `models/model2_ffnn_fasttext/` |
| 4. Model 3 | `python -m src.train_transformer` | `models/model3_mbert_teacher/` |
| 5. Model 4 | `python -m src.distill_quantize` | `models/model4a_student_fp32/`, `models/model4_student_int8/` |
| 6. Benchmark | `python -m src.evaluate` | `results/results.md`, `results/results.csv`, `results/predictions.csv`, `results/hardware.json` |

`python run_pipeline.py --from transformer` resumes the pipeline from a given step. `python run_pipeline.py --only evaluate` runs a single step.

**Runtime:** steps 1–3 take a few minutes on a laptop CPU. Steps 4–5 fine-tune two Transformers. That takes tens of minutes on a CPU, or a few minutes on a GPU (a free Google Colab T4 is enough). Training picks a GPU automatically if one is available (`--device cpu|cuda`).

> ⚠️ **Latency must be measured on the machine you report.** `evaluate.py` always runs inference on CPU and logs the CPU model, cores, RAM, OS and torch version. If you train on Colab, download `models/` and run `python -m src.evaluate` on your own computer.

Useful options:

```bash
python -m src.train_transformer --epochs 3 --lr 3e-5 --batch_size 32 --device auto
python -m src.distill_quantize --epochs 4 --temperature 2.0 --alpha 0.5
python -m src.distill_quantize --skip_distill          # re-quantize only
python -m src.evaluate --runs 200 --threads 4          # >=100 runs enforced
TEACHER_MODEL=xlm-roberta-base python -m src.train_transformer   # alternative teacher
```

Quick sanity check of the classical NLP layer: `python -m src.nlp_utils`.

---

## 📊 Data

| Source | Use | License / citation |
|---|---|---|
| **Amazon MASSIVE v1.1**, locales `en-US` and `tr-TR` | Main corpus. The official train/dev/test partitions are used, so the test set is fixed. | CC BY 4.0. FitzGerald et al., *MASSIVE: A 1M-Example Multilingual Natural Language Understanding Dataset with 51 Typologically-Diverse Languages*, ACL 2023 (arXiv:2204.08582) |
| **80 hand-written appliance examples** (`data/synthetic/appliance_examples.jsonl`, 40 EN + 40 TR) | Domain examples with model numbers, error codes and typos. Split 60/20/20 per (label, language). **Only the training part is augmented.** | Written for this project |

**Label design.** There are 5 routing classes. MASSIVE intents are mapped to support teams in `src/config.py → MASSIVE_INTENT_TO_LABEL`:

| Routing label | Team | Data |
|---|---|---|
| `technical_support` | Technical Support (faults & error codes) | hand-written domain examples |
| `smart_home_app` | Smart Home & App Support | MASSIVE `iot_*` + domain examples |
| `service_scheduling` | Service Scheduling | MASSIVE `calendar_*`, `alarm_*` + domain examples |
| `orders_delivery` | Orders, Delivery & Returns | MASSIVE `takeaway_*` + domain examples |
| `out_of_scope` | → Human Agent | all **other** MASSIVE intents (weather, music, news, trivia, jokes, …) |

The mapped MASSIVE intents are **proxies** for the teams. For example, "set an alarm" stands in for appointment scheduling. Fault reports come only from the hand-written examples. That is why `evaluate.py` reports the accuracy on the hand-written test examples separately (*Domain Acc*). Report this limitation honestly in the presentation.

Down-sampling caps (`config.CAPS`) and the fixed seed (`42`) make the splits deterministic.

If the S3 download is blocked on your network, download
`https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz`
manually and run `python -m src.data_prep --massive_tar path/to/amazon-massive-dataset-1.1.tar.gz`.

---

## 🧠 Architecture (summary)

```
user text ─► ELIZA rules ──(small talk)──► instant reply
                 │ (not small talk)
                 ▼
        preprocess(): regex normalize ─► code extraction ─► edit-distance spell-correct
                 │  model_input (identical for all models)
     ┌───────────┼──────────────┬──────────────────┬───────────────────────┐
     ▼           ▼              ▼                  ▼                       
 M1 TF-IDF+LR  M2 fastText+FFNN  M3 mBERT FP32   M4 DistilmBERT INT8 (production)
     └───────────┴──────────────┴──────────────────┤
                                                   ▼
                     route(): argmax + threshold ─► team or Human Agent ─► Gradio UI
```

The full Mermaid diagram is in `docs/architecture_and_slides_outline.md`.

`src/nlp_utils.preprocess()` and `src/inference.py` are shared by **every** trainer, `evaluate.py` and `app.py`. Paths, model names and preprocessing therefore match between training, evaluation and the demo.

---

## 📦 Model weights (not in Git)

`.gitignore` excludes `models/` and every `*.pt / *.bin / *.onnx / *.safetensors / *.joblib` file. To share your trained weights:

```bash
huggingface-cli login
python -m src.hub_weights upload   --repo <your-hf-username>/support-ticket-router
python -m src.hub_weights download --repo <your-hf-username>/support-ticket-router
```

Put the Hub link here: **Weights: https://huggingface.co/&lt;your-hf-username&gt;/support-ticket-router**.

Alternative: zip the `models/` folder, upload it to Google Drive ("anyone with the link"), link it here, and tell users to unzip it into `models/`.

---

## 📁 Repository structure

```
support-ticket-router/
├── app.py                      # Gradio app (single run command)
├── run_pipeline.py             # reproduces data → 4 models → benchmark
├── requirements.txt
├── data/synthetic/appliance_examples.jsonl
├── docs/architecture_and_slides_outline.md
├── models/                     # trained weights (git-ignored)
├── results/                    # benchmark output (results.md / .csv)
└── src/
    ├── config.py               # paths, labels, hyper-parameters (single source of truth)
    ├── data_prep.py            # MASSIVE download, mapping, splits, augmentation
    ├── nlp_utils.py            # regex normalizer, code extractor, Levenshtein, ELIZA
    ├── nn_models.py            # fastText-style embeddings + feed-forward network
    ├── hf_utils.py             # tokenization, training loop, INT8 quantization
    ├── train_baseline.py       # Model 1
    ├── train_nn.py             # Model 2
    ├── train_transformer.py    # Model 3
    ├── distill_quantize.py     # Model 4
    ├── inference.py            # unified loaders + routing (used by evaluate + app)
    ├── evaluate.py             # benchmark table, hardware, failure analysis
    ├── hub_weights.py          # upload / download weights to HF Hub
    └── utils.py
```

---

## 🩹 Troubleshooting

| Problem | Fix |
|---|---|
| `RuntimeError: Didn't find engine for operation quantized::linear_prepack` (Apple Silicon / ARM) | Handled automatically: `utils.configure_quantized_engine()` selects `qnnpack` on ARM. Make sure you use the pinned torch version. |
| MASSIVE download fails | Download the archive manually and use `--massive_tar` (see Data). |
| Hugging Face download is slow or blocked | Set `HF_ENDPOINT`, or download the checkpoints once with `huggingface-cli download bert-base-multilingual-cased`. |
| Out of memory while training | `--batch_size 16`, or train on Colab. |
| App says "Model 4 not found" | Run `python -m src.distill_quantize`, or download the weights. |

---

## 📚 References

- Weizenbaum (1966), *ELIZA*. Jurafsky & Martin, *Speech and Language Processing* (3rd ed.), ch. 2 (regex, minimum edit distance) and ch. 6 (tf-idf, embeddings).
- Bojanowski et al. (2017), *Enriching Word Vectors with Subword Information* (fastText).
- Devlin et al. (2019), *BERT*. Multilingual checkpoint `bert-base-multilingual-cased`.
- Sanh et al. (2019), *DistilBERT*. Checkpoint `distilbert-base-multilingual-cased`.
- Hinton, Vinyals & Dean (2015), *Distilling the Knowledge in a Neural Network*.
- PyTorch dynamic quantization: `torch.ao.quantization.quantize_dynamic`.
- FitzGerald et al. (2023), *MASSIVE* dataset (CC BY 4.0).

## 🤖 AI usage

This repository was scaffolded with the help of an AI coding assistant (Claude). All results in `results/` must come from your own runs. See the AI-usage slide template in `docs/`.
