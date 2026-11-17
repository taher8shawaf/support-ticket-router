# Benchmark results - Customer Support Ticket Router

All models evaluated on the same held-out test set: **2168 messages**
(1084 EN / 1084 TR, 600 out-of-scope,
18 hand-written domain examples).

## Main results table (R4)

| Model | Area | Precision | Accuracy (%) | OOS Recall (%) | Size (MB) | Latency mean (ms) | Latency p95 (ms) |
|---|---|---|---|---|---|---|---|
| M1 TF-IDF + LogReg | B | FP64 (sklearn) | 94.00 | 94.00 | 1.5 | 1.65 | 2.23 |
| M2 fastText-avg + FFNN (SGD) | A+B | FP32 | 91.84 | 88.00 | 21.5 | 0.39 | 0.52 |
| M3 mBERT fine-tuned (teacher) | C | FP32 | 94.46 | 91.83 | 682.2 | 32.85 | 42.51 |
| M4a Distilled student FP32 (ablation) | C+D | FP32 | 93.13 | 94.00 | 520.0 | 15.39 | 18.07 |
| M4 Distilled mBERT INT8 | C+D | INT8 (dynamic) | 89.53 | 97.50 | 134.0 | 14.45 | 17.41 |

*Latency = average over **200 single-message runs** on CPU after 10 warm-up runs
(tokenisation/vectorisation + forward pass + softmax). Shared preprocessing
(normalisation + spell correction) adds **0.160 ms** per message for every model.
Size = total bytes of the saved model directory (weights + tokenizer/vocab + config).*

## Additional metrics

| Model | OOS Recall + fallback (%) | In-scope Acc (%) | Macro-F1 (%) | Domain Acc (%) | EN Acc (%) | TR Acc (%) |
|---|---|---|---|---|---|---|
| M1 TF-IDF + LogReg | 95.17 | 94.01 | 85.65 | 50.00 | 93.27 | 94.74 |
| M2 fastText-avg + FFNN (SGD) | 88.17 | 93.30 | 80.60 | 55.56 | 91.88 | 91.79 |
| M3 mBERT fine-tuned (teacher) | 92.00 | 95.47 | 89.09 | 83.33 | 95.11 | 93.82 |
| M4a Distilled student FP32 (ablation) | 94.00 | 92.79 | 82.84 | 61.11 | 93.91 | 92.34 |
| M4 Distilled mBERT INT8 | 97.67 | 86.48 | 80.51 | 55.56 | 90.50 | 88.56 |

*OOS Recall = share of out-of-scope test messages predicted as `out_of_scope`.
"OOS Recall + fallback" also counts messages sent to the Human Agent because the
top probability was below the threshold (0.50).*

## Hardware

- **cpu**: 12th Gen Intel(R) Core(TM) i7-12650H
- **logical_cores**: 16
- **ram**: 15.7 GB
- **os**: Windows 11
- **python**: 3.12.10
- **torch**: 2.5.1+cpu
- **torch_threads**: 10
- **quantized_engine**: x86

## Confusion matrix - M4 Distilled mBERT INT8

| true \ pred | technical_support | smart_home_app | service_scheduling | orders_delivery | out_of_scope |
|---|---|---|---|---|---|
| technical_support | 3 | 2 | 0 | 0 | 1 |
| smart_home_app | 2 | 412 | 0 | 1 | 29 |
| service_scheduling | 1 | 0 | 846 | 1 | 152 |
| orders_delivery | 0 | 0 | 0 | 95 | 23 |
| out_of_scope | 1 | 1 | 4 | 9 | 585 |

## Failure analysis - M4 Distilled mBERT INT8 (227 errors of 2168)

**1. [EN, domain] in-scope rejected as out-of-scope (false reject)**

- Message: `fridge is making a buzzing sound all night`
- Model input after preprocessing: `fridge is making a buzzing sound all night`
- True label: `technical_support` (MASSIVE intent: `handwritten`)
- Predicted by M4 Distilled mBERT INT8: `out_of_scope` (confidence 0.97)
- All models: M1=service_scheduling (0.36), M2=smart_home_app (0.88), M3=out_of_scope (0.98), M4a=out_of_scope (0.56), M4=out_of_scope (0.97)

**2. [TR, domain] wrong team**

- Message: `çamaşır makinesinin kapağı yıkama bitince açılmıyor`
- Model input after preprocessing: `çamaşır makinesinin kapağı yıkama bitince akıtmıyor`
- True label: `technical_support` (MASSIVE intent: `handwritten`)
- Predicted by M4 Distilled mBERT INT8: `smart_home_app` (confidence 0.96)
- All models: M1=smart_home_app (0.72), M2=smart_home_app (0.98), M3=technical_support (0.99), M4a=smart_home_app (0.98), M4=smart_home_app (0.96)

**3. [TR, domain] wrong team**

- Message: `bulaşık makinesi bitince bildirim gelmiyor`
- Model input after preprocessing: `bulaşık makinesi bitince bildirim gelmiyor`
- True label: `smart_home_app` (MASSIVE intent: `handwritten`)
- Predicted by M4 Distilled mBERT INT8: `technical_support` (confidence 0.96)
- All models: M1=technical_support (0.93), M2=technical_support (1.00), M3=technical_support (0.99), M4a=technical_support (0.99), M4=technical_support (0.96)

**4. [TR, domain] in-scope rejected as out-of-scope (false reject)**

- Message: `akıllı kurutma makinemin yazılımını nasıl güncellerim`
- Model input after preprocessing: `akıllı kurutma makinemin yazılımını nasıl güncellerim`
- True label: `smart_home_app` (MASSIVE intent: `handwritten`)
- Predicted by M4 Distilled mBERT INT8: `out_of_scope` (confidence 0.87)
- All models: M1=out_of_scope (0.46), M2=out_of_scope (0.98), M3=smart_home_app (0.53), M4a=out_of_scope (0.88), M4=out_of_scope (0.87)

**5. [EN, massive] out-of-scope accepted as a team (false accept)**

- Message: `give me the details on purchase order`
- Model input after preprocessing: `give me the details on purchase order`
- True label: `out_of_scope` (MASSIVE intent: `lists_query`)
- Predicted by M4 Distilled mBERT INT8: `orders_delivery` (confidence 0.99)
- All models: M1=orders_delivery (0.77), M2=orders_delivery (0.86), M3=orders_delivery (0.99), M4a=orders_delivery (0.99), M4=orders_delivery (0.99)

