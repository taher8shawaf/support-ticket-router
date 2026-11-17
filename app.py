"""
Interactive demo - Customer Support Ticket Router (Gradio).

Flow for every message (identical to src/evaluate.py):
  1. ELIZA-style small-talk rules (src/nlp_utils.check_small_talk)
     -> instant rule-based reply, no model inference.
  2. Preprocessing (src/nlp_utils.preprocess): regex normalisation,
     model-code / error-code extraction, edit-distance spell correction.
  3. Model 4 (distilled multilingual BERT, dynamic INT8, CPU) = production
     routing decision with confidence threshold -> team or "Human Agent".
  4. Models 1-4 side by side with prediction, confidence, team and live latency.

Run:  python app.py            (then open http://127.0.0.1:7860)
      python app.py --share    (temporary public Gradio link)
"""
from __future__ import annotations

import argparse
import time

import gradio as gr
import pandas as pd

from src import config as C
from src.inference import load_routers, route
from src.nlp_utils import check_small_talk, preprocess
from src.utils import get_hardware_info

print("[app] loading models ...")
ROUTERS = load_routers(include_ablation=False)
PRIMARY = "m4" if "m4" in ROUTERS else None
HW = get_hardware_info()
TABLE_COLS = ["Model", "Precision", "Predicted intent", "Confidence",
              "Routed to", "Latency (ms)"]

EXAMPLES = [
    ("🇬🇧 English fault", "My washing machine shows error E04 and won't drain"),
    ("🇹🇷 Turkish + model code + typo", "bulasik makinem su boşaltmıyor, model DW-6534"),
    ("🇬🇧 Typos", "the refridgerator RF-1180X is leaking, washng machne too"),
    ("🇹🇷 Appointment", "yarın sabah buzdolabım için teknisyen gelebilir mi"),
    ("❓ Out of scope", "what's the weather going to be like in Istanbul tomorrow?"),
    ("👋 Greeting", "Merhaba!"),
]


def _empty_table(note: str) -> pd.DataFrame:
    return pd.DataFrame([[r.name, r.precision, note, "", "", ""] for r in ROUTERS.values()],
                        columns=TABLE_COLS)


def analyze(text: str, threshold: float):
    text = (text or "").strip()
    if not text:
        return ("### ✏️ Type a message first.", "", _empty_table("—"), None)

    # 1) ELIZA-style small talk check (before any model inference)
    t0 = time.perf_counter()
    small = check_small_talk(text)
    if small is not None:
        ms = (time.perf_counter() - t0) * 1000
        decision = (f"### 💬 Small talk detected — instant rule-based reply ({ms:.2f} ms)\n\n"
                    f"> {small.reply}\n\n"
                    f"No model was run. Matched ELIZA rule: `{small.pattern}`")
        prep = f"**Language of rule:** `{small.lang}`"
        return decision, prep, _empty_table("skipped (small talk)"), None

    # 2) Preprocessing (shared by all models)
    t0 = time.perf_counter()
    pre = preprocess(text)
    prep_ms = (time.perf_counter() - t0) * 1000
    fixes = ", ".join(f"`{c.original}` → `{c.corrected}` (edit distance {c.distance})"
                      for c in pre.corrections) or "none"
    prep_md = (f"**Normalized text:** `{pre.normalized}`\n\n"
               f"**Spell-corrected product terms:** {fixes}\n\n"
               f"**Model input:** `{pre.model_input}`\n\n"
               f"**Extracted model code(s):** {', '.join(f'`{c}`' for c in pre.model_codes) or 'none'}"
               f" &nbsp;|&nbsp; **Error code(s):** "
               f"{', '.join(f'`{c}`' for c in pre.error_codes) or 'none'}\n\n"
               f"*Preprocessing time: {prep_ms:.2f} ms*")

    # 3) + 4) all models side by side, Model 4 drives the decision
    rows, primary_probs, primary_dec, primary_ms = [], None, None, None
    for key, router in ROUTERS.items():
        t0 = time.perf_counter()
        probs = router.predict_proba([pre.model_input])[0]
        ms = (time.perf_counter() - t0) * 1000
        dec = route(probs, threshold)
        star = " ⭐ (production)" if key == PRIMARY else ""
        rows.append([router.name + star, router.precision, dec.label,
                     f"{dec.confidence:.3f}", dec.team, f"{ms:.1f}"])
        if key == PRIMARY:
            primary_probs, primary_dec, primary_ms = probs, dec, ms

    if primary_dec is None:
        decision = ("### ⚠️ Model 4 (INT8) not found\n"
                    "Run `python -m src.distill_quantize` first. Other models are shown below.")
        label_out = None
    else:
        icon = "🧑‍💼" if primary_dec.to_human else "✅"
        decision = (f"### {icon} Routed to: **{primary_dec.team}**\n\n"
                    f"Intent `{primary_dec.label}` · confidence **{primary_dec.confidence:.1%}** · "
                    f"reason: {primary_dec.reason} · Model 4 latency **{primary_ms:.1f} ms**")
        if pre.model_codes or pre.error_codes:
            decision += ("\n\nTicket metadata: "
                         + " ".join(f"`model={c}`" for c in pre.model_codes)
                         + " " + " ".join(f"`error={c}`" for c in pre.error_codes))
        label_out = {C.LABELS[i]: float(p) for i, p in enumerate(primary_probs)}
    return decision, prep_md, pd.DataFrame(rows, columns=TABLE_COLS), label_out


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Support Ticket Router", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# 🛠️ Customer Support Ticket Router (EN / TR)\n"
            "Type a support message in **English or Turkish**. Greetings are answered by "
            "ELIZA-style rules; everything else is routed by **Model 4 — distilled multilingual "
            "BERT, dynamic INT8 on CPU**. Models 1–4 are compared side by side.")
        with gr.Row():
            with gr.Column(scale=3):
                inp = gr.Textbox(label="Customer message", lines=3,
                                 placeholder="e.g. bulaşık makinem su boşaltmıyor, model DW-6534")
                with gr.Row():
                    submit = gr.Button("Route message", variant="primary")
                    clear = gr.Button("Clear")
                threshold = gr.Slider(0.0, 1.0, value=C.CONFIDENCE_THRESHOLD, step=0.05,
                                      label="Human-Agent confidence threshold")
                gr.Markdown("**Quick examples**")
                with gr.Row():
                    ex_buttons = [(gr.Button(lbl, size="sm"), txt) for lbl, txt in EXAMPLES]
            with gr.Column(scale=2):
                decision = gr.Markdown("### Routing decision will appear here")
                probs = gr.Label(label="Model 4 class probabilities", num_top_classes=5)
        prep = gr.Markdown(label="Preprocessing")
        table = gr.Dataframe(headers=TABLE_COLS, label="All models side by side (live CPU latency)",
                             interactive=False, wrap=True)
        gr.Markdown(f"<sub>Running on: {HW.get('cpu')} · {HW.get('logical_cores')} logical cores · "
                    f"torch {HW.get('torch')} · {HW.get('torch_threads')} threads · "
                    f"quantized engine `{HW.get('quantized_engine')}`</sub>")

        outputs = [decision, prep, table, probs]
        submit.click(analyze, [inp, threshold], outputs)
        inp.submit(analyze, [inp, threshold], outputs)
        threshold.release(analyze, [inp, threshold], outputs)
        clear.click(lambda: ("", "### Routing decision will appear here", "", None, None),
                    None, [inp, decision, prep, table, probs])
        for btn, txt in ex_buttons:
            btn.click(lambda t=txt: t, None, inp).then(analyze, [inp, threshold], outputs)
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="create a public Gradio link")
    args = ap.parse_args()
    if not ROUTERS:
        print("[app] WARNING: no trained models found - run the training pipeline first "
              "(see README).")
    build_ui().launch(server_name=args.host, server_port=args.port, share=args.share)
