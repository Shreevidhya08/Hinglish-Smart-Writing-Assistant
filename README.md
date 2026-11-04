# Hinglish Smart Writing Assistant

Type the way you actually text — Roman-script Hinglish — and get back clean native-script text, polished phrasing, and an English translation.

```
"mujhe kal office jaana hai par mera phone kharab ho gaya hai"
        │
        ▼  1. Transliteration (LLM + guardrails + lexicon)
"मुझे कल ऑफिस जाना है पर मेरा फोन खराब हो गया है"
        │
        ▼  2. Phrasing improvement (LLM, guarded)
"मुझे कल ऑफिस जाना है, लेकिन मेरा फोन खराब हो गया है।"
        │
        ▼  3. Translation (IndicTrans2)
"I have to go to the office tomorrow, but my phone is dead."
```

Everything lives in one notebook, `Hinglish_Smart_Writing_Assistant.ipynb`. It includes the pipeline, a small hand-written evaluation set, error analysis, and a Gradio web UI.

---

## Features

- **Three-stage pipeline:** transliterate → improve phrasing → translate to English. Each stage can be toggled independently.
- **LLM transliteration** with few-shot prompting, sentence-context awareness, and an option to keep English words in Latin script (`keep_english`).
- **Guardrails** that catch the LLM's typical failures:
  - Token-count and number checks against the input, with a stricter retry prompt if they fail.
  - A lexicon of high-frequency Hindi words that overrides the LLM for common function words (है, नहीं, में, कल…).
- **Guarded phrasing improvement**, with tone control (`neutral`, `formal`, `casual`). Rewrites are rejected and the original is kept if they change numbers, add or drop negation, or change length drastically (<0.5× or >2×).
- **IndicTrans2 translation** (Indic → English), batched, with beam search.
- **Optional IndicXlit engine** as a non-LLM transliteration baseline.
- **Evaluation harness:** CER, WER, word accuracy, exact match, BLEU, chrF, BERTScore, latency, and a leftover-Latin-characters check, plus a DEV/TEST split.
- **Gradio UI** with a shareable link.

## Supported languages

| Language | Few-shot examples | Lexicon | Evaluated |
|----------|:-:|:-:|:-:|
| Hindi    | ✅ | ✅ | ✅ |
| Marathi, Bengali, Tamil, Telugu, Kannada, Gujarati | — | — | — |

The non-Hindi languages are wired into the pipeline (the prompts and IndicTrans2 language codes work), but they have no few-shot examples, no lexicon, and no test data. Expect noticeably lower quality there until you add those.

## Models used

| Role | Model | Notes |
|------|-------|-------|
| Transliteration + phrasing | `Qwen/Qwen2.5-7B-Instruct` (4-bit) on GPU; `Qwen2.5-1.5B-Instruct` on CPU | Falls back to `Qwen2.5-3B-Instruct` if loading fails. `LLM_NAME` can be swapped for another HF causal LM. |
| Translation | `ai4bharat/indictrans2-indic-en-dist-200M` | Swap to `indictrans2-indic-en-1B` for higher quality. |
| Transliteration baseline (optional) | AI4Bharat IndicXlit | Not installed by default. |

## Requirements

- Python 3.10+
- **A GPU is strongly recommended.** The 7B model in 4-bit fits on a free Colab T4. CPU mode works with a 1.5B model but is slow and low quality; use it for smoke tests only.
- Internet access on first run to download models from Hugging Face.

## Quick start

### Google Colab (easiest)

1. Upload the notebook to Colab and select a **GPU runtime** (T4 is enough).
2. Run all cells top to bottom.
3. The last cell launches the Gradio app and prints a public `*.gradio.live` link (temporary, ~1 week).

### Local

```bash
pip install "transformers==4.46.3" "accelerate>=0.34" sentencepiece gradio jiwer \
            sacrebleu pandas IndicTransToolkit bitsandbytes bert-score datasets
jupyter notebook Hinglish_Smart_Writing_Assistant.ipynb
```

`bitsandbytes` 4-bit quantization requires an NVIDIA GPU with CUDA.

**Optional — IndicXlit engine:**

```bash
pip install ai4bharat-transliteration
```

If it is installed, "IndicXlit" appears as a second engine in the UI and in the evaluation configs.

## Usage

### Python

```python
native, improved, english, status = pipeline(
    "bhai assignment ka deadline kab hai, mujhe nahi pata",
    lang="Hindi",
    engine="LLM",          # or "IndicXlit"
    improve=True,          # run the phrasing step
    tone="neutral",        # "neutral" | "formal" | "casual"
    to_english=True,
    keep_english=False,    # True keeps English words in Latin script
    robust=True,           # guardrails + lexicon
)
```

`pipeline` returns `(native_script, improved_text, english, status)`. Disabled stages return an empty string, and failures return the error message in `status` instead of raising.

### Gradio UI

Section 8 of the notebook builds the web app. Choose a language and engine, toggle the stages, pick a tone, and press **Convert**. Output boxes show the native script, improved phrasing, English translation, and timing.

## Notebook layout

| Section | What it does |
|---------|--------------|
| 1. Install | Installs dependencies |
| 2. Config and imports | Device detection, model names, language-code map |
| 3. Load models | Loads the LLM (4-bit if GPU) and IndicTrans2 |
| 4. Core functions | Few-shots, lexicon, transliteration, phrasing, translation, `pipeline()` |
| 5. Test data | 60 hand-written Hinglish → Hindi → English triples, split 18 DEV / 42 TEST (seed 42) |
| 6. Evaluation | Compares pipeline configurations; writes `summary.csv` and per-config `results_*.csv` |
| 7. Error analysis | Worst cases, common word-level errors, phrasing drift, failures |
| 8. Gradio frontend | Interactive demo |

## Evaluation

Results on the 42-sentence TEST split (Hindi, Qwen2.5-7B-Instruct 4-bit, T4 GPU). Transliteration metrics are computed after normalizing nukta, chandrabindu/anusvara, and punctuation. Lower is better for CER and WER.

| Configuration | CER ↓ | WER ↓ | Word acc. | Exact match | BLEU | chrF | BERTScore | Latin left | Latency (s) |
|---|---|---|---|---|---|---|---|---|---|
| 1. LLM raw | 0.091 | 0.127 | 0.873 | 0.429 | 48.0 | 69.0 | 93.1 | 11.9% | 2.5 |
| 2. LLM + guardrails/lexicon | **0.076** | **0.108** | **0.892** | **0.500** | 45.3 | 67.6 | 92.9 | **7.1%** | 2.5 |
| 3. + phrasing improvement | 0.076 | 0.108 | 0.892 | 0.500 | 46.4 | 67.3 | 93.6 | 7.1% | 4.2 |

How to read this:

- **Guardrails + lexicon improve transliteration accuracy** (CER 0.091 → 0.076, exact match 43% → 50%) and reduce stray Latin characters in the output.
- **Translation scores (BLEU/chrF) dipped slightly** with guardrails. The improvement step changes the Hindi text, so its English output is compared against a single reference translation, which penalizes valid rewordings.
- **The phrasing step modified most sentences**, and its latency cost is about 1.7 s per sentence. Check its output before relying on it; see Limitations below.

Common transliteration errors seen in the analysis: dropped long vowels (बारिश → बरिश), the model translating instead of transliterating (टाइम → समय, हेल्प → मदद, टीचर → शिक्षक), and occasional garbled output such as stray Latin characters or broken Unicode on rare words.

> **Caveat:** the test set is small (42 sentences) with a single hand-written reference each, so treat the numbers as indicative, not as a benchmark. To avoid inflating scores, tune prompts, few-shots, and the lexicon against **DEV** only and report on **TEST**. The few-shot examples are deliberately different from the test sentences.

## Customizing

- **Add or improve Hindi words:** edit `LEXICON` (Section 4). Add only general words or words found in DEV, never from TEST.
- **Add few-shot examples:** edit `HI_SHOTS_DATA` with `(roman, native-with-loanwords, native-keeping-English)` triples. To support another language, add equivalent shots and handle that language in `transliterate_llm_raw`.
- **Use your own test set:** replace `DATA` with a CSV containing the columns `hinglish`, `ref_native`, `ref_en` (see the commented line in Section 5).
- **Change models:** edit `LLM_NAME` / `TRANS_NAME` in Section 2. For gated models, accept the licence on Hugging Face and run `huggingface_hub.login()` first.
- **Adjust tone and phrasing rules:** edit the system prompt and `reject` conditions in `improve_phrasing`.

## Limitations

- **Hindi-centric.** Other supported languages lack few-shots, a lexicon, and evaluation.
- **LLM transliteration is not deterministic in quality.** It can translate instead of transliterate, mangle rare words, or emit stray characters. Guardrails reduce but don't eliminate this. When the token-count check fails twice, the pipeline keeps the first output as-is.
- **The phrasing step can drift.** It blocks changes to numbers and negation, but it can't detect other meaning changes (tense, gender, politeness level, and so on). It also altered the output in the large majority of test sentences, so review it before using it in anything important.
- **Single-sentence, line-by-line processing.** Each line is handled independently, so there is no cross-sentence context.
- **Hindi-only guardrails.** The lexicon override applies only when `keep_english` is off.
- **Translation depends on upstream quality.** Transliteration or phrasing errors propagate into the English output.
- **Latency.** About 2.5 s per sentence without the phrasing step and about 4 s with it, on a T4.

## Notes

- `transformers` is pinned to `4.46.3` for IndicTrans2 compatibility. pip may print a `huggingface-hub` version warning in Colab; it did not affect the run.
- IndicTrans2 is loaded with `trust_remote_code=True`, which downloads and runs model code from Hugging Face. Pin a specific revision if you need reproducibility or want to audit the code.
- `demo.launch(share=True)` exposes the app through a temporary public URL. Remove `share=True` to keep it local.

## Acknowledgements

- [AI4Bharat](https://ai4bharat.iitm.ac.in/) for IndicTrans2, IndicTransToolkit and IndicXlit
- [Qwen](https://huggingface.co/Qwen) for the Qwen2.5 instruct models
- Hugging Face Transformers, Gradio, jiwer, sacreBLEU, and BERTScore

## License

Add a licence for your own code here. Check the individual licences of the models you use (Qwen2.5, IndicTrans2) before any commercial use.
