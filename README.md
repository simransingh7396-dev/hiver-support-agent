# Hiver AskPlayStation Support Agent

Turn messy Twitter support threads into a **working AI agent** that *classifies intent, drafts grounded replies, decides escalate* — and **prove it works**.

> Brand: **AskPlayStation** | 12,121 threads (avg 3.1 turns) | 7 intents | Pipeline `classify → retrieve → draft → escalate`

## Quickstart (15-min reproduce)

```bash
git clone https://github.com/simransingh7396-dev/hiver-support-agent.git
cd hiver-support-agent
pip install -r requirements.txt

# NOTE: data/resolved_memory.jsonl (6,447) + data/subsample_2k.jsonl + data/golden.jsonl (200)
# are already committed, so you can SKIP steps 1-2 for 15-min reproduce.
# Only run them if you want to rebuild from scratch (needs askplaystation_threads.jsonl):

# 1. (optional) Build resolved memory (uses askplaystation_threads.jsonl; ~30s)
# python src/build_memory.py
# -> data/resolved_memory.jsonl (6,447) + data/subsample_2k.jsonl

# 2. (optional) Build golden set (200 stratified, seed 42)
# python src/build_golden.py
# -> data/golden.jsonl + data/golden_sampling_note.md

# 3. Run harness without LLM (no API key, ~2 min) — THIS IS THE 15-MIN REPRO
python eval/harness.py --no-llm --judge-limit 20
# -> eval/results.json + eval/report.md
# headline: Intent acc 0.95 (10/200 hand-corrected), Escalation F1 0.943, Safe-to-send 0.0 (heuristic)

# 4. With LLM (needs .env GEMINI_API_KEY=...) — optional
python src/agent.py --query "@AskPlayStation Game keeps freezing error CE-34878-0"
python src/agent.py --query "@AskPlayStation I was banned please help"
# python eval/harness.py --judge-limit 5          # uses gemini-3.6-flash, 20/day free tier
```

Single query output:
```json
{
  "intent": "technical_error_bug",
  "confidence": 0.95,
  "escalate": false,
  "escalate_reason": "auto_handle_if_known_fix",
  "draft": "Hi there! Troubleshooting steps for CE-34878-0 are here: https://t.co/W7AhTUulnd ...",
  "retrieved": [{"thread_id": "2099451.0", "score": 0.76, "brand_resolution": "..."}]
}
```

## Repo Layout

```
data/taxonomy.yaml            # 7 intents with evidence + examples
data/resolved_memory.jsonl    # 6,447 grounded fix threads (289 resolved_confirmed + 6175 brand_with_fix)
data/golden.jsonl             # 200 stratified hand-proposed + human-review flag
data/subsample_2k.jsonl       # 2k random threads for fast eval
src/agent.py                  # classify (Gemini few-shot + keyword fallback) → retrieve → draft → escalate
src/retrieval.py              # TF-IDF 8000 ngrams (SBERT optional) over resolved_memory
src/build_memory.py           # filter resolved threads
src/build_golden.py           # stratified 28/intent sampler + sampling note
eval/harness.py               # metrics (acc, F1, escalation PR, retrieval) + LLM judge
eval/judge.py                 # 1-5 rubric: groundedness/actionability/tone/overall + safe_to_send
eval/human_judge_30.jsonl     # 30 human vs LLM scores (Pearson 0.804, kappa 0.659)
REPORT.md                     # 6-page report (framing, baselines, failures, misleading headline, next week, decision log)
```

## Taxonomy (7)

1. **Technical Error / Bug** — CE-*, crash, freeze → auto-handle if known fix
2. **Account Access & Security** — login/ban/hacked → escalate (security)
3. **Billing & Refunds** — charge/refund/subscription → escalate (money/policy)
4. **Purchase / Content Not Delivered** — bought DLC/code invalid → auto-handle triage first
5. **Network & Connectivity** — NAT/WiFi/PSN → auto-handle (power-cycle/DNS)
6. **Hardware Problem** — controller/disc/console → escalate (physical repair)
7. **General Inquiry / How-To** — informational → auto-handle

See `data/taxonomy.yaml:1` for keywords, examples, escalation notes.

## Evaluation

- **Baselines:** Trivial (majority `account_access_security` 15% acc) vs Simple (keyword) vs Agent (Gemini+retrieval). Full numbers in `REPORT.md:2`.
- **Golden:** `data/golden.jsonl` 200, `data/golden_sampling_note.md` explains method/limitations.
- **Harness:** `eval/harness.py` computes `accuracy, F1 macro/weighted, confusion_matrix, escalation prec/rec/F1, retrieval avg_top3, judge (groundedness/actionability/tone/overall)`.
- **Judge:** `eval/judge.py` LLM `gemini-3.6-flash` (fallback heuristic when quota 429/503). Human agreement: `eval/human_judge_30.jsonl` → Pearson 0.804, Cohen kappa 0.659 (3-way binned), within-1-point 93.3%.

## Headline (honest, after 10 hand-corrections)

`eval/results.json`: Intent acc **0.95** (190/200, 10 corrected), Escalation F1 **0.943**, Judge overall **2.0**, Safe **0.0** (heuristic fallback) — **see REPORT.md §5 "What is misleading"** for why 0.95 is still inflated (only 10/200 hand-fixed; fully hand-labelled would be lower) and why safe gate is strict. Human agreement: **Pearson 0.804, kappa 0.659 (n=30)**.

## Decisions

12 non-obvious decisions logged in `REPORT.md:6` (e.g., resolved = 289+6175, TF-IDF not SBERT by default, URL allowlist, force escalate phrases).

## Next Week

See `REPORT.md:5` (fix golden, intent-filtered SBERT+FAISS, multi-intent, calibration, live shadow).

## Env

- `twcs.csv` at `C:\Users\simra\OneDrive\Desktop\archive\twcs\twcs.csv` (not committed; subsample used for eval)
- `.env` → `GEMINI_API_KEY=...` (free tier 20/day on `gemini-3.6-flash`; harness degrades gracefully to keyword)
- Python 3.14, pandas, sklearn, google-genai, python-dotenv

## Submit

Form: https://intelligent-bar-256.notion.site/39492cbf0da2800682cfc78a600a745f — include repo link + REPORT.md.
