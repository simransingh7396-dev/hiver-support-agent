# Golden Eval Set Sampling Note

**Size:** 200 examples (target 150-250, we use 200)
**Source:** `subsample_2k.jsonl` -> stratified sample of 12121 AskPlayStation threads
**Method:**
1. Ran keyword classifier (`src/agent.py:keyword_classify`) on first customer message of each thread to bucket into 7 intents (counts shown in build log).
2. Sampled ~28 per intent (random, seed 42) to ensure coverage of rare intents (billing, hardware, ban). Stratified by thread length (2/3/4+ turns) to avoid only short deflections.
3. Filled to 200 randomly, shuffled, truncated.
4. Proposed `gold_intent` + `gold_should_escalate` via taxonomy defaults (security/money/hardware = escalate) + heuristics (hacked/refund/banned overrides).
5. **Human labelling (done):** Annotator opened `data/golden.jsonl` and verified 200 queries vs `data/taxonomy.yaml` definitions (2 hrs). 10 mismatches found and corrected (e.g. `1606626.0` refund was technical→billing, `910998.0` billing→general, `1500436.0` purchase→network, etc. — see `git log` `fix_golden`). Those 10 now have `human_corrected:true, needs_human_review:false`; remaining 190 verified correct and marked `needs_human_review:false`. Second pass on 30 for agreement → `eval/human_judge_30.jsonl`.

**Limitations:**
- Keyword pre-bucket biases toward lexical intents; true distribution in wild is ~30% network, 20% technical, 15% account, etc. but golden is balanced for eval power, not prevalence.
- Single annotator (you) -> risk of bias; mitigated by 30-sample double-label for kappa (see eval).
- Only first customer message used as query; multi-turn context ignored (matches real incoming message task).

**Schema per line:** `thread_id, query, gold_intent, gold_intent_name, gold_should_escalate, gold_escalate_reason, needs_human_review`
