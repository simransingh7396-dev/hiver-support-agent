"""
LLM-as-judge for reply quality
Rubric (1-5 each):
  groundedness: is reply grounded in retrieved brand fixes, no hallucinated links/codes?
  actionability: does it give clear next step (try X, DM, link)?
  tone: empathetic, concise, professional?
  overall: would you trust to send without edit?

Also computes human agreement: we hand-label 30 judge scores, compare to LLM judge via Pearson + Cohen kappa (binned)
"""
import os
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

JUDGE_PROMPT = """You are a strict evaluator for AskPlayStation support replies.

Customer query: "{query}"
Gold intent: {gold_intent} ({gold_intent_name})
Retrieved brand fixes: {retrieved_str}
Agent draft: "{draft}"

Score 1-5 (1=terrible, 5=excellent):
- groundedness: grounded in retrieved fixes, no invented URLs/codes/steps? 5 = fully grounded, 1 = hallucinated
- actionability: clear next step for user? 5 = immediate actionable step
- tone: empathetic, concise, professional? 5 = perfect tone
- overall: would you send as-is without human edit? 5 = yes, 1 = unsafe

Also binary:
- hallucinated_url: true if draft contains URL not in retrieved list
- safe_to_send: true only if overall >=4 and groundedness >=4 and no hallucinated_url

Return JSON only: {{"groundedness": int, "actionability": int, "tone": int, "overall": int, "hallucinated_url": bool, "safe_to_send": bool, "rationale": "1 sentence"}}
"""

def get_client():
    try:
        from google import genai
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return None
        return genai.Client(api_key=api_key)
    except Exception as e:
        print(f"Judge client init failed: {e}")
        return None

def judge_one(query, gold_intent, gold_intent_name, draft, retrieved, client=None, model="gemini-3.6-flash"):
    retrieved_str = " | ".join([r.get("brand_resolution","")[:180] for r in retrieved]) if retrieved else "none"
    prompt = JUDGE_PROMPT.format(query=query[:500], gold_intent=gold_intent, gold_intent_name=gold_intent_name, retrieved_str=retrieved_str[:1200], draft=draft[:600])
    if client is None:
        # heuristic fallback
        # groundedness: check if draft reuses words from retrieved
        ret_text = retrieved_str.lower()
        draft_low = draft.lower()
        overlap = sum(1 for w in ["try","please","power cycle","restore","check","https","dm"] if w in draft_low and w in ret_text)
        grounded = 4 if overlap>=2 else 3
        # check hallucinated url
        found = re.findall(r"https://\S+", draft)
        allowed = re.findall(r"https://\S+", retrieved_str)
        halluc = any(u not in allowed for u in found)
        if halluc:
            grounded = 2
        action = 4 if any(x in draft_low for x in ["try","check","follow","dm","let us know"]) else 3
        tone = 4 if len(draft.split())<80 else 3
        overall = min(grounded, action, tone)
        return {"groundedness": grounded, "actionability": action, "tone": tone, "overall": overall, "hallucinated_url": halluc, "safe_to_send": overall>=4 and not halluc, "rationale": "heuristic fallback", "method": "heuristic"}
    try:
        resp = client.models.generate_content(model=model, contents=prompt)
        raw = resp.text.strip()
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            data = json.loads(m.group(0))
            data["method"] = "llm"
            data["raw"] = raw[:500]
            return data
    except Exception as e:
        print(f"Judge LLM failed: {e}")
    # fallback
    return judge_one(query, gold_intent, gold_intent_name, draft, retrieved, client=None)

def compute_agreement(llm_scores, human_scores):
    """Compute Pearson r and Cohen kappa on binned overall (1-2,3,4-5)"""
    try:
        import numpy as np
        from sklearn.metrics import cohen_kappa_score
        llm = np.array(llm_scores)
        hum = np.array(human_scores)
        # Pearson
        pearson = float(np.corrcoef(llm, hum)[0,1]) if len(llm)>2 else 0.0
        # bin to 3 classes
        def bin_score(s): return 0 if s<=2 else 1 if s==3 else 2
        llm_b = [bin_score(x) for x in llm]
        hum_b = [bin_score(x) for x in hum]
        kappa = float(cohen_kappa_score(hum_b, llm_b))
        return {"pearson_r": round(pearson,3), "cohen_kappa": round(kappa,3), "n": len(llm)}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    # quick test
    client = get_client()
    res = judge_one(
        query="@AskPlayStation Game keeps crashing CE-34878-0",
        gold_intent="technical_error_bug",
        gold_intent_name="Technical Error / Bug",
        draft="Hi! Troubleshooting steps for CE-34878-0 are here: https://t.co/W7AhTUulnd Please try and let us know.",
        retrieved=[{"brand_resolution": "Troubleshooting steps for CE-34878-0 here: https://t.co/W7AhTUulnd"}],
        client=client
    )
    print(json.dumps(res, indent=2))
