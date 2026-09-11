"""
AskPlayStation AI Support Agent
Pipeline: classify -> retrieve -> draft -> escalate

Usage:
  python src/agent.py --query "@AskPlayStation Game keeps freezing error CE-34878-0"
  python src/agent.py --eval data/golden.jsonl --out eval/preds.jsonl

Design decisions (see Decision Log in report):
- 7 intents locked in data/taxonomy.yaml (evidence-based)
- Retrieval grounded in resolved_memory (6175 brand_with_fix + 289 resolved)
- Draft must not hallucinate URLs; only reuse URLs from retrieved brand messages
- Escalation: security/money/hardware always escalate per taxonomy; else rule+confidence
"""
import argparse
import json
import re
import os
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parent.parent
TAXONOMY_PATH = ROOT / "data" / "taxonomy.yaml"

# --- Intent definitions (mirrors taxonomy.yaml) ---
INTENTS = {
    "technical_error_bug": {
        "name": "Technical Error / Bug",
        "escalate": False,
        "reason": "auto_handle_if_known_fix",
        "keywords": ["error","code","ce-","nw-","wc-","e-","crash","freeze","freezing","corrupt","update","bug"]
    },
    "account_access_security": {
        "name": "Account Access & Security",
        "escalate": True,
        "reason": "security_sensitive - banned/hacked/login needs human verification",
        "keywords": ["login","log in","sign in","password","2-step","verification","banned","ban","hacked","stolen","suspended","account"]
    },
    "billing_refunds": {
        "name": "Billing & Refunds",
        "escalate": True,
        "reason": "money_sensitive_policy_bound - refunds require human approval",
        "keywords": ["refund","charge","charged","money","wallet","billing","payment","subscription","plus","auto-renew"]
    },
    "purchase_content_not_delivered": {
        "name": "Purchase / Content Not Delivered",
        "escalate": False,
        "reason": "auto_handle_first - try restore licenses/download queue first",
        "keywords": ["bought","purchase","download","dlc","code","redeem","pre-order","store","psn","not downloading","invalid"]
    },
    "network_connectivity": {
        "name": "Network & Connectivity",
        "escalate": False,
        "reason": "scripted_fix_reusable - power cycle/DNS fixes seen in data",
        "keywords": ["network","wifi","lan","nat","connect","connection","online","server","psn","timeout"]
    },
    "hardware_problem": {
        "name": "Hardware Problem",
        "escalate": True,
        "reason": "needs_physical_repair - controller/disc/console hardware",
        "keywords": ["controller","disc","disk","console","hdmi","turn on","won't turn","sync","humming","overheat"]
    },
    "general_inquiry_howto": {
        "name": "General Inquiry / How-To",
        "escalate": False,
        "reason": "informational_no_risk",
        "keywords": ["how","can i","do i need","what is","when","is there a way","guide","info"]
    }
}

# Escalation overrides: even if intent is auto_handle, these phrases force escalate
FORCE_ESCALATE_PHRASES = ["hacked","stolen","banned","suspend","refund","charge","payment not","fraud"]

def keyword_classify(text: str) -> Dict:
    """Simple keyword baseline + confidence (used as fallback and for fast eval)."""
    t = text.lower()
    scores = {}
    for intent, meta in INTENTS.items():
        kw = meta["keywords"]
        # count keyword hits weighted by length (error codes stronger)
        hits = sum(1 for k in kw if k in t)
        # bonus for error code pattern
        if intent == "technical_error_bug" and re.search(r"\b(ce|nw|wc|e)-?\d{4,}", t):
            hits += 2
        scores[intent] = hits
    # pick max; tie -> general
    best = max(scores, key=lambda k: scores[k])
    max_score = scores[best]
    # confidence heuristic: normalized by keywords
    total = sum(scores.values()) + 1e-6
    conf = max_score / (max(1, total)) if max_score>0 else 0.35
    # if no hits -> general
    if max_score == 0:
        best = "general_inquiry_howto"
        conf = 0.4
    return {"intent": best, "confidence": round(min(0.95, max(0.35, conf + 0.3)),2), "scores": scores}

def llm_classify(text: str, client=None, model: str = "gemini-3.6-flash") -> Dict:
    """Gemini few-shot classifier. Falls back to keyword if no client/key."""
    if client is None:
        return keyword_classify(text)
    # Build prompt from taxonomy
    intent_list = "\n".join([f"- {k}: {v['name']} - {v['reason']}" for k,v in INTENTS.items()])
    examples = """
Example 1: "Error CE-34878-0 game keeps crashing" -> technical_error_bug (0.92)
Example 2: "I think my account is hacked" -> account_access_security (0.95)
Example 3: "Why was I charged £49.99 twice" -> billing_refunds (0.93)
Example 4: "Bought DLC but not downloading code invalid" -> purchase_content_not_delivered (0.88)
Example 5: "Cannot connect to server NAT type 3" -> network_connectivity (0.90)
Example 6: "Controller won't sync blue light stays on" -> hardware_problem (0.89)
Example 7: "What's the free PS Plus game this month?" -> general_inquiry_howto (0.85)
"""
    prompt = f"""You are an intent classifier for AskPlayStation support. Classify into exactly one of:

{intent_list}

{examples}

Rules:
- Return JSON only: {{"intent": "<id>", "confidence": 0.0-1.0, "reason": "short phrase"}}
- intent must be one of the 7 ids above
- confidence high if clear error code / keywords, low if ambiguous

Message: "{text}"
"""
    try:
        resp = client.models.generate_content(model=model, contents=prompt)
        raw = resp.text.strip()
        # extract json
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            data = json.loads(m.group(0))
            if data.get("intent") in INTENTS:
                return {"intent": data["intent"], "confidence": float(data.get("confidence", 0.75)), "reason": data.get("reason","llm")}
    except Exception as e:
        print(f"LLM classify failed: {e}")
    return keyword_classify(text)

def decide_escalation(intent: str, confidence: float, text: str) -> Dict:
    meta = INTENTS[intent]
    t = text.lower()
    # hard rules
    if any(p in t for p in FORCE_ESCALATE_PHRASES):
        # but allow network/billing already escalate
        if intent in ("billing_refunds","account_access_security","hardware_problem"):
            return {"escalate": True, "reason": f"{meta['reason']} + trigger phrase"}
    if meta["escalate"]:
        return {"escalate": True, "reason": meta["reason"]}
    # auto_handle intents: escalate only if low confidence
    if confidence < 0.55:
        return {"escalate": True, "reason": f"low_confidence ({confidence}) - route to human"}
    # unresolved signal
    if any(w in t for w in ["still not","still doesn't","didn't work","not working","useless"]):
        return {"escalate": True, "reason": "unresolved_signal - customer says prior fix failed"}
    return {"escalate": False, "reason": meta["reason"]}

def draft_reply(query: str, intent: str, retrieved: List[Dict], client=None, model="gemini-3.6-flash") -> str:
    """Grounded draft: must reuse phrasing/steps from retrieved brand_resolution, no hallucinated URLs."""
    # collect allowed URLs from retrieved
    allowed_urls = []
    for r in retrieved:
        allowed_urls += re.findall(r"https://\S+", r.get("brand_resolution",""))
    allowed_urls = list(dict.fromkeys(allowed_urls))[:3]

    # Build context
    context = "\n".join([f"- Hist fix {i+1}: {r['brand_resolution'][:220]}" for i,r in enumerate(retrieved[:3])])
    intent_name = INTENTS[intent]["name"]

    if client is None:
        # template fallback (grounded, no LLM)
        primary = retrieved[0]["brand_resolution"] if retrieved else "Please try power cycling and check our help article."
        # keep full first resolution sentence, clean mentions
        draft = re.sub(r"@\w+", "", primary).strip()
        draft = re.sub(r"\s+", " ", draft)
        if not draft.endswith("."):
            draft += "."
        if allowed_urls:
            draft += f" More info: {allowed_urls[0]}"
        if not any(x in draft.lower() for x in ["let us know","dm","contact"]):
            draft += f" If this doesn't help, let us know - we'll escalate."
        return draft[:500]

    prompt = f"""You are AskPlayStation support. Draft a helpful reply.

Intent: {intent_name}
Customer: "{query}"

Historical brand fixes (ground your reply in these, reuse steps/phrasing):
{context}

Allowed URLs (you may ONLY use these URLs, or no URL): {allowed_urls if allowed_urls else "none - do not invent links"}

Rules:
- Be concise (2-3 sentences), empathetic, actionable
- Reuse troubleshooting steps verbatim where possible
- Do NOT invent error codes, prices, or links
- If steps are generic (power cycle, restore licenses), keep them
- End with clear next step (try X, or DM if account-specific)

Reply:"""
    try:
        resp = client.models.generate_content(model=model, contents=prompt)
        text = resp.text.strip().replace("\n"," ")
        # post-filter: remove any URL not in allowed list
        found_urls = re.findall(r"https://\S+", text)
        for url in found_urls:
            if url not in allowed_urls:
                text = text.replace(url, "")
        return text[:600].strip()
    except Exception as e:
        print(f"LLM draft failed: {e}")
        return retrieved[0]["brand_resolution"][:400] if retrieved else "Thanks for reaching out. Could you DM us your PSN ID so we can help?"

class Agent:
    def __init__(self, use_llm: bool = True, use_sbert: bool = False):
        self.use_llm = use_llm
        self.client = None
        if use_llm:
            try:
                from google import genai
                from dotenv import load_dotenv
                load_dotenv()
                api_key = os.getenv("GEMINI_API_KEY")
                if api_key:
                    self.client = genai.Client(api_key=api_key)
                    print("Gemini client initialized")
                else:
                    print("No GEMINI_API_KEY - using keyword/template fallback")
            except Exception as e:
                print(f"Gemini init failed: {e} - fallback")
        # lazy retriever
        try:
            from src.retrieval import Retriever
        except ModuleNotFoundError:
            from retrieval import Retriever
        self.retriever = Retriever(use_sbert=use_sbert)

    def handle(self, query: str) -> Dict:
        # 1 classify
        cls = llm_classify(query, client=self.client) if self.use_llm else keyword_classify(query)
        intent = cls["intent"]
        conf = cls["confidence"]
        # 2 retrieve
        retrieved = self.retriever.retrieve(query, k=3)
        # 3 escalate
        esc = decide_escalation(intent, conf, query)
        # 4 draft
        draft = draft_reply(query, intent, retrieved, client=self.client)
        return {
            "query": query,
            "intent": intent,
            "intent_name": INTENTS[intent]["name"],
            "confidence": conf,
            "escalate": esc["escalate"],
            "escalate_reason": esc["reason"],
            "draft": draft,
            "retrieved": [{"thread_id": r["thread_id"], "score": r["score"], "brand_resolution": r["brand_resolution"][:200]} for r in retrieved]
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, help="single query")
    parser.add_argument("--eval", type=str, help="golden jsonl path")
    parser.add_argument("--out", type=str, help="preds out")
    parser.add_argument("--no-llm", action="store_true", help="disable gemini, use keyword baseline")
    args = parser.parse_args()

    agent = Agent(use_llm=not args.no_llm)

    if args.query:
        res = agent.handle(args.query)
        print(json.dumps(res, indent=2, ensure_ascii=False))
    elif args.eval:
        import pathlib
        in_path = Path(args.eval)
        out_path = Path(args.out) if args.out else Path("eval/preds.jsonl")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        n=0
        with open(in_path, "r", encoding="utf-8") as fin, open(out_path, "w", encoding="utf-8") as fout:
            for line in fin:
                if not line.strip(): continue
                ex = json.loads(line)
                q = ex.get("query") or ex.get("text") or ex.get("customer_msg") or ""
                pred = agent.handle(q)
                # keep gold + pred
                out = {"gold": ex, "pred": pred}
                fout.write(json.dumps(out, ensure_ascii=False)+"\n")
                n+=1
                if n%20==0: print(f"processed {n}")
        print(f"Saved {n} -> {out_path}")
    else:
        # demo
        for q in [
            "@AskPlayStation Game keeps freezing and crashing error CE-34878-0",
            "@AskPlayStation I was banned for no reason please help",
            "@AskPlayStation I bought FIFA points but didn't get them, code says invalid"
        ]:
            print("\n==>", q)
            print(json.dumps(agent.handle(q), indent=2, ensure_ascii=False))
