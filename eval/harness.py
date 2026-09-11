"""
Evaluation harness
- Runs agent on data/golden.jsonl
- Compares to gold: intent accuracy/F1, escalation precision/recall
- Baselines: trivial (majority class) + keyword (simple) vs agent (Gemini)
- LLM judge on 50 drafts, human agreement on 30
Run: python eval/harness.py [--limit 50] [--no-llm]
Outputs: eval/results.json + eval/report.md
"""
import json
import random
import argparse
from pathlib import Path
from collections import Counter, defaultdict
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

GOLDEN = ROOT / "data" / "golden.jsonl"
OUT_RESULTS = ROOT / "eval" / "results.json"
OUT_REPORT = ROOT / "eval" / "report.md"

def load_golden(path=GOLDEN):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items

def majority_baseline_predict(query, majority_intent="general_inquiry_howto"):
    return {"intent": majority_intent, "confidence": 0.5, "escalate": False, "escalate_reason": "trivial_baseline"}

def keyword_baseline_predict(query):
    from agent import keyword_classify, decide_escalation
    cls = keyword_classify(query)
    esc = decide_escalation(cls["intent"], cls["confidence"], query)
    return {"intent": cls["intent"], "confidence": cls["confidence"], "escalate": esc["escalate"], "escalate_reason": esc["reason"]}

def compute_metrics(golds, preds, label_key="gold_intent", pred_key="intent"):
    from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix
    y_true = [g[label_key] for g in golds]
    y_pred = [p[pred_key] for p in preds]
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    # per class
    p,r,f,_ = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    per_class = {lab: {"prec": round(float(p[i]),3), "rec": round(float(r[i]),3), "f1": round(float(f[i]),3)} for i,lab in enumerate(labels)}
    return {"accuracy": round(float(acc),3), "f1_macro": round(float(f1_macro),3), "f1_weighted": round(float(f1_weighted),3), "per_class": per_class, "labels": labels, "confusion_matrix": cm.tolist()}

def escalation_metrics(golds, preds):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    y_true = [1 if g["gold_should_escalate"] else 0 for g in golds]
    y_pred = [1 if p["escalate"] else 0 for p in preds]
    acc = accuracy_score(y_true, y_pred)
    p,r,f,_ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
    return {"accuracy": round(float(acc),3), "precision": round(float(p),3), "recall": round(float(r),3), "f1": round(float(f),3), "true_escalate_rate": round(sum(y_true)/len(y_true),3), "pred_escalate_rate": round(sum(y_pred)/len(y_pred),3)}

def run_agent_on_golden(golds, use_llm=True, limit=None):
    from agent import Agent
    agent = Agent(use_llm=use_llm)
    preds = []
    # limit for quick runs
    items = golds[:limit] if limit else golds
    for i,g in enumerate(items):
        q = g["query"]
        res = agent.handle(q)
        preds.append(res)
        if (i+1)%20==0:
            print(f"  agent progress {i+1}/{len(items)}")
    return preds

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="limit golden for quick run")
    parser.add_argument("--no-llm", action="store_true", help="use keyword baseline as agent (no gemini)")
    parser.add_argument("--judge-limit", type=int, default=50, help="how many drafts to judge")
    args = parser.parse_args()

    golds = load_golden()
    if args.limit:
        golds = golds[:args.limit]
    print(f"Loaded golden: {len(golds)}")

    # Find majority intent
    cnt = Counter([g["gold_intent"] for g in golds])
    majority = cnt.most_common(1)[0][0]
    print(f"Majority intent: {majority} ({cnt[majority]}/{len(golds)})")

    # Baselines
    trivial_preds = [majority_baseline_predict(g["query"], majority) for g in golds]
    keyword_preds = [keyword_baseline_predict(g["query"]) for g in golds]

    # Agent
    print(f"\nRunning agent (use_llm={not args.no_llm})...")
    agent_preds = run_agent_on_golden(golds, use_llm=not args.no_llm, limit=None)

    # Metrics
    print("\nComputing metrics...")
    trivial_intent = compute_metrics(golds, trivial_preds)
    keyword_intent = compute_metrics(golds, keyword_preds)
    agent_intent = compute_metrics(golds, agent_preds)

    trivial_esc = escalation_metrics(golds, trivial_preds)
    keyword_esc = escalation_metrics(golds, keyword_preds)
    agent_esc = escalation_metrics(golds, agent_preds)

    # Retrieval hit check: we can't know gold retrieval, so measure avg score + intent-consistency proxy
    # For now just report avg retrieval score
    avg_scores = []
    for p in agent_preds:
        if p.get("retrieved"):
            avg_scores.append(sum(r["score"] for r in p["retrieved"])/len(p["retrieved"]))
    retrieval_stats = {"avg_top3_score": round(float(sum(avg_scores)/len(avg_scores)),3) if avg_scores else 0, "num_with_retrieval": len(avg_scores)}

    # LLM Judge on subset
    print(f"\nRunning LLM judge on {args.judge_limit} drafts...")
    from eval.judge import get_client, judge_one
    client = get_client()
    # sample balanced
    random.seed(42)
    judge_indices = random.sample(range(len(golds)), min(args.judge_limit, len(golds)))
    judge_results = []
    for idx in judge_indices:
        g = golds[idx]
        p = agent_preds[idx]
        j = judge_one(g["query"], g["gold_intent"], g["gold_intent_name"], p["draft"], p.get("retrieved",[]), client=client)
        judge_results.append({"idx": idx, "query": g["query"][:120], "gold_intent": g["gold_intent"], "pred_intent": p["intent"], "draft": p["draft"], "judge": j})
        if len(judge_results)%10==0:
            print(f"  judged {len(judge_results)}/{len(judge_indices)}")

    # summarize judge
    if judge_results:
        avg_ground = sum(r["judge"]["groundedness"] for r in judge_results)/len(judge_results)
        avg_act = sum(r["judge"]["actionability"] for r in judge_results)/len(judge_results)
        avg_tone = sum(r["judge"]["tone"] for r in judge_results)/len(judge_results)
        avg_overall = sum(r["judge"]["overall"] for r in judge_results)/len(judge_results)
        safe_rate = sum(1 for r in judge_results if r["judge"].get("safe_to_send"))/len(judge_results)
        halluc_rate = sum(1 for r in judge_results if r["judge"].get("hallucinated_url"))/len(judge_results)
        judge_summary = {"avg_groundedness": round(avg_ground,2), "avg_actionability": round(avg_act,2), "avg_tone": round(avg_tone,2), "avg_overall": round(avg_overall,2), "safe_to_send_rate": round(safe_rate,3), "hallucinated_url_rate": round(halluc_rate,3), "n": len(judge_results)}
    else:
        judge_summary = {}

    # Human agreement: simulate 30 hand labels vs LLM judge
    # For real submission, you hand-score 30 drafts on same rubric, then we compute agreement
    # Here we create placeholder: we treat gold intent correctness as human proxy for quick demo
    # Provide code to compute later when human_scores.json exists
    human_path = ROOT / "eval" / "human_judge_30.jsonl"
    agreement = {}
    if human_path.exists():
        from eval.judge import compute_agreement
        llm_scores = []
        human_scores = []
        with open(human_path, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                llm_scores.append(d["llm_overall"])
                human_scores.append(d["human_overall"])
        agreement = compute_agreement(llm_scores, human_scores)
        print(f"Human agreement computed: {agreement}")
    else:
        agreement = {"note": "Create eval/human_judge_30.jsonl with {llm_overall, human_overall} to compute kappa. See eval/README.",
                     "required_n": 30, "status": "pending_human_labels"}

    results = {
        "golden_size": len(golds),
        "majority_intent": majority,
        "intent_metrics": {
            "trivial_majority": trivial_intent,
            "keyword_simple": keyword_intent,
            "agent": agent_intent
        },
        "escalation_metrics": {
            "trivial": trivial_esc,
            "keyword": keyword_esc,
            "agent": agent_esc
        },
        "retrieval": retrieval_stats,
        "judge_summary": judge_summary,
        "judge_details_sample": judge_results[:5],
        "human_agreement": agreement,
        "failure_examples": judge_results[:3]  # placeholder, real analysis in report
    }

    OUT_RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_RESULTS, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved results -> {OUT_RESULTS}")

    # Write mini report
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write(f"# Eval Report (auto-generated)\n\n")
        f.write(f"Golden: {len(golds)} | Agent LLM: {not args.no_llm}\n\n")
        f.write(f"## Intent Classification\n")
        f.write(f"- Trivial (majority {majority}): acc {trivial_intent['accuracy']} f1_macro {trivial_intent['f1_macro']}\n")
        f.write(f"- Keyword simple: acc {keyword_intent['accuracy']} f1_macro {keyword_intent['f1_macro']}\n")
        f.write(f"- Agent: acc {agent_intent['accuracy']} f1_macro {agent_intent['f1_macro']}\n\n")
        f.write(f"## Escalation\n")
        f.write(f"- Trivial: {trivial_esc}\n")
        f.write(f"- Keyword: {keyword_esc}\n")
        f.write(f"- Agent: {agent_esc}\n\n")
        f.write(f"## Retrieval\n{retrieval_stats}\n\n")
        f.write(f"## Judge (n={judge_summary.get('n',0)})\n{json.dumps(judge_summary, indent=2)}\n\n")
        f.write(f"## Human Agreement\n{json.dumps(agreement, indent=2)}\n")
    print(f"Saved report -> {OUT_REPORT}")

    # Print headline
    print("\n" + "="*60)
    print(f"Headline: Intent accuracy {agent_intent['accuracy']} | Escalation F1 {agent_esc['f1']} | Safe-to-send {judge_summary.get('safe_to_send_rate',0)}")
    print("="*60)
    return results

if __name__ == "__main__":
    run()
