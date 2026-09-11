# Eval Report (auto-generated)

Golden: 200 | Agent LLM: False

## Intent Classification
- Trivial (majority general_inquiry_howto): acc 0.165 f1_macro 0.04
- Keyword simple: acc 0.95 f1_macro 0.95
- Agent: acc 0.95 f1_macro 0.95

## Escalation
- Trivial: {'accuracy': 0.575, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'true_escalate_rate': 0.425, 'pred_escalate_rate': 0.0}
- Keyword: {'accuracy': 0.95, 'precision': 0.921, 'recall': 0.965, 'f1': 0.943, 'true_escalate_rate': 0.425, 'pred_escalate_rate': 0.445}
- Agent: {'accuracy': 0.95, 'precision': 0.921, 'recall': 0.965, 'f1': 0.943, 'true_escalate_rate': 0.425, 'pred_escalate_rate': 0.445}

## Retrieval
{'avg_top3_score': 0.389, 'num_with_retrieval': 200}

## Judge (n=5)
{
  "avg_groundedness": 2.0,
  "avg_actionability": 4.0,
  "avg_tone": 4.0,
  "avg_overall": 2.0,
  "safe_to_send_rate": 0.0,
  "hallucinated_url_rate": 1.0,
  "n": 5
}

## Human Agreement
{
  "pearson_r": 0.804,
  "cohen_kappa": 0.659,
  "n": 30
}
