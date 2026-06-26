import pickle, torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

m = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
print("FinBERT id2label:", m.config.id2label)

tok = AutoTokenizer.from_pretrained("ProsusAI/finbert")
recs = pickle.load(open("figures/soft_records.pkl", "rb"))

n_correct = 0
for r in recs:
    enc = tok(r["text"], return_tensors="pt", truncation=True, padding="max_length", max_length=32)
    with torch.no_grad():
        logits = m(**enc).logits[0]
    pred = int(logits.argmax().item())
    if pred == r["true_label"]:
        n_correct += 1
print(f"FinBERT accuracy on sample: {n_correct}/{len(recs)} = {n_correct/len(recs):.1%}")