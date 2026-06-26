"""
Pre-generate paraphrases for the §8.3 experiment, saving to JSON.

Run this once before `main.py --run_cloud`. It generates GPT-4o paraphrases
for the same n sentences that `main.py` will sample, filters by proxy
similarity, and saves to `figures/paraphrases.json`.

Usage:
    export OPENAI_API_KEY=sk-...
    python generate_paraphrases.py --n 200 --K 128 --seed 42

Then run the cloud-chart pipeline with the cached paraphrases:
    python main.py --n 200 --q 32 --max_len 32 --K 128 \\
                   --run_soft_token --run_cloud

"""

import argparse
import json
import os

import numpy as np

# Reuse the same dataset loader and label-remap logic from main.py so the
# pre-generated paraphrases line up exactly with the sentences main.py
# will sample.
from main import load_financial_phrasebank_train_split
from paraphrase_gpt4o import GPT4oTuretkenLeippoldParaphraser


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--K", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--similarity_threshold", type=float, default=0.80)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--output", type=str, default="figures/paraphrases.json")
    p.add_argument("--model", type=str, default="gpt-4o")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    print(f"Loading Financial PhraseBank (sentences_allagree)...")
    ds = load_financial_phrasebank_train_split()
    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(ds), size=min(args.n, len(ds)), replace=False)
    sentences = [ds[int(i)]["sentence"] for i in idx]
    print(f"Selected n = {len(sentences)} sentences.")

    print(f"Initialising GPT-4o paraphraser (model={args.model}, "
          f"temp={args.temperature}, threshold={args.similarity_threshold})...")
    gen = GPT4oTuretkenLeippoldParaphraser(
        model=args.model,
        temperature=args.temperature,
        similarity_threshold=args.similarity_threshold,
    )

    print(f"Generating paraphrases → {args.output}")
    table = gen.paraphrase_batch_to_file(
        sentences=sentences,
        K=args.K,
        output_path=args.output,
        resume=True,
    )

    kept_counts = [len(v) for v in table.values()]
    print(f"\nDone. Wrote {len(table)} sentence → paraphrase entries.")
    print(f"  Mean kept: {np.mean(kept_counts):.1f}")
    print(f"  Median kept: {int(np.median(kept_counts))}")
    print(f"  Min/max kept: {min(kept_counts)}/{max(kept_counts)}")


if __name__ == "__main__":
    main()
