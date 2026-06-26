"""
GPT-4o-based paraphrase generator with semantic-similarity filtering.

Reconstruction of the attack methodology of Can Türetken & Leippold (2026):
- Paraphrases produced by GPT-4o under structured prompting.
- Each candidate filtered by cosine similarity in a Sentence-BERT proxy
  embedding to enforce semantic coherence.
- Diversity encouraged by temperature and explicit prompt instructions.

This is NOT a literal port of their attack -- the exact prompts and loss
weighting from the paper are not public. The generator preserves the
qualitative regime (targeted, LLM-generated, semantically constrained)
which is what the soft-token vs cloud-chart comparison in Section 8.3
needs.

Requires:
    pip install openai sentence-transformers
    export OPENAI_API_KEY=sk-...

Cost estimate (May 2026 GPT-4o pricing): ~$0.001-0.002 per input at
K=128 paraphrases per call (using batched output in a single completion).
For n=200 inputs, total cost is typically under $1.
"""

import json
import os
import re
import time
from typing import List, Optional

import numpy as np
from openai import OpenAI


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

PARAPHRASE_PROMPT_TEMPLATE = """You are a linguistic assistant generating paraphrases of a financial sentence for an academic study on the robustness of sentiment-classification models.

Generate exactly {K} distinct paraphrases of the sentence below. Each paraphrase must:
1. Preserve the original meaning and sentiment exactly.
2. Use different wording or structure from the original (synonyms, reordering, voice changes, alternative constructions).
3. Be grammatically correct and read naturally as financial English.
4. Differ from every other paraphrase in the list.

Output only a JSON object with one key "paraphrases" whose value is a JSON array of exactly {K} paraphrase strings. No preamble, no commentary, no other keys.

ORIGINAL SENTENCE:
{sentence}"""


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class GPT4oTuretkenLeippoldParaphraser:
    """
    GPT-4o paraphraser with proxy-embedding similarity filtering.

    Usage:
        gen = GPT4oTuretkenLeippoldParaphraser(
            proxy_model_name="sentence-transformers/all-MiniLM-L6-v2",
            similarity_threshold=0.80,
        )
        paraphrases = gen.paraphrase("The company posted strong earnings.", K=128)
    """

    def __init__(
        self,
        model: str = "gpt-4o",
        temperature: float = 1.0,
        similarity_threshold: float = 0.80,
        proxy_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_retries: int = 3,
        api_key: Optional[str] = None,
    ):
        self.client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        self.model = model
        self.temperature = temperature
        self.similarity_threshold = similarity_threshold
        self.max_retries = max_retries

        # Lazy import: only loaded if we're actually generating.
        from sentence_transformers import SentenceTransformer
        self.proxy = SentenceTransformer(proxy_model_name)

    # -----------------------------------------------------------------------
    # Single-input paraphrase
    # -----------------------------------------------------------------------

    def paraphrase(self, text: str, K: int = 128) -> List[str]:
        """
        Generate up to K paraphrases of `text`, filtered by proxy similarity.

        Returns a list of accepted paraphrases (length <= K). Each is a
        non-empty string distinct from `text`.
        """
        # GPT-4o practically returns at most ~25-50 paraphrases per call.
        # Ask for slightly more than K but never more than 40.
        n_candidates = min(int(K * 1.4), 40)
        raw = self._call_gpt(text, n_candidates)

        # Filter: remove empty, dedupe, drop the original.
        cleaned = []
        seen = {text.strip().lower()}
        for p in raw:
            p = p.strip()
            if not p:
                continue
            key = p.lower()
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(p)

        # Score with proxy similarity and keep above threshold.
        if not cleaned:
            return []

        emb_orig = self.proxy.encode(
            [text], normalize_embeddings=True, show_progress_bar=False,
        )[0]
        emb_par = self.proxy.encode(
            cleaned, normalize_embeddings=True, show_progress_bar=False,
        )
        sims = emb_par @ emb_orig  # both already L2-normalised

        accepted = [
            (p, float(s)) for p, s in zip(cleaned, sims)
            if s >= self.similarity_threshold
        ]

        # Sort by descending similarity to bias toward semantically tight
        # paraphrases (closer to the proxy ball at small eta).
        accepted.sort(key=lambda ps: -ps[1])
        return [p for p, _ in accepted[:K]]

    # -----------------------------------------------------------------------
    # GPT-4o call with retries
    # -----------------------------------------------------------------------

    def _call_gpt(self, text: str, K: int) -> List[str]:
        prompt = PARAPHRASE_PROMPT_TEMPLATE.format(K=K, sentence=text)

        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    response_format={"type": "json_object"},
                )
                if not response.choices:
                    raise RuntimeError("API returned no choices")
                content = response.choices[0].message.content
                if content is None:
                    finish = response.choices[0].finish_reason
                    refusal = getattr(response.choices[0].message, "refusal", None)
                    raise RuntimeError(
                        f"API returned None content (finish_reason={finish!r}, "
                        f"refusal={refusal!r})"
                    )
                return self._parse_json_array(content, K)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    print(f"  GPT-4o call failed after {self.max_retries} retries: {type(e).__name__}: {e}")
                    return []
                time.sleep(2 ** attempt)
        return []

    # -----------------------------------------------------------------------
    # JSON parsing (lenient -- GPT-4o sometimes wraps the array in an object)
    # -----------------------------------------------------------------------

    @staticmethod
    def _parse_json_array(content: str, K: int) -> List[str]:
        """Robustly extract a list of paraphrase strings from GPT output."""
        try:
            obj = json.loads(content)
        except json.JSONDecodeError:
            # Fallback: regex-extract a JSON array.
            m = re.search(r"\[.*\]", content, re.DOTALL)
            if not m:
                return []
            try:
                obj = json.loads(m.group(0))
            except json.JSONDecodeError:
                return []

        # The model may return {"paraphrases": [...]} or just [...].
        if isinstance(obj, dict):
            # Refusals come back as {"error": "...", "...": "..."} -- skip them.
            if "error" in obj or "refusal" in obj:
                return []
            for v in obj.values():
                if isinstance(v, list) and v and isinstance(v[0], str):
                    return v[:K]
            return []
        if isinstance(obj, list):
            return [p for p in obj if isinstance(p, str)][:K]
        return []

    # -----------------------------------------------------------------------
    # Batch generation with caching to a JSON file
    # -----------------------------------------------------------------------

    def paraphrase_batch_to_file(
        self,
        sentences: List[str],
        K: int,
        output_path: str,
        resume: bool = True,
    ) -> dict:
        """
        Generate paraphrases for a list of sentences and save incrementally
        to `output_path` as JSON. Safe to interrupt and resume.

        Output format: {sentence: [paraphrase, ...], ...}

        If `resume=True` and `output_path` exists, sentences already present
        are skipped.
        """
        table = {}
        if resume and os.path.exists(output_path):
            with open(output_path) as f:
                table = json.load(f)
            print(f"  Resuming: {len(table)} sentences already paraphrased.")

        todo = [s for s in sentences if s not in table]
        print(f"  Generating paraphrases for {len(todo)} sentences "
              f"(K={K} each, threshold={self.similarity_threshold}).")

        for i, text in enumerate(todo):
            paraphrases = self.paraphrase(text, K=K)
            table[text] = paraphrases
            if (i + 1) % 5 == 0 or (i + 1) == len(todo):
                with open(output_path, "w") as f:
                    json.dump(table, f, indent=1, ensure_ascii=False)
                print(f"  [{i+1}/{len(todo)}] {len(paraphrases)} kept | "
                      f"saved → {output_path}")

        return table
