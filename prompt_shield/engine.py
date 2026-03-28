"""
prompt_shield/engine.py — Paraphrase variant generator for brittleness testing.

PAT-049 (Matthew 7:24-27 — Two Builders):
Three storm vectors stress-test the prompt:
  - Rain (lexical): synonym substitution
  - Streams (syntactic): structural transformation
  - Wind (semantic): full meaning reformulation via T5/LLM
"""
from __future__ import annotations

import re
from typing import Callable, Optional
from .models import ParaphraseVariant, ParaphraseLevel


class BrittlenessEngine:
    """
    Generates and validates paraphrase variants for brittleness testing.

    Accepts an optional ``similarity_fn`` to bypass ML model loading in tests:
        engine = BrittlenessEngine(similarity_fn=lambda a, b: 0.85)
    """

    def __init__(
        self,
        variants_per_input: int = 8,
        levels: list[ParaphraseLevel] = None,
        min_similarity: float = 0.75,
        max_similarity: float = 0.98,
        paraphrase_model: str = "ramsrigouthamg/t5_paraphraser",
        similarity_model: str = "all-MiniLM-L6-v2",
        similarity_fn: Optional[Callable[[str, str], float]] = None,
    ):
        self.variants_per_input = variants_per_input
        self.levels = levels or ["lexical", "semantic"]
        self.min_similarity = min_similarity
        self.max_similarity = max_similarity
        self._paraphrase_model = None
        self._paraphrase_tokenizer = None
        self._similarity_model = None
        self._paraphrase_model_name = paraphrase_model
        self._similarity_model_name = similarity_model
        self._similarity_fn = similarity_fn  # injection point for tests

    def _load_models(self):
        """Lazy load models on first use. Skipped if similarity_fn is injected."""
        if self._similarity_fn is not None:
            return  # test injection bypasses model loading

        if self._similarity_model is None:
            from sentence_transformers import SentenceTransformer
            self._similarity_model = SentenceTransformer(self._similarity_model_name)

        if self._paraphrase_model is None and "t5" in self._paraphrase_model_name.lower():
            from transformers import T5ForConditionalGeneration, T5Tokenizer
            self._paraphrase_model = T5ForConditionalGeneration.from_pretrained(
                self._paraphrase_model_name
            )
            self._paraphrase_tokenizer = T5Tokenizer.from_pretrained(
                self._paraphrase_model_name
            )

    def generate_variants(self, text: str) -> list[ParaphraseVariant]:
        """Generate all requested paraphrase variants for a single input."""
        self._load_models()
        variants = []

        variants_per_level = max(1, self.variants_per_input // len(self.levels))

        for level in self.levels:
            if level == "lexical":
                raw = self._generate_lexical(text, variants_per_level)
            elif level == "syntactic":
                raw = self._generate_syntactic(text, variants_per_level)
            elif level == "semantic":
                raw = self._generate_semantic(text, variants_per_level)
            else:
                raw = []

            for candidate in raw:
                similarity = self._compute_similarity(text, candidate)
                if (self.min_similarity <= similarity <= self.max_similarity
                        and candidate.strip() != text.strip()):
                    variants.append(ParaphraseVariant(
                        original=text,
                        variant=candidate,
                        level=level,
                        similarity_score=similarity,
                        validated=True
                    ))

        return variants

    def _generate_semantic(self, text: str, n: int) -> list[str]:
        """
        Generate semantic paraphrases using T5 paraphraser.
        Falls back to rule-based if T5 model not loaded.
        Corresponds to the 'wind' stress vector in Matthew 7.
        """
        if self._paraphrase_model is None:
            return self._generate_semantic_fallback(text, n)

        import torch
        input_ids = self._paraphrase_tokenizer.encode(
            f"paraphrase: {text} </s>",
            return_tensors="pt",
            max_length=256,
            truncation=True
        )
        with torch.no_grad():
            outputs = self._paraphrase_model.generate(
                input_ids,
                max_length=256,
                num_beams=n * 2,
                num_return_sequences=n,
                temperature=1.5,
                early_stopping=True
            )
        return [
            self._paraphrase_tokenizer.decode(o, skip_special_tokens=True)
            for o in outputs
        ]

    def _generate_lexical(self, text: str, n: int) -> list[str]:
        """
        Generate lexical variants via synonym substitution.
        Corresponds to the 'rain' stress vector in Matthew 7.
        """
        try:
            import nltk
            from nltk.corpus import wordnet
            nltk.download("wordnet", quiet=True)
            nltk.download("averaged_perceptron_tagger", quiet=True)
        except ImportError:
            return self._generate_semantic_fallback(text, n)

        import random
        words = text.split()
        variants = []
        for _ in range(n):
            new_words = []
            for word in words:
                synsets = wordnet.synsets(word)
                synonyms = set()
                for syn in synsets:
                    for lemma in syn.lemmas():
                        synonym = lemma.name().replace("_", " ")
                        if synonym.lower() != word.lower():
                            synonyms.add(synonym)
                if synonyms and len(word) > 3:
                    new_words.append(random.choice(list(synonyms)))
                else:
                    new_words.append(word)
            variants.append(" ".join(new_words))
        return list(set(variants))

    def _generate_syntactic(self, text: str, n: int) -> list[str]:
        """
        Generate syntactic variants via structural transformation.
        Corresponds to the 'streams' stress vector in Matthew 7.
        v0.1: Rule-based transformations. v0.2: dependency parsing.
        """
        variants = []
        # Contraction expansion/contraction
        contractions = {
            "what's": "what is", "what is": "what's",
            "i'm": "i am", "i am": "i'm",
            "can't": "cannot", "cannot": "can't",
            "don't": "do not", "do not": "don't",
            "how's": "how is", "isn't": "is not",
        }
        variant = text
        for k, v in contractions.items():
            if k.lower() in variant.lower():
                variant = re.sub(re.escape(k), v, variant, flags=re.IGNORECASE)
                break
        if variant != text:
            variants.append(variant)

        # Question restructuring
        if text.lower().startswith("what is"):
            variants.append(text.replace("What is", "Tell me", 1).replace("what is", "tell me", 1))
        if text.lower().startswith("how do i"):
            variants.append(text.replace("How do I", "What is the way to", 1)
                             .replace("how do i", "what is the way to", 1))

        # Pad to n with semantic fallback
        while len(variants) < n:
            variants.append(text)  # will be filtered as near-duplicate
        return variants[:n]

    def _generate_semantic_fallback(self, text: str, n: int) -> list[str]:
        """Fallback: simple question starters for short texts."""
        starters = [
            "Could you tell me",
            "I'd like to know",
            "Can you show me",
            "Please tell me",
            "I need to know",
            "What would be",
        ]
        variants = []
        for starter in starters[:n]:
            if "?" in text:
                core = text.replace("What is ", "").replace("?", "").strip()
                variants.append(f"{starter} {core.lower()}?")
            else:
                variants.append(f"{starter} about: {text.lower()}")
        return variants

    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity between two texts (0.0 to 1.0)."""
        if self._similarity_fn is not None:
            return self._similarity_fn(text1, text2)

        from sentence_transformers import util
        embeddings = self._similarity_model.encode([text1, text2])
        return float(util.cos_sim(embeddings[0], embeddings[1]))
