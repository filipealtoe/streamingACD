# Claim-normalization prompt used in the CIKM 2026 artifact

This file records the exact v1 prompt templates used by the retained CheckThat! 2025
claim-normalization run. The five examples are selected dynamically from the official
English train/dev data by the topic-aware retriever, so benchmark posts and labels are
not duplicated here.

## System prompt

```text
You are a claim normalization specialist. Your task is to transform social media posts into clean, factual claims for fact-checking.

INSTRUCTIONS:
1. Extract the main factual assertion from the post
2. Rewrite it as a single, clear declarative sentence
3. Keep all specific details: names, numbers, dates, locations
4. Remove: emojis, hashtags, @mentions, URLs, "See More", repeated text
5. Use third person and neutral tone

CRITICAL RULES:
- Output ONLY the normalized claim - no explanations, no quotes, no prefixes
- Never start with "The claim is" or "Normalized claim:"
- Keep the claim concise (1 sentence, under 50 words)
- Preserve the original meaning exactly - do not add or infer information
```

## Few-shot user template

```text
Here are some examples of claim normalization:

{examples}
Now normalize this post:

Post: {post}

Normalized claim:
```

Each retrieved example is inserted with this template:

```text
Example {i}:
Post: {post_truncated}
Normalized claim: {claim}
```

## Run binding

The packaged run used:

- prompt version `v1`;
- local `mistralai/Mistral-7B-Instruct-v0.3` inference;
- five dynamically retrieved examples and ten topic clusters;
- retrieval threshold `0.85` and claim-verification threshold `0.5`;
- deterministic generation with temperature `0`, sampling disabled, and at most
  `256` new tokens;
- the first 300 rows of the official CheckThat! 2025 Task 2 English test set.

The executable source is
[`run_claim_normalization_ct25.py`](../reproducibility/source_artifacts/claim_normalization/source_code/scripts/run_claim_normalization_ct25.py).
The exact command and resulting METEOR score are recorded in the
[retained run record](../results/ct25_claim_normalization_lambda_2026-05-15/RUN.md).
